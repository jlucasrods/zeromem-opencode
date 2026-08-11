use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use regex::Regex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

const WINDOW_SIZE: usize = 4;
const EPISODE_GAP_SECONDS: i64 = 6 * 60 * 60;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Database(#[from] rusqlite::Error),
    #[error("embedding failed: {0}")]
    Embedding(String),
    #[error("turn text cannot be empty")]
    EmptyTurn,
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait Embedder: Send + Sync {
    fn id(&self) -> &str;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

pub struct FastEmbedder {
    model: Mutex<TextEmbedding>,
}

impl FastEmbedder {
    pub fn open(cache_dir: &Path) -> Result<Self> {
        let options =
            InitOptions::new(EmbeddingModel::BGESmallENV15).with_cache_dir(cache_dir.to_path_buf());
        let model =
            TextEmbedding::try_new(options).map_err(|error| Error::Embedding(error.to_string()))?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }
}

impl Embedder for FastEmbedder {
    fn id(&self) -> &str {
        "bge-small-en-v1.5"
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut model = self
            .model
            .lock()
            .map_err(|error| Error::Embedding(error.to_string()))?;
        let mut vectors = model
            .embed(texts.to_vec(), None)
            .map_err(|error| Error::Embedding(error.to_string()))?;
        for vector in &mut vectors {
            normalize(vector);
        }
        Ok(vectors)
    }
}

pub struct HashEmbedder;

impl Embedder for HashEmbedder {
    fn id(&self) -> &str {
        "test-hash-v1"
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut vector = vec![0.0; 256];
                for token in tokenize(text) {
                    let mut hash = 0xcbf29ce484222325_u64;
                    for byte in token.bytes() {
                        hash ^= u64::from(byte);
                        hash = hash.wrapping_mul(0x100000001b3);
                    }
                    let index = hash as usize % vector.len();
                    vector[index] += if hash & (1 << 32) == 0 { 1.0 } else { -1.0 };
                }
                normalize(&mut vector);
                vector
            })
            .collect())
    }
}

#[derive(Deserialize)]
pub struct IngestTurn {
    pub identity: String,
    pub session_id: String,
    pub speaker: String,
    pub text: String,
    pub ts: i64,
}

#[derive(Clone)]
struct Turn {
    id: i64,
    session_id: String,
    speaker: String,
    text: String,
    ts: i64,
    embedding: Vec<f32>,
    tokens: Vec<String>,
    entities: Vec<String>,
}

struct Window {
    turns: Vec<usize>,
    centroid: Vec<f32>,
}

#[derive(Serialize)]
pub struct Evidence {
    pub turn_id: i64,
    pub session_id: String,
    pub speaker: String,
    pub text: String,
    pub ts: i64,
    pub score: f32,
}

#[derive(Serialize)]
pub struct QueryResult {
    pub route: &'static str,
    pub evidence: Vec<Evidence>,
}

#[derive(Serialize)]
pub struct Stats {
    pub turns: usize,
    pub sessions: usize,
    pub entities: usize,
    pub windows: usize,
    pub embedder: String,
}

pub struct MemoryStore {
    connection: Connection,
    embedder: Box<dyn Embedder>,
    turns: Vec<Turn>,
    entity_turns: HashMap<String, Vec<usize>>,
    entity_edges: HashMap<String, HashMap<String, usize>>,
    windows: Vec<Window>,
}

impl MemoryStore {
    pub fn open(path: &Path, embedder: Box<dyn Embedder>) -> Result<Self> {
        let connection = Connection::open(path)?;
        connection.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY,
                identity TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                ts INTEGER NOT NULL,
                embedding BLOB
             );
             CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, id);
             CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
             );",
        )?;

        let previous_embedder: Option<String> = connection
            .query_row("SELECT value FROM meta WHERE key = 'embedder'", [], |row| {
                row.get(0)
            })
            .optional()?;
        if previous_embedder.as_deref() != Some(embedder.id()) {
            connection.execute("UPDATE turns SET embedding = NULL", [])?;
            connection.execute(
                "INSERT INTO meta(key, value) VALUES ('embedder', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [embedder.id()],
            )?;
        }

        let mut store = Self {
            connection,
            embedder,
            turns: Vec::new(),
            entity_turns: HashMap::new(),
            entity_edges: HashMap::new(),
            windows: Vec::new(),
        };
        store.reload()?;
        Ok(store)
    }

    pub fn ingest(&mut self, turn: &IngestTurn) -> Result<(bool, i64)> {
        let mut results = self.ingest_batch(std::slice::from_ref(turn))?;
        Ok(results.remove(0))
    }

    pub fn ingest_batch(&mut self, turns: &[IngestTurn]) -> Result<Vec<(bool, i64)>> {
        let mut results = vec![None; turns.len()];
        let mut duplicate_of = vec![None; turns.len()];
        let mut pending = HashMap::new();
        let mut candidates = Vec::new();

        for (index, turn) in turns.iter().enumerate() {
            if turn.text.trim().is_empty() {
                return Err(Error::EmptyTurn);
            }
            let existing: Option<i64> = self
                .connection
                .query_row(
                    "SELECT id FROM turns WHERE identity = ?1",
                    [&turn.identity],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(id) = existing {
                results[index] = Some((false, id));
            } else if let Some(candidate) = pending.get(&turn.identity) {
                duplicate_of[index] = Some(*candidate);
            } else {
                pending.insert(turn.identity.clone(), index);
                candidates.push((index, turn));
            }
        }

        if candidates.is_empty() {
            return Ok(results.into_iter().map(Option::unwrap).collect());
        }

        let texts: Vec<&str> = candidates
            .iter()
            .map(|(_, turn)| turn.text.as_str())
            .collect();
        let embeddings = self.embedder.embed(&texts)?;
        if embeddings.len() != candidates.len() {
            return Err(Error::Embedding(
                "embedder returned an unexpected number of vectors".into(),
            ));
        }

        let transaction = self.connection.transaction()?;
        for ((index, turn), embedding) in candidates.iter().zip(embeddings) {
            transaction.execute(
                "INSERT INTO turns(identity, session_id, speaker, text, ts, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    turn.identity,
                    turn.session_id,
                    turn.speaker,
                    turn.text,
                    turn.ts,
                    vector_to_bytes(&embedding)
                ],
            )?;
            results[*index] = Some((true, transaction.last_insert_rowid()));
        }
        transaction.commit()?;

        for (index, candidate) in duplicate_of.into_iter().enumerate() {
            if let Some(candidate) = candidate {
                results[index] = results[candidate].map(|(_, id)| (false, id));
            }
        }
        self.reload()?;
        Ok(results.into_iter().map(Option::unwrap).collect())
    }

    pub fn query(
        &self,
        query: &str,
        top_k: usize,
        excluded_session: Option<&str>,
    ) -> Result<QueryResult> {
        if query.trim().is_empty() || top_k == 0 || self.turns.is_empty() {
            return Ok(QueryResult {
                route: "balanced",
                evidence: Vec::new(),
            });
        }

        let query_embedding = self.embedder.embed(&[query])?.remove(0);
        let query_tokens = tokenize(query);
        let query_entities = extract_entities(query);
        let temporal = is_temporal_query(query);
        let route = if temporal {
            "temporal"
        } else if query_entities.len() >= 2 {
            "entity"
        } else {
            "balanced"
        };

        let dense: Vec<f32> = self
            .turns
            .iter()
            .map(|turn| cosine(&query_embedding, &turn.embedding))
            .collect();
        let lexical = self.bm25_scores(&query_tokens);
        let graph = self.graph_scores(&query_entities);
        let hierarchy = self.hierarchy_scores(&query_embedding);
        let relevant: Vec<bool> = dense
            .iter()
            .zip(&lexical)
            .zip(&graph)
            .map(|((dense, lexical), graph)| *dense >= 0.35 || *lexical > 0.0 || *graph > 0.0)
            .collect();
        let weights = match route {
            "temporal" => [0.25, 0.20, 0.15, 0.40],
            "entity" => [0.30, 0.20, 0.35, 0.15],
            _ => [0.40, 0.30, 0.20, 0.10],
        };
        let channels = [
            scale_scores(dense),
            scale_scores(lexical),
            scale_scores(graph),
            scale_scores(hierarchy),
        ];

        let mut ranked: Vec<(usize, f32)> = (0..self.turns.len())
            .filter(|index| {
                relevant[*index] && excluded_session != Some(self.turns[*index].session_id.as_str())
            })
            .map(|index| {
                let score = channels
                    .iter()
                    .zip(weights)
                    .map(|(channel, weight)| channel[index] * weight)
                    .sum();
                (index, score)
            })
            .collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| self.turns[right.0].ts.cmp(&self.turns[left.0].ts))
                .then_with(|| self.turns[left.0].id.cmp(&self.turns[right.0].id))
        });

        let evidence = ranked
            .into_iter()
            .take(top_k)
            .map(|(index, score)| {
                let turn = &self.turns[index];
                Evidence {
                    turn_id: turn.id,
                    session_id: turn.session_id.clone(),
                    speaker: turn.speaker.clone(),
                    text: turn.text.clone(),
                    ts: turn.ts,
                    score,
                }
            })
            .collect();
        Ok(QueryResult { route, evidence })
    }

    pub fn delete_session(&mut self, session_id: &str) -> Result<usize> {
        let deleted = self
            .connection
            .execute("DELETE FROM turns WHERE session_id = ?1", [session_id])?;
        self.reload()?;
        Ok(deleted)
    }

    pub fn stats(&self) -> Stats {
        Stats {
            turns: self.turns.len(),
            sessions: self
                .turns
                .iter()
                .map(|turn| turn.session_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
            entities: self.entity_turns.len(),
            windows: self.windows.len(),
            embedder: self.embedder.id().to_string(),
        }
    }

    fn reload(&mut self) -> Result<()> {
        let rows = {
            let mut statement = self.connection.prepare(
                "SELECT id, session_id, speaker, text, ts, embedding
                 FROM turns ORDER BY id",
            )?;
            let mapped = statement.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, Option<Vec<u8>>>(5)?,
                ))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, _>>()?
        };

        self.turns.clear();
        for (id, session_id, speaker, text, ts, stored_embedding) in rows {
            let embedding = if let Some(bytes) = stored_embedding {
                bytes_to_vector(&bytes)
            } else {
                let generated = self.embedder.embed(&[&text])?.remove(0);
                self.connection.execute(
                    "UPDATE turns SET embedding = ?2 WHERE id = ?1",
                    params![id, vector_to_bytes(&generated)],
                )?;
                generated
            };
            self.turns.push(Turn {
                id,
                session_id,
                speaker,
                tokens: tokenize(&text),
                entities: extract_entities(&text),
                text,
                ts,
                embedding,
            });
        }
        self.rebuild_views();
        Ok(())
    }

    fn rebuild_views(&mut self) {
        self.entity_turns.clear();
        self.entity_edges.clear();
        self.windows.clear();

        for (index, turn) in self.turns.iter().enumerate() {
            for entity in &turn.entities {
                self.entity_turns
                    .entry(entity.clone())
                    .or_default()
                    .push(index);
            }
            for left in 0..turn.entities.len() {
                for right in (left + 1)..turn.entities.len() {
                    *self
                        .entity_edges
                        .entry(turn.entities[left].clone())
                        .or_default()
                        .entry(turn.entities[right].clone())
                        .or_default() += 1;
                    *self
                        .entity_edges
                        .entry(turn.entities[right].clone())
                        .or_default()
                        .entry(turn.entities[left].clone())
                        .or_default() += 1;
                }
            }
        }

        let mut sessions: HashMap<String, Vec<usize>> = HashMap::new();
        for (index, turn) in self.turns.iter().enumerate() {
            sessions
                .entry(turn.session_id.clone())
                .or_default()
                .push(index);
        }
        for indices in sessions.into_values() {
            let mut current = Vec::new();
            for index in indices {
                let starts_episode = current.last().is_some_and(|previous: &usize| {
                    self.turns[index].ts - self.turns[*previous].ts > EPISODE_GAP_SECONDS
                });
                if current.len() == WINDOW_SIZE || starts_episode {
                    self.push_window(&current);
                    current.clear();
                }
                current.push(index);
            }
            self.push_window(&current);
        }
    }

    fn push_window(&mut self, turns: &[usize]) {
        if turns.is_empty() {
            return;
        }
        let dimension = self.turns[turns[0]].embedding.len();
        let mut centroid = vec![0.0; dimension];
        for index in turns {
            for (target, value) in centroid.iter_mut().zip(&self.turns[*index].embedding) {
                *target += value;
            }
        }
        normalize(&mut centroid);
        self.windows.push(Window {
            turns: turns.to_vec(),
            centroid,
        });
    }

    fn bm25_scores(&self, query: &[String]) -> Vec<f32> {
        if query.is_empty() {
            return vec![0.0; self.turns.len()];
        }
        let average_length = self
            .turns
            .iter()
            .map(|turn| turn.tokens.len())
            .sum::<usize>() as f32
            / self.turns.len().max(1) as f32;
        let mut document_frequency = HashMap::new();
        for token in query {
            let count = self
                .turns
                .iter()
                .filter(|turn| turn.tokens.contains(token))
                .count();
            document_frequency.insert(token, count);
        }
        self.turns
            .iter()
            .map(|turn| {
                let frequencies = turn
                    .tokens
                    .iter()
                    .fold(HashMap::new(), |mut counts, token| {
                        *counts.entry(token).or_insert(0_usize) += 1;
                        counts
                    });
                query
                    .iter()
                    .map(|token| {
                        let frequency = *frequencies.get(token).unwrap_or(&0) as f32;
                        let documents = *document_frequency.get(token).unwrap_or(&0) as f32;
                        let inverse =
                            ((self.turns.len() as f32 - documents + 0.5) / (documents + 0.5) + 1.0)
                                .ln();
                        let denominator = frequency
                            + 1.2
                                * (0.25
                                    + 0.75 * turn.tokens.len() as f32 / average_length.max(1.0));
                        inverse * frequency * 2.2 / denominator.max(f32::EPSILON)
                    })
                    .sum()
            })
            .collect()
    }

    fn graph_scores(&self, query_entities: &[String]) -> Vec<f32> {
        let mut scores = vec![0.0; self.turns.len()];
        for entity in query_entities {
            if let Some(turns) = self.entity_turns.get(entity) {
                for index in turns {
                    scores[*index] += 1.0;
                }
            }
            if let Some(neighbors) = self.entity_edges.get(entity) {
                for (neighbor, edge_weight) in neighbors {
                    if let Some(turns) = self.entity_turns.get(neighbor) {
                        let propagated = 0.35 * (*edge_weight as f32).ln_1p();
                        for index in turns {
                            scores[*index] += propagated;
                        }
                    }
                }
            }
        }
        scores
    }

    fn hierarchy_scores(&self, query_embedding: &[f32]) -> Vec<f32> {
        let mut scores = vec![0.0; self.turns.len()];
        for window in &self.windows {
            let score = cosine(query_embedding, &window.centroid);
            for index in &window.turns {
                scores[*index] = score;
            }
        }
        scores
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| {
        !character.is_alphanumeric() && character != '_' && character != '/'
    })
    .filter(|token| token.chars().count() >= 2)
    .map(str::to_lowercase)
    .collect()
}

fn extract_entities(text: &str) -> Vec<String> {
    let matcher = Regex::new(r"(?u)\b[\p{L}_][\p{L}\p{N}_./:-]{2,}\b").expect("valid entity regex");
    let stopwords: HashSet<&str> = [
        "the", "and", "for", "with", "this", "that", "from", "have", "was", "were", "uma", "para",
        "com", "que", "por", "dos", "das", "como", "mais", "não", "nao", "seu", "sua", "isso",
        "esta", "esse", "ser", "tem", "nos", "nas",
    ]
    .into_iter()
    .collect();
    let mut seen = HashSet::new();
    matcher
        .find_iter(text)
        .map(|matched| matched.as_str().to_lowercase())
        .filter(|entity| !stopwords.contains(entity.as_str()) && seen.insert(entity.clone()))
        .take(32)
        .collect()
}

fn is_temporal_query(query: &str) -> bool {
    let normalized = query.to_lowercase();
    [
        "quando",
        "antes",
        "depois",
        "últim",
        "ultimo",
        "ontem",
        "hoje",
        "recent",
        "previous",
        "last time",
        "earlier",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn scale_scores(mut scores: Vec<f32>) -> Vec<f32> {
    let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !min.is_finite() || max - min <= f32::EPSILON {
        let value = if max > 0.0 { 1.0 } else { 0.0 };
        return vec![value; scores.len()];
    }
    for score in &mut scores {
        *score = (*score - min) / (max - min);
    }
    scores
}

fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_extraction_is_unique_and_filters_common_words() {
        assert_eq!(
            extract_entities("OpenCode usa SQLite e OpenCode com memória"),
            vec!["opencode", "usa", "sqlite", "memória"]
        );
    }

    #[test]
    fn score_scaling_handles_constant_channels() {
        assert_eq!(scale_scores(vec![2.0, 2.0]), vec![1.0, 1.0]);
        assert_eq!(scale_scores(vec![0.0, 0.0]), vec![0.0, 0.0]);
        assert_eq!(scale_scores(vec![1.0, 3.0]), vec![0.0, 1.0]);
    }
}
