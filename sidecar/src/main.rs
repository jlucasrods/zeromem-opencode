mod memory;

use memory::{Embedder, FastEmbedder, HashEmbedder, MemoryStore};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

type AnyError = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, AnyError>;

#[derive(Deserialize)]
struct Request {
    id: Value,
    command: String,
    #[serde(default)]
    params: Value,
}

#[derive(Serialize)]
struct Response {
    id: Value,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Deserialize)]
struct IngestParams {
    identity: String,
    session_id: String,
    speaker: String,
    text: String,
    ts: i64,
}

#[derive(Deserialize)]
struct QueryParams {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    exclude_session_id: Option<String>,
}

#[derive(Deserialize)]
struct DeleteSessionParams {
    session_id: String,
}

fn default_top_k() -> usize {
    5
}

fn handle(memory: &mut MemoryStore, request: Request) -> (Response, bool) {
    let result: Result<Value> = match request.command.as_str() {
        "ingest" => serde_json::from_value::<IngestParams>(request.params)
            .map_err(Into::into)
            .and_then(|params| {
                let (ingested, turn_id) = memory.ingest(
                    &params.identity,
                    &params.session_id,
                    &params.speaker,
                    &params.text,
                    params.ts,
                )?;
                Ok(json!({ "ingested": ingested, "turn_id": turn_id }))
            }),
        "query" => serde_json::from_value::<QueryParams>(request.params)
            .map_err(Into::into)
            .and_then(|params| {
                Ok(serde_json::to_value(memory.query(
                    &params.query,
                    params.top_k,
                    params.exclude_session_id.as_deref(),
                )?)?)
            }),
        "stats" => Ok(serde_json::to_value(memory.stats()).expect("stats serialize")),
        "delete_session" => serde_json::from_value::<DeleteSessionParams>(request.params)
            .map_err(Into::into)
            .and_then(|params| {
                Ok(json!({ "deleted": memory.delete_session(&params.session_id)? }))
            }),
        "shutdown" => Ok(json!({ "shutdown": true })),
        _ => Err(format!("unknown command: {}", request.command).into()),
    };
    let shutdown = request.command == "shutdown";
    let response = match result {
        Ok(value) => Response {
            id: request.id,
            ok: true,
            result: Some(value),
            error: None,
        },
        Err(error) => Response {
            id: request.id,
            ok: false,
            result: None,
            error: Some(error.to_string()),
        },
    };
    (response, shutdown)
}

fn parse_args() -> Result<(PathBuf, PathBuf)> {
    let mut args = std::env::args().skip(1);
    let mut db = None;
    let mut cache = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--db" => db = args.next().map(PathBuf::from),
            "--cache" => cache = args.next().map(PathBuf::from),
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    Ok((
        db.ok_or("--db is required")?,
        cache.ok_or("--cache is required")?,
    ))
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) -> Result<()> {
    Ok(())
}

fn run() -> Result<()> {
    #[cfg(unix)]
    unsafe {
        libc::umask(0o077);
    }

    let (db_path, cache_path) = parse_args()?;
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
        set_mode(parent, 0o700)?;
    }
    std::fs::create_dir_all(&cache_path)?;
    set_mode(&cache_path, 0o700)?;
    let embedder: Box<dyn Embedder> =
        if std::env::var("OPENCODE_ZEROMEM_EMBEDDER").as_deref() == Ok("hash") {
            Box::new(HashEmbedder)
        } else {
            Box::new(FastEmbedder::open(&cache_path)?)
        };
    let mut memory = MemoryStore::open(&db_path, embedder)?;
    set_mode(&db_path, 0o600)?;

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line?;
        let parsed = serde_json::from_str::<Request>(&line);
        let (response, shutdown) = match parsed {
            Ok(request) => handle(&mut memory, request),
            Err(error) => (
                Response {
                    id: Value::Null,
                    ok: false,
                    result: None,
                    error: Some(format!("invalid request: {error}")),
                },
                false,
            ),
        };
        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
        if shutdown {
            break;
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("opencode-zeromem-sidecar: {error}");
        std::process::exit(1);
    }
}
