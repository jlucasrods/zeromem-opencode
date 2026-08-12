mod memory;

use memory::{Embedder, FastEmbedder, HashEmbedder, IngestTurn, MemoryStore};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

type AnyError = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, AnyError>;
const DEFAULT_MAX_CPUS: usize = 4;

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
struct BatchIngestParams {
    turns: Vec<IngestTurn>,
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

#[derive(Deserialize)]
struct BackfillLeaseParams {
    owner: String,
    #[serde(default)]
    lease_seconds: i64,
}

fn default_top_k() -> usize {
    5
}

fn handle(memory: &mut MemoryStore, request: Request) -> (Response, bool) {
    let result: Result<Value> = match request.command.as_str() {
        "ingest" => serde_json::from_value::<IngestTurn>(request.params)
            .map_err(Into::into)
            .and_then(|params| {
                let (ingested, turn_id) = memory.ingest(&params)?;
                Ok(json!({ "ingested": ingested, "turn_id": turn_id }))
            }),
        "ingest_batch" => serde_json::from_value::<BatchIngestParams>(request.params)
            .map_err(Into::into)
            .and_then(|params| {
                let results = memory.ingest_batch(&params.turns)?;
                let ingested = results.iter().filter(|(ingested, _)| *ingested).count();
                Ok(json!({
                    "ingested": ingested,
                    "skipped": results.len() - ingested,
                }))
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
        "acquire_backfill" => serde_json::from_value::<BackfillLeaseParams>(request.params)
            .map_err(Into::into)
            .and_then(|params| {
                if params.owner.is_empty() || params.lease_seconds <= 0 {
                    return Err("owner and a positive lease_seconds are required".into());
                }
                let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
                Ok(json!({
                    "acquired": memory.acquire_backfill(
                        &params.owner,
                        now,
                        params.lease_seconds,
                    )?
                }))
            }),
        "release_backfill" => serde_json::from_value::<BackfillLeaseParams>(request.params)
            .map_err(Into::into)
            .and_then(|params| {
                Ok(json!({
                    "released": memory.release_backfill(&params.owner)?
                }))
            }),
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

#[cfg(target_os = "linux")]
fn limit_cpu_affinity(max_cpus: usize) -> Result<()> {
    let mut available = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let size = std::mem::size_of::<libc::cpu_set_t>();
    if unsafe { libc::sched_getaffinity(0, size, &mut available) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }

    let mut limited = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let mut selected = 0;
    for cpu in 0..libc::CPU_SETSIZE as usize {
        if unsafe { libc::CPU_ISSET(cpu, &available) } && selected < max_cpus {
            unsafe { libc::CPU_SET(cpu, &mut limited) };
            selected += 1;
        }
    }
    if selected == 0 || unsafe { libc::sched_setaffinity(0, size, &limited) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn limit_cpu_affinity(_max_cpus: usize) -> Result<()> {
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
    let max_cpus = std::env::var("OPENCODE_ZEROMEM_MAX_CPUS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_CPUS);
    limit_cpu_affinity(max_cpus)?;
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
