use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use tempfile::TempDir;

struct Sidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Sidecar {
    fn start(directory: &TempDir) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_opencode-zeromem-sidecar"))
            .args([
                "--db",
                directory.path().join("memory.db").to_str().unwrap(),
                "--cache",
                directory.path().join("models").to_str().unwrap(),
            ])
            .env("OPENCODE_ZEROMEM_EMBEDDER", "hash")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    fn request(&mut self, command: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        writeln!(
            self.stdin,
            "{}",
            json!({ "id": id, "command": command, "params": params })
        )
        .unwrap();
        self.stdin.flush().unwrap();
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["id"], id);
        assert_eq!(response["ok"], true, "{response}");
        response["result"].clone()
    }

    fn shutdown(mut self) {
        self.request("shutdown", json!({}));
        assert!(self.child.wait().unwrap().success());
    }
}

#[test]
fn protocol_persists_deduplicates_excludes_and_deletes() {
    let directory = TempDir::new().unwrap();
    let mut sidecar = Sidecar::start(&directory);

    let first = sidecar.request(
        "ingest",
        json!({
            "identity": "turn-a",
            "session_id": "old-session",
            "speaker": "user",
            "text": "O banco principal do projeto Atlas usa PostgreSQL.",
            "ts": 100
        }),
    );
    assert_eq!(first["ingested"], true);
    let duplicate = sidecar.request(
        "ingest",
        json!({
            "identity": "turn-a",
            "session_id": "old-session",
            "speaker": "user",
            "text": "O banco principal do projeto Atlas usa PostgreSQL.",
            "ts": 100
        }),
    );
    assert_eq!(duplicate["ingested"], false);
    sidecar.request(
        "ingest",
        json!({
            "identity": "turn-b",
            "session_id": "current-session",
            "speaker": "assistant",
            "text": "PostgreSQL também apareceu na conversa atual.",
            "ts": 200
        }),
    );

    let query = sidecar.request(
        "query",
        json!({
            "query": "Qual banco o Atlas usa? PostgreSQL?",
            "top_k": 5,
            "exclude_session_id": "current-session"
        }),
    );
    assert_eq!(query["evidence"].as_array().unwrap().len(), 1);
    assert_eq!(query["evidence"][0]["session_id"], "old-session");
    sidecar.shutdown();

    let mut restarted = Sidecar::start(&directory);
    let stats = restarted.request("stats", json!({}));
    assert_eq!(stats["turns"], 2);
    assert_eq!(stats["sessions"], 2);
    let deletion = restarted.request("delete_session", json!({ "session_id": "old-session" }));
    assert_eq!(deletion["deleted"], 1);
    let stats = restarted.request("stats", json!({}));
    assert_eq!(stats["turns"], 1);
    restarted.shutdown();
}
