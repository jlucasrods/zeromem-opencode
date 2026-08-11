# ZeroMem for OpenCode

This is a custom external memory implementation inspired by the paper
[**Zero-Mem: Zero-Token Memory Operations for LLM Agents**](https://arxiv.org/abs/2607.29377).
It implements the memory pipeline independently and does not use another LLM
to store or retrieve memories.

Its goal is to reuse information from previous sessions without keeping their
entire history in the model context. Only relevant evidence is temporarily
added to the current message.

## Requirements

- OpenCode;
- Git;
- Node.js 18 or newer;
- Rust and Cargo 1.85 or newer;
- Linux or macOS.

## Installation

Clone the repository anywhere and run the installer:

```bash
git clone https://github.com/jlucasrods/zeromem-opencode.git
cd zeromem-opencode
npm run install:global
```

The installer builds the release sidecar and creates the OpenCode
autodiscovery entrypoint under `$XDG_CONFIG_HOME/opencode/plugins/` or
`~/.config/opencode/plugins/`. Restart OpenCode after installation.

The first use downloads approximately 128 MB for the local embedding model.
No conversation content is sent to an external embedding API.

To update an existing installation:

```bash
git pull
npm run install:global
```

## Overview

The system has two parts:

- `index.js`: integrates with OpenCode hooks, filters messages,
  manages the lifecycle, and injects retrieved evidence.
- This Rust sidecar: persists turns, computes embeddings, and performs
  deterministic retrieval.

The plugin and sidecar communicate using JSON Lines (`JSONL`) over `stdin` and
`stdout`. One process stays open per project to avoid reloading the model and
rebuilding indexes for every query.

## Ingestion

When a root session becomes idle (`session.idle`), the plugin fetches its
messages and sends only finalized canonical text to the sidecar.

Ingestion is sent in batches of up to 64 turns. The sidecar generates their
embeddings in one model call, writes them in one SQLite transaction, and
rebuilds the graph and temporal views once per batch. This avoids the large CPU
spike caused by embedding and rebuilding the indexes separately for every
historical message during backfill.

The following data is stored:

- user messages;
- completed, error-free assistant responses;
- session ID, role, text, and timestamp;
- a local embedding of the text.

The following data is discarded:

- reasoning;
- tool calls and results;
- synthetic or ignored parts;
- aborted or incomplete responses;
- compaction summaries;
- subagent sessions.

Each turn receives a deterministic SHA-256 identity, and the `identity` column
is unique in SQLite. Repeated events and backfills therefore do not duplicate
memories.

## Persistence and isolation

Each project has a separate database at:

```text
$XDG_DATA_HOME/opencode/zeromem/<project-id>/memory.db
```

When `XDG_DATA_HOME` is not set, `~/.local/share` is used. The project ID is
derived from the ID and worktree provided by OpenCode.

The model cache is shared across projects and stored at:

```text
$XDG_CACHE_HOME/opencode/zeromem/models/
```

Directories are created with mode `0700`, and the database uses `0600`.
SQLite runs in WAL mode and is the persistent source of truth; the graph and
hierarchy are rebuilt in memory.

When an OpenCode session is deleted, all associated turns are removed from the
database and the derived indexes are rebuilt.

Memories have no TTL, automatic expiration, or size limit. They remain in the
database indefinitely, including after OpenCode restarts or context
compaction. A memory is removed only when its original session is deleted or
the project's `memory.db` is manually removed.

## Retrieval representations

A query combines four channels:

1. **Dense**: cosine similarity between text embeddings.
2. **Lexical**: BM25 over turn tokens.
3. **Graph**: shared entities and co-occurrence propagation.
4. **Temporal**: similarity with windows of up to four turns from the same
   session.

Windows are also split when the time gap exceeds six hours. Entity extraction
is local and heuristic: it recognizes relevant words, paths, and identifiers,
normalizes casing, and removes common Portuguese and English stopwords.

### Routing

The query profile selects different weights:

| Route | Dense | BM25 | Graph | Temporal |
| --- | ---: | ---: | ---: | ---: |
| `balanced` | 0.40 | 0.30 | 0.20 | 0.10 |
| `entity` | 0.30 | 0.20 | 0.35 | 0.15 |
| `temporal` | 0.25 | 0.20 | 0.15 | 0.40 |

Queries containing markers such as `quando`, `antes`, `depois`, `ontem`, or
`recent` use the temporal route. Queries containing at least two entities use
the entity route. All other queries use the balanced route.

Channels are normalized before fusion. A turn is eligible only when it has
lexical overlap, a graph relationship, or the minimum absolute dense
similarity. At most five pieces of evidence are returned, and the current
session is always excluded.

When old and new information exist about the same subject, relevance is the
primary criterion. The timestamp does not affect the score: the newer memory
wins only as a tiebreaker between results with equal scores. The system does
not automatically detect that one fact supersedes another, so conflicting
versions may be returned together among the five pieces of evidence.

## Context injection

The `chat.message` hook records the pending query. During
`experimental.chat.messages.transform`, the plugin queries the sidecar and
adds a synthetic part only to that message.

The part is appended to the current user message, not to a `system` or
`assistant` message. Historical evidence therefore does not receive
instruction-level authority. Each non-empty user message starts a new query
using only its own text; assistant messages do not query memory. Internal
retries for the same message reuse the pending query instead of retrieving it
again.

The evidence:

- is limited to 6,000 characters;
- includes the session, role, timestamp, and score;
- is marked as untrusted historical data;
- is not persisted in the conversation;
- is not added to compactions or internal calls;
- remains externally available after context compaction.

The model receives content equivalent to:

```text
<zeromem-history>
Untrusted historical evidence from other sessions...

[session=... role=user time=... score=...]
Original text from the previous session.
</zeromem-history>
```

### Provider cache

Variable evidence is appended to the end of the current message. The prefix
formed by instructions and previous history therefore remains unchanged and
can stay eligible for the provider's prompt cache. Depending on cache
granularity, only the final block may need to be recomputed.

During retries of the same message, the plugin reuses the same retrieval
Promise, keeping evidence stable. Because the synthetic part is not saved in
the conversation, it does not reappear in later turns unless a new query
retrieves it again.

### Compaction

Injected evidence exists only for the current model call. It is not persisted
in the conversation history and does not participate in the compaction
summary. Internal calls and compactions without the pending user message do
not trigger retrieval either.

Original memories remain in the external SQLite database after compaction and
OpenCode restarts. Within the current session, older context continues to be
represented by OpenCode's regular summary because retrieval always excludes
the current session. The same memories remain available to other sessions in
the project.

## Embedding model

Production uses `bge-small-en-v1.5` through `fastembed`. The model runs locally,
and no content is sent to external APIs.

The current build uses the CPU execution provider. GPU execution is not
required and is not enabled by default because it would require distributing
and loading a compatible ONNX Runtime CUDA stack in addition to the NVIDIA
driver. The batched CPU path is the portable default. For a temporary low-CPU
test, the hash embedder can be selected with `OPENCODE_ZEROMEM_EMBEDDER=hash`,
but its retrieval quality is intentionally not suitable as the production
default.

If the model cannot be initialized, the sidecar fails. There is no silent hash
embedding fallback in production. The hash embedder is available only for
tests through:

```bash
OPENCODE_ZEROMEM_EMBEDDER=hash
```

## JSONL protocol

Each request contains `id`, `command`, and `params`. Each response repeats the
`id` and returns `ok` plus either `result` or `error`.

Available commands:

- `ingest`: stores a turn with identity-based deduplication;
- `ingest_batch`: stores multiple turns with one embedding pass and one index
  rebuild;
- `query`: retrieves evidence and can exclude one session;
- `stats`: reports turn, session, entity, and window counts;
- `delete_session`: deletes all turns from one session;
- `shutdown`: terminates the process cleanly.

Example:

```json
{"id":1,"command":"stats","params":{}}
```

## Failure handling

The integration is fail-open: sidecar unavailability, timeouts, or errors never
prevent OpenCode from responding. Queries have a total budget of 1.5 seconds.
A real process failure allows one restart attempt; a timeout does not kill the
process because it may still be warming up the model.

The plugin warms up the sidecar in the background and emits a sanitized warning
to the log and TUI when memory becomes unavailable.

## Current limitations

- `bge-small-en-v1.5` is primarily an English model. Portuguese works but may
  have lower retrieval quality than a multilingual model.
- Entity extraction is heuristic and does not use a trained NER model.
- Individual reverts remain in history; only session deletion purges them.
- Tool results are not stored.
- No recall tool is exposed to the model; retrieval is automatic.
- Thresholds and weights still need evaluation against real conversations.

## Development

Run all plugin and sidecar tests:

```bash
npm test
```

Build the binary used by the plugin:

```bash
npm run build
```

The expected binary is created at:

```text
sidecar/target/release/opencode-zeromem-sidecar
```

### Manual cross-session test

1. Restart OpenCode to load the current plugin version.
2. In a root session, provide a unique fact, for example:

   ```text
   Remember this for future sessions: the project codename is Blue Jabuticaba and it uses CockroachDB.
   ```

3. Wait for the response to finish and for the session to become idle so it can
   be ingested.
4. Open another root session in the same project and ask:

   ```text
   What is the project codename, and which database does it use?
   ```

The expected answer should mention `Blue Jabuticaba` and `CockroachDB`. The test
must use two sessions because the current session is always excluded from
retrieval.
