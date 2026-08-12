import assert from "node:assert/strict"
import test from "node:test"
import ZeroMemPlugin from "../index.js"

function waitFor(check, timeout = 500) {
  const deadline = Date.now() + timeout
  return new Promise((resolve, reject) => {
    function poll() {
      if (check()) {
        resolve()
      } else if (Date.now() >= deadline) {
        reject(new Error("condition timed out"))
      } else {
        setTimeout(poll, 5)
      }
    }
    poll()
  })
}

function fixture({ sidecar, sessions = [], messages = new Map() }) {
  const logs = []
  const toasts = []
  const client = {
    app: {
      log: async (request) => {
        logs.push(request)
        return { data: true }
      },
    },
    tui: {
      showToast: async (request) => {
        toasts.push(request)
        return { data: true }
      },
    },
    session: {
      get: async ({ path }) => ({
        data: sessions.find((session) => session.id === path.id),
      }),
      list: async () => ({ data: sessions }),
      messages: async ({ path }) => ({ data: messages.get(path.id) || [] }),
    },
  }
  return {
    input: {
      client,
      project: { id: "project", worktree: "/workspace" },
      directory: "/workspace",
    },
    options: { sidecar, disableBackfill: true, disableWarmup: true },
    logs,
    toasts,
  }
}

function userMessage(id, text, extraPart = {}) {
  return {
    info: {
      id,
      sessionID: "root",
      role: "user",
      time: { created: 1000 },
    },
    parts: [{ type: "text", text, ...extraPart }],
  }
}

function assistantMessage(id, text, overrides = {}) {
  return {
    info: {
      id,
      sessionID: "root",
      role: "assistant",
      time: { created: 2000, completed: 3000 },
      ...overrides,
    },
    parts: [{ type: "text", text }],
  }
}

test("idle ingests only canonical finalized text from root sessions", async () => {
  const calls = []
  const sidecar = {
    request: async (command, params) => {
      calls.push({ command, params })
      if (command === "acquire_backfill") {
        return { acquired: true }
      }
      return { ingested: true }
    },
  }
  const sessions = [{ id: "root", time: { created: 1 } }]
  const messages = new Map([
    ["root", [
      userMessage("u1", "decisão do usuário"),
      userMessage("u2", "texto sintético", { synthetic: true }),
      assistantMessage("a1", "resposta incompleta", { time: { created: 2000 } }),
      assistantMessage("a2", "resposta final"),
      assistantMessage("a3", "resumo", { summary: true }),
    ]],
  ])
  const setup = fixture({ sidecar, sessions, messages })
  const hooks = await ZeroMemPlugin(setup.input, setup.options)

  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "root" } } })
  await waitFor(() => calls.length === 1)

  assert.deepEqual(calls[0].params.turns.map((turn) => turn.text), [
    "decisão do usuário",
    "resposta final",
  ])
  assert.deepEqual(calls[0].params.turns.map((turn) => turn.speaker), ["user", "assistant"])
})

test("retrieval is injected once only for the pending user message", async () => {
  const sidecar = {
    request: async (command) => {
      assert.equal(command, "query")
      return {
        evidence: [{
          session_id: "older",
          speaker: "assistant",
          text: "Use PostgreSQL for Atlas.",
          ts: 100,
          score: 0.9,
        }],
      }
    },
  }
  const setup = fixture({ sidecar })
  const hooks = await ZeroMemPlugin(setup.input, setup.options)
  const message = userMessage("u1", "qual banco usar?")

  await hooks["chat.message"](
    { sessionID: "root", messageID: "u1" },
    { message: message.info, parts: message.parts },
  )
  await hooks["experimental.chat.messages.transform"]({}, { messages: [message] })
  await hooks["experimental.chat.messages.transform"]({}, { messages: [message] })

  assert.equal(message.parts.length, 2)
  assert.equal(message.parts[1].synthetic, true)
  assert.equal(message.parts[1].metadata.zeromem, true)
  assert.match(message.parts[1].text, /Untrusted historical evidence/)
  assert.match(message.parts[1].text, /Use PostgreSQL for Atlas/)
  assert.ok(message.parts[1].text.length <= 6000)
})

test("transform ignores unrelated calls and retrieval failures fail open", async () => {
  const sidecar = {
    request: async () => {
      throw new Error("secret detail\nmodel failed")
    },
  }
  const setup = fixture({ sidecar })
  const hooks = await ZeroMemPlugin(setup.input, setup.options)
  const pending = userMessage("u1", "remember this")
  const unrelated = userMessage("u2", "compaction-like internal call")

  await hooks["chat.message"](
    { sessionID: "root", messageID: "u1" },
    { message: pending.info, parts: pending.parts },
  )
  await hooks["experimental.chat.messages.transform"]({}, { messages: [unrelated] })
  assert.equal(unrelated.parts.length, 1)
  await hooks["experimental.chat.messages.transform"]({}, { messages: [pending] })

  assert.equal(pending.parts.length, 1)
  assert.equal(setup.toasts.length, 1)
  assert.equal(setup.logs.length, 1)
  assert.doesNotMatch(setup.logs[0].body.message, /\n/)
})

test("session deletion purges sidecar memory", async () => {
  const calls = []
  const sidecar = {
    request: async (command, params) => {
      calls.push({ command, params })
      return { deleted: 2 }
    },
  }
  const setup = fixture({ sidecar })
  const hooks = await ZeroMemPlugin(setup.input, setup.options)

  await hooks.event({
    event: {
      type: "session.deleted",
      properties: { info: { id: "removed" } },
    },
  })
  await waitFor(() => calls.length === 1)

  assert.deepEqual(calls[0], {
    command: "delete_session",
    params: { session_id: "removed" },
  })
})

test("backfill processes existing root sessions and skips children", async () => {
  const calls = []
  const sidecar = {
    request: async (command, params) => {
      calls.push({ command, params })
      if (command === "acquire_backfill") {
        return { acquired: true }
      }
      return { ingested: true }
    },
  }
  const sessions = [
    { id: "root", time: { created: 1 } },
    { id: "child", parentID: "root", time: { created: 2 } },
  ]
  const messages = new Map([["root", [userMessage("u1", "memória antiga")]]])
  const setup = fixture({ sidecar, sessions, messages })
  setup.options.disableBackfill = false
  setup.options.backfillDelay = 0

  await ZeroMemPlugin(setup.input, setup.options)
  await waitFor(() => calls.some((call) => call.command === "release_backfill"))

  const ingestion = calls.find((call) => call.command === "ingest_batch")
  assert.equal(ingestion.params.turns[0].session_id, "root")
  assert.equal(calls[0].command, "acquire_backfill")
  assert.equal(calls.at(-1).command, "release_backfill")
})

test("backfill keeps ingestion batches bounded", async () => {
  const calls = []
  const sidecar = {
    request: async (command, params) => {
      calls.push({ command, params })
      if (command === "acquire_backfill") {
        return { acquired: true }
      }
      return { ingested: params.turns?.length || 0 }
    },
  }
  const sessions = [{ id: "root", time: { created: 1 } }]
  const messages = new Map([[
    "root",
    Array.from({ length: 9 }, (_, index) => userMessage(`u${index}`, `memória ${index}`)),
  ]])
  const setup = fixture({ sidecar, sessions, messages })
  setup.options.disableBackfill = false
  setup.options.backfillDelay = 0

  await ZeroMemPlugin(setup.input, setup.options)
  await waitFor(() => calls.some((call) => call.command === "release_backfill"))

  assert.deepEqual(
    calls.filter((call) => call.command === "ingest_batch").map((call) => call.params.turns.length),
    [8, 1],
  )
})

test("backfill is skipped without the database lease", async () => {
  const calls = []
  const sidecar = {
    request: async (command, params) => {
      calls.push({ command, params })
      return { acquired: false }
    },
  }
  const sessions = [{ id: "root", time: { created: 1 } }]
  const setup = fixture({ sidecar, sessions })
  setup.options.disableBackfill = false
  setup.options.backfillDelay = 0

  await ZeroMemPlugin(setup.input, setup.options)
  await waitFor(() => calls.length === 1)

  assert.equal(calls[0].command, "acquire_backfill")
})

test("backfill only processes the configured most recent root sessions", async () => {
  const calls = []
  const sidecar = {
    request: async (command, params) => {
      calls.push({ command, params })
      if (command === "acquire_backfill") {
        return { acquired: true }
      }
      return { ingested: params.turns?.length || 0 }
    },
  }
  const sessions = Array.from({ length: 5 }, (_, index) => ({
    id: `root-${index}`,
    time: { created: index },
  }))
  const messages = new Map(sessions.map((session) => [
    session.id,
    [{
      ...userMessage(`message-${session.id}`, `memory ${session.id}`),
      info: {
        ...userMessage(`message-${session.id}`, `memory ${session.id}`).info,
        sessionID: session.id,
      },
    }],
  ]))
  const setup = fixture({ sidecar, sessions, messages })
  setup.options.disableBackfill = false
  setup.options.backfillDelay = 0
  setup.options.backfillSessions = 2

  await ZeroMemPlugin(setup.input, setup.options)
  await waitFor(() => calls.some((call) => call.command === "release_backfill"))

  assert.deepEqual(
    calls
      .filter((call) => call.command === "ingest_batch")
      .map((call) => call.params.turns[0].session_id),
    ["root-3", "root-4"],
  )
})
