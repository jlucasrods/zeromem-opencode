import { spawnSync } from "node:child_process"
import { mkdirSync, writeFileSync } from "node:fs"
import { homedir } from "node:os"
import { dirname, join } from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const manifest = join(root, "sidecar", "Cargo.toml")
const build = spawnSync("cargo", [
  "build",
  "--release",
  "--locked",
  "--manifest-path",
  manifest,
], { stdio: "inherit" })

if (build.error) {
  console.error(`Failed to start Cargo: ${build.error.message}`)
  process.exit(1)
}
if (build.status !== 0) {
  process.exit(build.status ?? 1)
}

const configRoot = process.env.XDG_CONFIG_HOME || join(homedir(), ".config")
const pluginsDir = join(configRoot, "opencode", "plugins")
const entrypoint = pathToFileURL(join(root, "index.js")).href
mkdirSync(pluginsDir, { recursive: true, mode: 0o700 })
writeFileSync(
  join(pluginsDir, "zeromem.js"),
  `export { default } from ${JSON.stringify(entrypoint)}\n`,
  { mode: 0o600 },
)

console.log("ZeroMem installed. Restart OpenCode to load the plugin.")
