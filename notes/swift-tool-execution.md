---
uuid: "8e1f5553-7073-4050-8396-6c6a02917eea"
title: "TurboSparkApp: tool execution and containment"
summary: "The permission gate lives in the two agent-loop CALLERS, never in AppToolRegistry.execute. A direct call to execute is ungated. run_command has no path, so for shell the gate is the whole containment story"
tags: ["swift", "app", "tools", "security"]
source: "swift/docs/SWIFT_TOOLS.md, swift/CLAUDE.md Gotchas 11, 30, 32"
created: "2026-09-05"
updated: "2026-09-05"
depends_on: ["73c2c625-6312-4bc9-931d-fc1f1391506c"]
---

## How does the app execute a model-proposed tool call, and where is it stopped?

A call passes five stages: parse (`Tools/Core/ToolCallParser`, one parser,
two callers), gate (in the CALLER, not the registry), refuse-or-run
(`AppToolRegistry.execute`), return (a `.tool` message, never mid-history
`system`), render (a name-keyed card). The gate is
`AppToolPermissionEngine.evaluate`, called by both agent loops, and for a
shell command it consults
`TerminalCommandClassifier.isAutoApprovable` (a positive allowlist) and
`CommandGate` (advisory only by default). No project means no root, and no
root means file and shell tools are refused BY NAME
(`workspaceRootedToolNames`): there is no defensible default root, since
`/` makes containment a no-op and `~` holds every credential on the
machine. A projectless chat roots at the home directory instead, still
wide, so path- and process-taking tools stay refused rather than trusting
that root.

## Don't

- Don't call `AppToolRegistry.execute` directly expecting it to be gated.
  It isn't. `execute` does exactly one check of its own (the rootless
  refusal), then runs the `case`. The permission engine only runs in the
  two agent-loop callers, and the unit tests rely on that separation.
- Don't assume `resolveSecurePath` was always doing containment. Until
  2026-08-28 it standardized a path and appended it to the root, which
  resolves `..` and follows it OUT: `../../../../etc/passwd` against root
  `/Users/me/proj` resolved to `/etc/passwd`. It now refuses absolute and
  `~` paths up front and compares resolved target against resolved root,
  symlinks followed both sides.
- Don't assume containment applies to shell commands. `resolveSecurePath`
  constrains file tools only, and a shell leaves the root by its own
  means, so the permission gate is the WHOLE containment story there. The
  app is unsandboxed. `AppToolSandbox` is a path and domain policy, not an
  OS sandbox.
- Don't let an unimplemented transport return a fake success.
  `McpClientEngine`'s SSE path once returned the literal string `"SSE
  remote tool execution completed."` without sending a request. An
  unimplemented arm must throw and name the transport.
- Don't seed a spawned child process (a custom tool, an MCP stdio server)
  from `ProcessInfo.processInfo.environment`. That hands a third-party
  binary every credential the app launched with. Pass only `PATH`, `HOME`,
  `LANG`, `TMPDIR`, and the config's own declared `env`.
- Don't add a new tool without touching all five vocabulary surfaces:
  schema + catalog registration, `supportedToolNames`,
  `workspaceRootedToolNames` if path/process-taking, a `category(for:)`
  arm, and the risk classifier's safe list if read-only.
  `FabricatedToolSuccessTests` reddens on a name advertised but not wired.

See [[swift-binding-basics]] for the `-L` linker trap that also applies
when testing this package, and the paired page for why the gate's own
classifier ships disabled.
