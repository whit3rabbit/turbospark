---
uuid: "4be2f6e6-79e2-474b-ab8b-955d7958588b"
title: "TurboSparkApp: lifecycle and naming traps"
summary: "AppModel.server outlives AppModel.session unless a caller explicitly stops it. Two unrelated features are both called guardrails in this codebase"
tags: ["swift", "app", "architecture"]
source: "swift/CLAUDE.md"
depends_on: ["f872646f-6cbf-44b4-a2bd-85a13de114f8"]
created: "2026-09-05"
updated: "2026-09-05"
---

## What naming and lifecycle traps exist in AppModel?

Several `AppModel` properties look independent and are not, and this app
tracks the fixes for such defects as `(state#N)` comment markers with a
lookup table in `swift/CLAUDE.md`'s `## The state#N ledger` section, so
grep a comment's number there before re-deriving what it means.

## Don't

- Don't treat `AppGuardrailsMode` and `AppLoadGuardOption` as the same
  setting. `AppGuardrailsMode` (`State/MacAppSettings.swift`) is tool-call
  dialect rescue and schema validation. `AppLoadGuardOption`
  (`State/AppRuntimeOptions.swift`) is how much memory a model may commit
  at load (`docs/LOAD_GUARD.md`). They share no code, no settings key, and
  no UI surface. `MacAppSettings` persists both as separate keys, so
  grepping for "the guardrails setting" finds the wrong one about half the
  time.
- Don't build a fourth `recommend` call site (beyond `ModelHubView`,
  `CatalogSheet`, `ModelInstallView`, `buildOpenOptions`) from
  `runtimeOptions` directly. Use `AppModel.activeLoadGuard`. The hub and
  the loader share one memory budget by construction, and a hub ranking
  computed under a different tier than sessions actually open under
  promises a fit the loader then refuses, invisibly to the user.
- Don't clear `AppModel.session` directly and assume the model unloads.
  `AppModel.server` holds its own reference to the engine on the Rust
  side, so `session = nil` alone leaves the model resident and the server
  still answering for it. `open(_:)`, `unloadModel()`, and
  `setModelURL(_:)` route through `detachChatSessionFromServer()` (or, for
  a fully-stopped server, `stopServer()`) first. A caller since 2026-08-30
  must use the detach path specifically: a server can hold several models,
  and `stopServer()` would take all of them down to unload one.
- Don't assume `TurboSparkServer.stop()` and `deinit` calling the C
  function separately is safe. Both route through one `NSLock`-guarded
  idempotent path, because calling it twice (once explicit, once at scope
  exit) would double-free.
- Don't have a view state an address, port, or auth status it computed
  itself. All three are read back off `info()` into a published snapshot
  (`AppModel.serverInfo`), refreshed only on start/attach/detach/poll-tick,
  never from a SwiftUI body. A hand-restated `127.0.0.1` was one of the
  three that had silently drifted from what the engine actually bound.
- Don't assume a projectless chat is safe to root anywhere convenient.
  There's no defensible default filesystem root for a chat with no
  project, so path-taking and process-spawning tools are refused BY NAME
  for that case (`workspaceRootedToolNames`) rather than sandboxed to some
  directory that merely looks safer than `/`. See
  [[swift-tool-execution]] for the rest of the tool-execution/permission
  story.
