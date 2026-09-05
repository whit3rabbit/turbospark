---
uuid: "b8b6dcbc-87fe-49b1-a76c-7047aabf5047"
title: "TurboSparkApp: persistence and data safety"
summary: "A one-chat XCTest fixture replaced a real user's entire chat history, silently, because seven stores each spelled their own path with no test seam"
tags: ["swift", "app", "testing"]
source: "swift/CLAUDE.md"
depends_on: ["46b9d0c4-642c-4f54-a02a-c3787756f7fd"]
created: "2026-09-05"
updated: "2026-09-05"
---

## How does this app lose a user's data, and how is that prevented now?

It already happened. `HookDecisionRoutingTests` builds an `AppChat(title:
"T10 chat")` fixture, and every one of ~20 `AppModel()` constructions
across 10 test files read and wrote the REAL
`~/Library/Application Support/TurboSpark/chats_archive.json`, because
seven stores (`AppChatFileStore`, `AppProjectFileStore`, `MacAppSettings`,
`GlobalMcpFileStore`, `AppHookStore`, `CustomToolManager`,
`AppHookStdinPayload`) each computed that path inline with no test seam.
Since the archive is written WHOLE with `.atomic` on every mutation, the
one-chat fixture replaced a real conversation history outright. The bug
report read as "a chat won't stay deleted": deleting it in the UI did
nothing, since the next `swift test` run wrote it straight back.

The fix is `AppStorageRoot`, one root that redirects automatically (keyed
on the XCTest host process, not a flag a test opts into) and per-process
rather than per-test, since these stores are `.shared` singletons whose
first touch can precede any test body. `TURBOSPARK_STATE_DIR` overrides it
by hand. `StorageIsolationTests` guards it directly.

## Don't

- Don't assume a JSON store's `load()` errors on a bad decode. It swallows
  the failure and returns the empty default, so a non-backwards-compatible
  schema change (a new non-optional field) silently discards the user's
  data rather than failing loudly. Add fields with `decodeIfPresent` and a
  default, the way `MacAppSettings`'s hand-written `init(from:)` does.
- Don't trust a green test suite as proof nothing was touched. This whole
  class of bug leaves the suite green, the app launching fine, and the only
  tell is data the user didn't create. Nothing errors and nothing logs.
- Don't assume `AppStorageRoot` covers everything. It's the app's own
  seven stores only. `~/.turbospark/installed.json` is the CATALOG's, reached
  through the FFI, and nothing redirects it: a test driven through
  `refreshModels`, `deleteModel`, or `installed()` answers differently
  depending on what's actually installed on the developer's machine, which
  reads as flakiness but is really the test measuring the disk.
- Don't test a persisted `@Published` property's declared default directly
  against a fresh `AppModel()`. `AppModel.init()` calls `loadSettings()`
  first, so the value a caller observes comes from `MacAppSettings`'s own
  default, not the property's declared one. `AppModel().interactionMode ==
  .chat` tests whichever `settings.json` happens to sit in the shared,
  per-process `AppStorageRoot.directory`, carrying state across every test
  file that ran earlier in the same `swift test` invocation. Test
  `MacAppSettings`'s default directly, or delete the settings file first.
- Don't seed the real catalog store to make a test pass. Extract the
  decision as a pure static function fed a fixture instead (the pattern:
  `AppModel.isCatalogTracked(model:in:)`), or the fixture just moves the
  same failure mode to a different API.
