# Syntext: Indexed Code Search

Syntext is an indexed code search engine and context engine built in Rust
with Swift bindings. This page covers code indexing: indexed regex and literal
search (`grep_search`) across a project workspace, returning ripgrep-formatted
output with line numbers and context lines. It also owns the package pin and
the link setup that both products share.

The same library's context engine (`SyntextContext`) powers conversation
recall, documented separately in [SYNTEXT_CONTEXT.md](SYNTEXT_CONTEXT.md). The
two share a dependency and nothing else: recall never reads a code index, and
code search never reads chat turns.

All code, comments, and documentation must remain ASCII: no emojis and no em
dashes (project rule).

## 1. Scope: Projects and User-Level Enablement

Code indexing is scoped to **projects** and requires both gates:
- Indexing is controlled globally at the **user level** via
  `MacAppSettings.syntextIndexingEnabled` (App Settings > General, default on),
  and per project via `AppProject.syntextIndexEnabled` (default off for new and
  previously saved projects). When either gate is off, search uses the
  unindexed scan (`searchCode`).
- Search tools are workspace-rooted (`workspaceRootedToolNames`), so a chat
  with no project is refused before dispatch and never reaches either path.
- Selecting an opted-in project, or launching with one selected, opens or
  builds its index in the background. Activations carry an ordered token, so
  rapid switches cannot land out of order. Selecting another project releases
  the prior mmap-backed handle while keeping its on-disk cache.
- The search re-reads the project's opt-in at call time (as the MCP path
  does), so turning it off during a turn holds for that turn's next search.

## 2. Architecture and Actors

### SyntextCodeSearchTool
`Tools/File/SyntextCodeSearchTool.swift` defines `public actor SyntextCodeSearchTool`.
Each instance manages the on-disk index for a single project root.

- **Index Directory**:
  `~/Library/Caches/TurboSpark/syntext/<digest>.syntext`, where `<digest>` is
  the first 32 hex characters of SHA-256 over the standardized, symlink-resolved
  root path. A fixed-length name: the earlier base64-of-path name passed
  APFS's 255-byte limit for long roots (every build failed) and decoded back
  to the project path. The legacy `com.turbospark.app/syntext` folder is
  removed on first use. Tests get `AppStorageRoot.machineRoot/Caches/syntext`.
- **`ensureIndex()`**: opens an existing index, or builds one when the open
  reports `IndexNotFound` (code 2), and waits. Background activation uses it.
  Concurrent callers share one in-flight open/build.
- **`grep(...)`**: never waits for a build. With no index on disk it starts
  one in the background and throws `SyntextIndexNotReady`, and the registry
  answers that call with the unindexed scan. A failed open or build is not
  retried for 60 seconds.
- **Builds**: one at a time across every project (`SyntextBuildQueue`). A
  build can peak around 1.5 GB. Opens and builds retry a lock conflict up to
  three times (`isRetryableLockConflict`), since the app spawns `git`.
- **`buildIndex()`**: waits for any in-flight open of the same root, then
  forces a fresh build.
- **`deleteIndex()`**: cancels pending commits, waits for an in-flight build to
  release its handles, and removes the on-disk directory.

### SyntextIndexManager
`SyntextIndexManager.shared` routes tools by repository root. It keeps one
instance per root for the life of the process. Only the **active** root keeps
a resident handle. A root switched away from keeps its in-flight build and
buffered changes, so switching back joins that build instead of starting a
second one. `buildIndex(for:)` (Reindex) does not change which project is
active. It exposes `tool(for:)` (activates), `activate(_:token:)`,
`isIndexed`, `buildIndex`, `deleteIndex`, `stats`, `notifyChange`,
`notifyDelete`, `syncQuietly`, `ensureActiveProjectIndexed`, and
`deactivateAll`.

## 3. LLM Tool Schema and Execution

### Tool Definition
`FileSearchToolDefinitions.grepSearch` in `Tools/File/FileSearchTools.swift`:
`grep_search` with `query`, `literal_search`, `path_filter`, `file_types`,
`case_sensitive`, `context_lines` (default 2), and `max_results` (default 50).

### Execution
`AppToolRegistry.swift` handles `search_code`, `grep`, `search`, and
`grep_search` in one case:
- `max_results` is clamped to 1...500 and `context_lines` to 0...10 on both
  paths, and indexed output goes through `compactOutput`.
- `path` becomes a scope (`syntextPathScope`: a `dir/` prefix or a file). A
  path that `resolveSecurePath` refuses is an error on both paths. It used to
  become a whole-repository search. The scope is enforced in Swift even when
  `path_filter` is also given (the library takes one filter).
- Results pass `GrepPathPolicy` (`Tools/File/GrepPathPolicy.swift`) on both
  paths: no hidden path components below the scope, no `.git`,
  `node_modules`, `target`, or `.build`, and nothing
  `ToolRiskClassifier.isSensitivePath` matches (`.env*`, `credentials`,
  `*.pem`, `*.key`, `.npmrc`, `.netrc`). Syntext's own walker indexes hidden
  files and `.git/`, and search is auto-allowed, so this filter is the only
  thing standing between the model and those files.
- `grep_search` is in the classifier's always-safe list, which means its path
  arguments get the sensitive-path check (a sensitive `path` is high risk).
- The built-in `explore` subagent designates `grep_search` as its primary
  search tool.

### Freshness

Measured against syntext 2.7.0 (2026-10-16): an edit to a file the index
already holds shows up in search without any notification, because syntext
re-reads changed files at query time, and a deleted file drops out the same
way. A file created after the build is invisible until a notification for it
is committed.

- Agent writes notify the index inline (awaited, not fire-and-forget):
  `write_file`, `edit_file`, `multiedit`, `notebook_edit`, and `apply_patch`
  (deletes call `notifyDelete`). A search commits pending notifications before
  it runs, so a file written and then searched in the same batch is found.
  Idle commits still debounce for 1.5 seconds.
- **Overlay full.** Syntext refuses a commit once the overlay outgrows the
  base index (code 9), and keeps refusing. The commit used to be `try?`, so
  every file created after that point stayed invisible until a manual Reindex.
  Now the handle is dropped and the index rebuilds in the background, and
  searches use the unindexed scan until it lands.
- Changes that arrive while no handle is loaded (a build in flight, an
  inactive project with an index on disk) are buffered, up to 512 paths, and
  replayed into the next handle. Past that, the next load rebuilds.
- Git repositories (`<root>/.git` present) use `searchFreshAsync`: one bounded
  `git status` detection (200 files / 150 ms by default) applied in memory.
  When it cannot apply the delta (`budgetExceeded`, `tooManyFiles`,
  `overlayFull`), the output says the results may be stale. A project root
  nested inside a larger repository has no `.git` of its own and gets no git
  detection.
- In a non-git directory, a file created outside the app needs the explicit
  Reindex action. The app runs no polling loop and installs no repository
  hooks.

## 4. UI: Settings and Project Index Management

- **App Settings > General ("Code Search & Project Indexing")**: the global
  toggle.
- **Project Settings Sheet ("Code Search Index (Syntext)")**: the per-project
  opt-in, live stats, Reindex / Index Now, and Remove Index. Removing the
  index while the project stays opted in rebuilds it at the next search.

## 5. Dual-Staticlib Symbol Isolation

`TurboSparkApp` links two Rust static libraries:
1. `libturbospark_ffi.a` (from `crates/ffi`)
2. `libsyntext.a` / `SyntextFFI.xcframework` (from syntext, which since 2.6.0
   also carries the context engine's C ABI)

Both export `_rust_eh_personality`, and ld64 fails on the duplicate. The app
build sets `TURBOSPARK_LOCALIZE_RUST_SYMBOLS=1`, and `scripts/swift-lib.sh`
then makes the symbol local in `libturbospark_ffi.a` with `nmedit -R`. Moving
to 2.7.0 added no new duplicate (checked with a clean link, 2026-10-16). The
2.7.0 archive's `blake3_neon.o` is built for macOS 14.5 against the app's
14.0 floor. ld warns, and the object is SIMD code with no OS calls.

## 6. Package Dependency

In `swift/TurboSparkApp/Package.swift`:
```swift
.package(url: "https://github.com/whit3rabbit/syntext",
         revision: "1a26441ddf646720cfb8828d894353659b48af56"),
```
with products `Syntext` and `SyntextContext`.

**Pin by revision, never by an exact `vX.Y.Z` tag.** syntext's release
workflow publishes the xcframework from the tag push, then commits the
`url`/`checksum` pin to `main` and tags that commit `swift-vX.Y.Z`. Each
`vX.Y.Z` tag therefore pins the PREVIOUS release's binary: `exact: "2.5.0"`
linked the 2.4.0 xcframework, which silently ignored `fixedStrings`,
`wordRegexp`, and context lines. SwiftPM cannot use `swift-vX.Y.Z` as a
version, so the app pins that commit by revision. The revision above is
`swift-v2.7.0` (xcframework 2.7.0, checksum `510f0c33...a510`). syntext's
`docs/SWIFT.md` documents the same rule. To upgrade: find the
`swift-vX.Y.Z` commit (`git rev-list -n1 swift-vX.Y.Z`), update the revision,
`swift package resolve`, and confirm `.build/workspace-state.json` names the
matching `syntext-swift-X.Y.Z.xcframework.zip`.

CI and contributors need no Rust toolchain for syntext: SwiftPM downloads
the release zip and verifies its checksum.

### Local Development Override
```bash
cd swift/TurboSparkApp
swift package edit syntext --path ../../../syntext
```
syntext's root `Package.swift` prefers `swift/build/SyntextFFI.xcframework`
when it exists, so rebuild it (`swift/Scripts/build-xcframework.sh`) before
editing against a local checkout. Revert with `swift package unedit syntext`.

## 7. Conversation Recall

Conversation recall (`SyntextContext`) has its own page:
[SYNTEXT_CONTEXT.md](SYNTEXT_CONTEXT.md). It covers how recall works, how it
differs from memory ([SWIFT_MEMORY.md](SWIFT_MEMORY.md)), where it is wired in,
the plaintext store and its gates, and its limitations.

## 8. Verification

Code search, in `swift/TurboSparkApp/Tests/TurboSparkAppTests/`:
- `SyntextCodeSearchToolTests.swift`: build and search, single-flight opens,
  handle release on project switch, live modification, registry execution,
  opt-out fallback, removal, and settings persistence.
- `SyntextGrepSafetyTests.swift`: literal metacharacters, context lines,
  sensitive/hidden/vendor exclusion on both paths, scope with `path_filter`,
  escaping paths, result caps, a new file found right after its write, the
  overlay-full rebuild, deletes, the
  non-blocking first search, the fixed-length cache name, risk
  classification, and that the registry answered from the index.

Recall tests are listed in [SYNTEXT_CONTEXT.md](SYNTEXT_CONTEXT.md).

```bash
cd swift/TurboSparkApp && swift test --filter Syntext
```
