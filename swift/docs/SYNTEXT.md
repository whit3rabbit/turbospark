# Syntext: Indexed Code Search and Project Indexing

Syntext is a high-performance indexed code search engine and project indexer
built in Rust with Swift bindings (v2.5.0). In TurboSparkApp, it provides
sub-millisecond regex and literal code search (`grep_search`) across project
workspaces, returning ripgrep-formatted output with line numbers and context lines.

All code, comments, and documentation must remain ASCII: no emojis and no em
dashes (project rule).

## 1. Scope: Projects and User-Level Enablement

Indexing is strictly scoped to **projects** and requires both gates:
- Chats without an attached project workspace root directory do not run Syntext
  indexing.
- Projectless chats calling code search fall back to the existing filesystem
  scanner (`searchCode`).
- Indexing is controlled globally at the **user level** via `MacAppSettings.syntextIndexingEnabled`
  (App Settings > General), and per project via `AppProject.syntextIndexEnabled`.
  New and previously saved projects default to off. When either gate is off,
  search falls back to unindexed scanning.
- Selecting an opted-in project opens or builds its index in the background.
  Selecting another project unloads the prior mmap-backed handle while keeping
  its on-disk cache for fast reuse.

## 2. Architecture and Actors

### SyntextCodeSearchTool
`Tools/File/SyntextCodeSearchTool.swift` defines `public actor SyntextCodeSearchTool`.
Each instance manages the on-disk index for a single project root.

- **Index Directory**:
  Default cache path: `~/Library/Caches/com.turbospark.app/syntext/<repoId>.syntext`,
  where `repoId` is a sanitized base64 representation of the standardized root path.
- **`ensureIndex()`**:
  Attempts to open an existing on-disk index instantly using `SyntextIndex(indexDir:repoRoot:)`.
  If no index exists (error code 2, `IndexNotFound`), it triggers an asynchronous
  build on a utility task. Concurrent callers share one in-flight open/build;
  project selection and the first search cannot start duplicate builds.
- **`buildIndex()`**:
  Releases the existing shared-lock handle before forcing a fresh on-disk build
  off the main thread. Syntext builds in 256 MB batches and can peak around
  1.5 GB, which is why builds are opt-in and single-flight.
- **`deleteIndex()`**:
  Cancels pending debounced commits, unloads in-memory handles, and removes the
  on-disk `.syntext` cache directory.
- **`grep(query:pathFilter:fileTypes:caseSensitive:literalSearch:wordMatch:contextLines:maxResults:)`**:
  Configures `SyntextSearchOptions` and searches using `index.grepAsync(...)`.
  If a `.git` repository is present at the root, it queries via `index.searchFreshAsync(...)`
  to automatically pick up uncommitted git working-tree changes.
  Output is formatted in standard ripgrep format (`path:line:content` and `-` for context lines).
- **`fileDidChange(at:)` and Debounced Commit**:
  Buffers file modifications through `index.notifyChange(...)` and sets a 1.5-second
  quiet timer before committing batch changes (`index.commitBatch()`). Rapid edits
  reset the timer so disk writes happen only when the workspace is quiet.
- **`syncQuietly()`**:
  Commits pending in-app edit notifications. It does not run a second git
  freshness pass at the end of every generation turn.

### SyntextIndexManager
`SyntextIndexManager` is a global shared actor (`SyntextIndexManager.shared`) that
routes `SyntextCodeSearchTool` instances by repository root URL. It retains at
most one active project tool and unloads the previous index handle on a project
switch or when global indexing is disabled.
It exposes:
- `tool(for: rootURL) -> SyntextCodeSearchTool`
- `isIndexed(for: rootURL) async -> Bool`
- `buildIndex(for: rootURL) async throws -> SyntextStats`
- `deleteIndex(for: rootURL) async throws`
- `stats(for: rootURL) async throws -> SyntextStats`
- `notifyChange(at: fileURL, rootURL: rootURL) async`
- `syncQuietly(for: rootURL) async`
- `ensureActiveProjectIndexed(for: rootURL) async`
- `deactivateAll() async`

## 3. LLM Tool Schema and Execution

### Tool Definition
The tool is defined in `Tools/File/FileSearchTools.swift` as `FileSearchToolDefinitions.grepSearch`:
- **Tool name**: `grep_search`
- **Parameters**:
  - `query` (String, required): Search pattern (regular expression or literal string).
  - `literal_search` (Bool, optional): Exact string search without regex metacharacters (`-F`).
  - `path_filter` (String, optional): Glob filter (e.g. `*.swift`, `Sources/**`).
  - `file_types` ([String], optional): File extension list (e.g. `["swift", "rs"]`).
  - `case_sensitive` (Bool, optional): Case-sensitive matching (default `false`).
  - `context_lines` (Int, optional): Lines of context before/after matches (default `2`).
  - `max_results` (Int, optional): Maximum matches returned (default `50`).

### Execution and Aliases
In `Tools/Registry/AppToolRegistry.swift`, both the ordinary model-visible
`Grep` tool and the explicit indexed-search tool are handled under:
```swift
case "search_code", "grep", "search", "grep_search":
```
- If global `AppToolRegistry.syntextIndexingEnabled` and the selected project's
  `syntextIndexEnabled` are both true, execution dispatches to
  `SyntextIndexManager.shared.tool(for: rootURL).grep(...)`.
- If indexing is disabled, unavailable, or throws an error, execution transparently
  falls back to standard unindexed file scanning (`searchCode`).
- Tool names `grep_search`, `grep`, and `search_code` are registered in `supportedToolNames`
  and `workspaceRootedToolNames` in `Tools/Registry/AppToolRegistry+Vocabulary.swift`.

### Incremental Live Updates and Quiet Sync
When agent tools modify workspace files (`write_file`, `edit_file`, `apply_patch`),
`AppToolRegistry.execute` notifies the index:
```swift
Task { await SyntextIndexManager.shared.notifyChange(at: targetURL, rootURL: rootURL) }
```
When generation turns complete and the agent goes idle, `AppModel.generating.didSet`
commits any pending notifications for the active opted-in project.

Git repositories use `searchFreshAsync` before an indexed search. Syntext performs
one bounded `git status` detection (the default cap is 200 files / 150 ms), applies
the delta in memory, then searches. This is not a full rebuild. TurboSparkApp does
not install repository hooks or enable `core.fsmonitor` behind the user's back.

Plain directories and machines without a usable `git` binary remain supported.
The initial walk builds the same index, TurboSparkApp tool writes stay fresh through
`notifyChange` plus a debounced commit, and searches use the index normally. Changes
made outside TurboSparkApp in a non-git directory require the explicit Reindex action;
the app deliberately does not run a permanent filesystem polling loop.

## 4. UI: Settings & Project Index Management

- **App Settings > General ("Code Search & Project Indexing")**:
  Provides a global user-level toggle for Syntext indexing. Toggling it off disables
  indexing across all projects simultaneously.
- **Project Settings Sheet ("Code Search Index (Syntext)")**:
  - If indexing is disabled globally, displays an informational callout pointing
    the user to Settings > General.
  - Provides the per-project opt-in toggle. New and legacy projects default off.
  - When enabled, displays live index stats (document count and disk footprint in megabytes).
  - Offers **"Reindex"** (or **"Index Now"**) to rebuild the index on demand.
  - Offers **"Remove Index"** to purge the `.syntext` cache directory from disk
    and reset memory handles.

## 5. Dual-Staticlib Symbol Isolation

`TurboSparkApp` links two Rust static libraries:
1. `libturbospark_ffi.a` (from `crates/ffi`)
2. `libsyntext.a` / `SyntextFFI.xcframework` (from `Syntext`)

Both static libraries are compiled by `rustc` with standard library support, and both
export `_rust_eh_personality`. On macOS, linking multiple static libraries exporting
the same global symbol causes Apple's linker (`ld64`) to fail with:
```
duplicate symbol '_rust_eh_personality' in libsyntext.a and libturbospark_ffi.a
```

### The Fix
In `scripts/swift-lib.sh`, immediately after copying `libturbospark_ffi.a`, the build script runs:
```bash
printf "_rust_eh_personality\n" > "$dest/.hide_symbols"
nmedit -R "$dest/.hide_symbols" "$dest/libturbospark_ffi.a"
rm -f "$dest/.hide_symbols"
```
`nmedit -R` changes `_rust_eh_personality` in `libturbospark_ffi.a` into a static (local)
symbol. The symbol remains fully functional for exception handling within `turbospark_ffi`
while eliminating the duplicate symbol collision during executable linking.

## 6. Package Dependency and CI Configuration

In `swift/TurboSparkApp/Package.swift`:
```swift
dependencies: [
    .package(path: "../TurboSpark"),
    .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.4.0"),
    .package(url: "https://github.com/whit3rabbit/syntext", exact: "2.5.0")
],
```

### Why a Remote Release Dependency?
- **CI Independence**: CI runners and contributors do not need a Rust toolchain to build
  or test `Syntext`. SwiftPM downloads the official release zip (`syntext-swift-2.5.0.xcframework.zip`)
  and verifies its SHA-256 checksum automatically.
- **Build Speed**: Avoids compiling Syntext's Rust crates on every clean build.

### Local Development Override
To iterate on `syntext` locally without changing `Package.swift`:
```bash
cd swift/TurboSparkApp
swift package edit Syntext --path ../../../syntext
```
To revert back to the remote package:
```bash
cd swift/TurboSparkApp
swift package unedit Syntext
```

## 7. Verification

Unit tests are located in `swift/TurboSparkApp/Tests/TurboSparkAppTests/SyntextCodeSearchToolTests.swift`:
- `testAppProjectSyntextIndexEnabledCoding`: Codable roundtrip and backward compatibility.
- `testSyntextToolBuildAndSearch`: Index creation, document counts, and filtered grep searches.
- `testConcurrentEnsureIndexUsesSingleBuild`: Concurrent first use starts one build, not one per caller.
- `testManagerUnloadsInactiveProjectHandle`: Project switches release the previous mmap-backed handle.
- `testSyntextToolFileModification`: Live file modification and incremental commit verification.
- `testAppToolRegistryGrepSearchExecution`: End-to-end `grep_search` execution via `AppToolRegistry`.
- `testGrepSearchFallsBackWithoutProjectOptIn`: An opted-out project searches without creating an index.
- `testDeleteIndexRemovesDirectoryAndResetsState`: Removal clears both live and on-disk state.
- `testMacAppSettingsSyntextIndexingEnabledPersistence`: Global settings persist and legacy values decode.

Run tests with:
```bash
cd swift/TurboSparkApp && swift test --filter SyntextCodeSearchToolTests
```
