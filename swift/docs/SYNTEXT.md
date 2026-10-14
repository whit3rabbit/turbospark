# Syntext: Indexed Code Search and Project Indexing

Syntext is a high-performance indexed code search engine and project indexer
built in Rust with Swift bindings, compiled into `libturbospark_ffi.a` at the
revision `Cargo.lock` pins (section 5). In TurboSparkApp, it provides
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
- The built-in `explore` subagent (`AgentManager+BuiltIns.swift`) and project-level
  `explore.md` definitions designate `grep_search` as their primary code-discovery tool,
  using Syntext indexed search by default rather than shelling out to grep/find via Bash.

### Incremental Live Updates and Quiet Sync
When agent tools modify workspace files (`write_file`, `edit_file`, `apply_patch`),
`AppToolRegistry.execute` notifies the index:
```swift
Task { await SyntextIndexManager.shared.notifyChange(at: targetURL, rootURL: rootURL) }
```
When generation turns complete and the agent goes idle, `AppModel.onGenerationFinished()`
(called from `generating.didSet`) commits any pending notifications for the
active opted-in project.

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

The same archive carries syntext's context engine, which the app uses for
retrieved evidence, conversation recall and the `context_*` tools; see
[SWIFT_CONTEXT_ENGINE.md](SWIFT_CONTEXT_ENGINE.md).

## 5. One Rust Library

A process can link only one Rust static library: each one carries its own
copy of the Rust standard library, so two of them collide at link time
(`duplicate symbol '_rust_eh_personality'`). Until syntext-context, the app
linked syntext's own `SyntextFFI.xcframework` beside `libturbospark_ffi.a`
and hid the collision with `nmedit -R _rust_eh_personality` plus a
placeholder `CSyntext.o` for Swift Build. Both workarounds are gone.

syntext is now compiled into `libturbospark_ffi.a`:

- `crates/ffi/Cargo.toml` depends on `syntext-context` (git, pinned `rev`,
  `features = ["ffi"]`), which brings syntext with its own `ffi` feature.
  `crates/ffi/src/lib.rs` has `extern crate syntext_context;`; without that
  reference the crate is not linked and no `syntext_*` or
  `syntext_context_*` symbol reaches the archive. The dependency sits behind
  the default `syntext` feature, which only the workflow-profile
  compatibility harness turns off (its historical app still links syntext's
  xcframework).
- `crates/ffi` needs Rust 1.89 (syntext's MSRV); the rest of the workspace
  keeps 1.82.
- The hand-written headers and the Swift wrappers live in
  `swift/TurboSpark/Vendor/syntext`, copied verbatim from that revision:
  `CSyntext` is a system-library target, `Syntext` and `SyntextContext` are
  products of the binding package (they share `package`-access helpers, so
  they must be in one package), and syntext's own XCTest suites run there
  under `make swift-test` against the real archive.

## 6. Bumping syntext

The headers describe the ABI of one revision. Bump all three together:

```bash
# 1. Point crates/ffi/Cargo.toml's `rev` at the new syntext commit, then:
cargo update -p syntext-context
# 2. Copy that revision's headers, wrappers and tests:
scripts/vendor-syntext.sh
# 3. Rebuild, and prove the headers against the archive:
make swift-lib && make swift-test && make swift-test-app
```

`scripts/swift-lib.sh` refuses to stage a library when `Cargo.lock`'s
revision differs from `Vendor/syntext/REVISION`. Never edit the vendored
files by hand; change syntext and re-vendor. Never add syntext's Swift
package or xcframework to either `Package.swift`.

To try an unmerged syntext change, push it to a syntext branch, pin that
commit and re-vendor; repin to the merge commit on syntext's `main` before
this repository merges, since a squash merge can orphan the branch commit.
A path or `[patch]` override has no revision for the check to compare, so
`swift-lib.sh` stops on it by design.

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
