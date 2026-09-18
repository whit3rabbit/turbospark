# Singleton background-work audit

Source audit on 2026-09-10 for ROADMAP Priority 0 item 8. The audit is
needed: a file-system redirect does not isolate work already acting on a
shared object. Two concrete gaps were confirmed after the cron repair and are now fixed.

## Selection and scope

The referenced roadmap task finished the batched GEMV A/B. The current
earlier engine entries require further hardware measurements or model
quality investigation. The cron verification and timer repair are already
present, with mutation results and full-suite limitations recorded in
[storage.md](storage.md#cron-stores-and-background-polling).
They should not be reimplemented because the roadmap still lists them.

This pass surveyed app-owned `static let shared` / `static var shared`
declarations, initializer work, timer/task launch sites, and relevant test
callers. Source was read directly after synrepo reported its index stale.
This is not an exhaustive audit of every static callback, framework-owned
singleton, persisted setting, or parallel-test interleaving.

## Original finding: MCP cache reset did not fence pending work

`Tools/MCP/McpToolCatalogCache.swift` starts discovery in an unstructured
Task. `removeAll()` clears entries and the in-flight name set, but a pending
task can subsequently call `setTools` and restore the removed entry.
`removeServer(named:)` has the same problem. Completion also clears the
in-flight marker by name, without proving it owns the current request.

This matters to tests: `McpCatalogApprovalRulesTests` resets the shared
cache in setup and teardown. AppModel startup, project selection, plugin
reload, and MCP configuration edits can initiate discovery through
`refreshMcpToolCatalog`. Resetting the cache is not a boundary against an
earlier request finishing after setup. No actual full-suite failure is
attributed to this race by this audit.

It also matters in the app: disabling/removing a server while discovery is
pending can restore its cached row. A transport replacement can receive an
old result. Those additional interleavings are inferred from the same
unconditional completion path, not independently reproduced here.

### Controlled reproduction

The probe compiles the actual cache source with small fixture definitions
for its MCP dependencies. Its discovery actor suspends on a continuation,
so no server, network, model, fan command, or real directory scan runs.

1. Start discovery and wait until the fixture receives it.
2. Call `removeAll()` and assert the entry is absent.
3. Resume discovery with a fixture tool.
4. Observe the removed entry returning.

The run printed:

```text
REPRODUCED: in-flight discovery repopulates cache after removeAll
```

Cache source SHA-256 at reproduction:
`8f5fe6261d0d746dddb1833ec116f8be04ea636917ab32efeca5c9ebc1834e99`.

From the repository root:

```sh
mkdir -p /tmp/roadmap-singleton-audit
swiftc -parse-as-library \
  -module-cache-path /tmp/roadmap-singleton-audit/module-cache \
  swift/TurboSparkApp/Sources/TurboSparkApp/Tools/MCP/McpToolCatalogCache.swift \
  docs/verification/swift-singleton-cache-probe-2026-09-10.swift \
  -o /tmp/roadmap-singleton-audit/probe
/tmp/roadmap-singleton-audit/probe
```

The saved probe now awaits the cache task and asserts that the removed
entry stays absent, printing `PASS` on the fixed source. It remains a
fixture-only cache check, not an MCP integration test. The old output and
source hash above record the original reproduction.

## Original finding: AppModel tests initialized the hardware poller

`AppModel.init()` calls `loadSettings()`. The latter assigns
`FanController.shared.keepFansPinnedOnQuit`, initializing the singleton.
`FanController.init()` resolves `thermalforge` and calls `startPolling()`.
When the executable exists, this immediately schedules a status refresh
and a repeating timer. `AppStorageRoot` does not redirect this dependency.

`FanControllerTests` already supplies an executable fixture and
`pollInterval: nil` for action tests, and explicitly stops the timer in its
polling test. That seam does not cover ordinary AppModel construction.
The roadmap's claim that the fan initializer invalidates on restart was
stale: current `startPolling` is private and called once by initialization;
`stopPolling()` performs explicit invalidation.

A test-safe shared default was needed; the fix below uses fixture-only
verification.
This audit did not construct AppModel, initialize the fan singleton, check
the host's fan state, or change cooling. It establishes the call path, not
that this poller caused any prior flake or energy outlier.

## Other background surfaces

| Surface | Observed contract and remaining limitation |
| --- | --- |
| CronScheduler | Private directory/instance seam, locked in-flight reservation; AppModel timer restart/stop/deinit cleanup already implemented. Existing mutation evidence is in SWIFT_STORAGE. |
| BackgroundShellManager | Launch-driven processes, no initializer poller. BackgroundShellTests and KillSurfacesTests call resetForTests, which removes records and terminates/reaps running processes. The shared observer and queued notifications still warrant scrutiny before parallelizing these tests. |
| McpClientEngine | Empty initializer and per-call subprocess cleanup with terminateAndReap. Engine tests construct separate instances. This does not fence cache publication by callers. |
| SystemPermissionsManager | Initializer launches a finite directory probe. Tests construct separate managers; a generation check prevents an older probe overwriting a newer refresh. The real-home probe and shared UserDefaults are environmental dependencies, not repeating singleton polling. |
| ProjectFileIndex | Request-driven bounded scan, keyed by canonical root and awaited by callers; no initializer poller. invalidate clears cached rows but not pending scans, a secondary stale-publication candidate not reproduced here. |
| AttachmentThumbnailStore | Request-driven QuickLook operation awaited through a continuation; tests use unique temporary paths. No initializer poller. Framework cancellation and cache capacity are outside this audit. |
| AppHookExecutionEngine | Empty initializer, but notification hooks can run detached and dispatch reads AppHookStore.shared even on a separate engine instance. Private engine construction alone is not hook-store isolation; draining detached hooks needs a separate audit. |
| AppSpeechSynthesizer | Speech starts explicitly; initializer installs a delegate. No repeating poller. |
| ArtifactContentRuleList | Compilation starts on demand and completes queued callbacks; no initializer polling. |
| AppModel server/goal timers | Per-model work with explicit stop/cancel methods. These are lifecycle references, not proof every teardown path invokes them. |

Other declarations route to stores, registries, presentation assets, or
request-driven services: SkillManager, AgentManager, PluginManager,
CustomToolManager, MemoryStore, AppHookStore, ModelOrganizationStore,
AppearanceManager, TaskManager, SessionApprovalStore, FileSnapshotStore,
ShellCwdTracker, AgentModeGate, the three marketplace managers,
GreetingProvider, the logo/ghost/welcome/task-progress caches, and
AppShutdownCoordinator. The declaration/initializer survey did not identify
another autonomous repeating poller in this group. Shared mutable state
itself is not claimed to be isolated by that observation.

## Fix and validation

Each pending cache request now owns a UUID. Publication and cleanup check
that identity under the same lock. Reset, removal, pruning/disable,
transport replacement, and explicit `setTools` revoke the old identity.
A failed current discovery preserves cached tools and permits retry.
Already-running discoveries may finish, but obsolete results have no
publication or cleanup rights. This change does not cancel their processes.

Discovery is injectable. Internal task handles let regression tests await
completion of publication, so their assertions do not race a background
callback. `McpCacheLifecycleTests` exercises reset, removal, disable,
pruning, old success/failure during replacement, same-transport replacement
after reset, explicit publication, retry, and deduplication.

The fan singleton is unavailable and has no poll interval under XCTest.
It skips executable lookup; explicit fixture controller initialization and
normal app behavior retain their existing contracts. The new shared-default
test checks availability and polling. Its mutation runs with a fake
`thermalforge` first on PATH, so it cannot invoke the real hardware CLI.

Eight asserted, restored mutations caught missing reset/removal/pruning,
request identity, manual-publication invalidation, retry cleanup, and the
fan default. Pruning serves both disable and absent-server cases; request
identity serves both publication and cleanup. Those shared invariants
appropriately fail multiple matching cases. Evidence is in
`docs/verification/swift-singleton-fix-2026-09-10.json`.

The restored full Swift suite passed: 1,545 tests, one skipped, zero
failures. The saved standalone cache probe also passed. Rust workspace
build, tests (with host Metal access), formatting, and Clippy all passed.
No power or throughput measurements were run.
