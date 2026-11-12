# Server

## Overview
The Server (API) pane and the menu bar panel of the macOS app. It is the UI and pure-logic layer over two separate local services. The first is the in-process TurboSpark HTTP server, shared by the Text, Image, Audio and Embeddings API modes. The second is TypeSafe (OpenKind), a separate loopback daemon with its own port and Keychain key [TypeSafeAPIPaneView.swift:TypeSafeAPIPaneView]. Server lifecycle and state live on `AppModel` (outside this directory). This directory holds the views, the shared API pane components, the pure helpers that tests pin (metrics window, port parser, status phase, endpoint catalog, connect recipes, daemon arguments), Keychain storage for both API keys, and server presets (favorites).

## Interface
- Pane shell and shared components:
  - `ServerPaneView` switches on `model.apiWorkspaceTab` and hosts a resizable console [ServerPaneView.swift:ServerPaneView].
  - `APIWorkspaceTab` (text, image, audio, embeddings, typeSafe) [APIWorkspaceSelector.swift:APIWorkspaceTab].
  - `APISidebarPanel` and the `SidebarPanelMode` conformance (symbols per tab) [APISidebarPanel.swift:APISidebarPanel].
  - Shared building blocks: `APIControlSize`, `APIStatusBand`, `APIPanel`, `apiCard()`, `APIStatusLine`, `APIStatusText`, `APIRunIndicator`, `APICodeBlock`, `APIRequestPreview`, `APIRowList`, `APIDetachButton`, `APICopyButton`, `APIServedModelRow`, `APINothingServedRow` [APIPaneComponents.swift:APIStatusBand].
- Text mode views: `ServerHeaderBandView`, `ServerConnectionView`, `ServerLoadedModelsView` (and a private model picker sheet), `ServerLiveChartsView`, `ServerTextPreviewView`, `ServerChartsView`, `ServerConnectCardView`, `ServerConsoleView`, `ServerInspectorView`, `ServerAdvancedSettingsView`.
- Other API modes: `ImageAPIPaneView`, `AudioAPIPaneView`, `EmbeddingsAPIPaneView`, `EmbeddingsAPITestPanel`, `TypeSafeAPIPaneView`, `TypeSafePlaygroundView`, `TypeSafeBenchmarkView`, `TypeSafeQuestionCard`, `TypeSafeAnswerCard`.
- Pure logic and models:
  - `ServerMetricsStore`, `ServerRequestRecord`, `ServerMetricPoint`: bounded rolling window folded from `ServerEvent`s [ServerMetricsStore.swift:ServerMetricsStore].
  - `ServerLiveHistory`, `ServerLivePoint`, plus `AppModel.currentPhysFootprintBytes()` and `sampleServerLive()` [ServerLiveHistory.swift:ServerLiveHistory].
  - `ServerPortInput.parse` [ServerPortInput.swift:ServerPortInput].
  - `ServerAPIFamily`, `ServerEndpoint`, `ServerEndpointCatalog.all`, `ServerConnectSnippet`, `ServerConnectRecipes.snippets` [ServerEndpointCatalog.swift:ServerEndpointCatalog].
  - `ServerDaemonLaunch.args` [ServerDaemonLaunch.swift:ServerDaemonLaunch].
  - `ServerAPIKeyGenerator.generate` [ServerAPIKeyGenerator.swift:ServerAPIKeyGenerator].
  - `ServerFavorite` (Codable preset) and its resolution: `ServerFavoriteModel`, `resolve(in:)`, `rekeyed(in:)`, `followingMove(from:oldPath:to:)`, and `AppModel` extensions `saveServerFavorite`, `serverFavoriteRows`, `resolveServerFavoriteModels`, `loadServerFavorite` [ServerFavorite.swift:ServerFavorite].
  - TypeSafe form model: `TypeSafePlaygroundDraft` (`request()`, `from(_:)`, `pretty`), `TypeSafeQuestionDraft`, `TypeSafeChoiceOption`, `TypeSafePlaygroundPreset`, `TypeSafeLatencySummary` [TypeSafePlaygroundDraft.swift:TypeSafePlaygroundDraft].
- Menu bar panel:
  - `ServerMenuDashboardView` is the panel, `ServerMenuBarView` holds the daemon controls, stats and the agent and endpoint submenus, and `MenuBarStatusLine`, `MenuBarSlotRow`, `MenuBarRowLabel`, `MenuBarRowButton`, `MenuBarSubmenuRow`, `MenuBarDivider` are its rows [ServerMenuDashboardRows.swift:MenuBarStatusLine].
  - Pure state: `MenuBarServerPhase.resolve`, `MenuBarModelKind`, `MenuBarUnloadAction`, `MenuBarModelSlot`, `MenuBarSlotInput`, `MenuBarSlots.build`, `MenuBarMemoryReading`, `MenuBarUptime` [MenuBarServerState.swift:MenuBarServerPhase].
- Keychain and launch:
  - `ServerKeychain` and `TypeSafeKeychain` [ServerKeychain.swift:ServerKeychain].
  - `KeychainReadResult` and `KeychainSecretSync` [KeychainRead.swift:KeychainReadResult].
  - `AgentTerminalLaunch` (`findCLI`, `openInTerminal`, `writeAPIKeyFile`, `commandExportingAPIKey`, `copyToClipboard`) [AgentTerminalLaunch.swift:AgentTerminalLaunch].
- Child contracts: none (no subdirectories).

## Invariants
- A Keychain read distinguishes found, notFound and error. Only a successful read can lead to a delete, so a locked or denied Keychain can never erase a stored key. An empty field after a failed load is `.adopt` or `.none`, never `.delete` [KeychainRead.swift:KeychainSecretSync].
- Keychain use is opt-in per key (`serverKeyInKeychain`, `typeSafeKeyInKeychain` in UserDefaults). Without it the Keychain is neither read nor written at launch and the key is memory-only. `resolveOptIn` keeps a key stored by an earlier version and records consent. `itemExists` asks for attributes only, so it cannot raise an authorization prompt [ServerKeychain.swift:resolveOptIn]. TypeSafe generates a fresh in-memory key per launch unless opted in [TypeSafeKeychain.swift:initialKey].
- The server key and the TypeSafe key use different service names (`TurboSpark.server`, `TurboSpark.typesafe`) and are never shared. An empty key passed to `saveKey` deletes the item [TypeSafeKeychain.swift:saveKeyStatus].
- A server option read at start time is disabled while the server runs. The HF mirror endpoint stays live, and also calls `TurboSparkCatalog.setHfEndpoint` on change [ServerAdvancedSettingsView.swift:ServerAdvancedSettingsView].
- The Memory guard picker is built from `AppLoadGuardOption.allCases` and the reasoning picker goes through `model.setReasoning` with `availableReasoningLevels`, so Custom is never dropped and per-model memory is written [ServerAdvancedSettingsView.swift:ServerAdvancedSettingsView].
- The API key reaches a launched Terminal through a 0600 file created with its final mode and deleted by the shell before the command runs, so it never appears in command text, scrollback, history or `ps` [AgentTerminalLaunch.swift:writeAPIKeyFile]. If Terminal cannot be scripted the command is copied to the clipboard [ServerAdvancedSettingsView.swift:launchAgent].
- The daemon is a separate process that inherits nothing. `ServerDaemonLaunch.args` always passes `--guardrails`, omits `--port` when the port is 0 (automatic), and passes `--api-key` only when set [ServerDaemonLaunch.swift:args].
- `ServerPortInput` maps empty to 0 (automatic) and rejects typos instead of mapping them to automatic. Valid ports are 1...65535 [ServerPortInput.swift:parse]. `ServerConnectionView` sets `serverPortIsValid` and clears stale error state when it syncs [ServerConnectionView.swift:syncPort].
- `ServerEndpointCatalog.all` must mirror the router's route table (`turbospark_server::build_router_with_options`), and a test enforces it. Only implemented routes are listed [ServerEndpointCatalog.swift:ServerEndpointCatalog].
- `ServerConnectRecipes` escapes every interpolated value for its context (POSIX double quote, POSIX single quote, JSON or Python string), because the API key and model id land in pasted shell commands. A nil key becomes the placeholder `unused`, never an omitted header [ServerEndpointCatalog.swift:snippets].
- `ServerMetricsStore` is bounded (2,000 records) and `trim()` runs on every `requestStarted`, shifting the `open` indices. An event for a request whose start was never seen is dropped, not invented. A retry's counters accumulate. `droppedEvents` sums per-poll counts. Throughput is total tokens over total decode seconds, and percentiles are nearest-rank. `queuedSeconds` and `tokensPerSecond` are nil when their inputs are missing [ServerMetricsStore.swift:ServerMetricsStore].
- `ServerLiveHistory` keeps at most 120 points (about one minute at the 2 Hz server poll) and computes a rate only from non-decreasing counters. A missing sample resets the baseline [ServerLiveHistory.swift:sample].
- `ServerFavorite` has two lists aligned by index (`modelPaths` and `modelInstallIds`). A placeholder row (another app's folder) has an empty id slot and never answers an id lookup. Trailing empty ids are trimmed. Resolution order is exact install id, then the single row sharing the library-independent part, then the legacy path. Decoding tolerates an absent or malformed `modelInstallIds`. No keys or traffic are stored [ServerFavorite.swift:ServerFavorite].
- `loadServerFavorite` starts only when no server is running and none is busy. A drive that is away warns and attaches nothing, while the server still starts [ServerFavorite.swift:loadServerFavorite].
- Menu bar phase precedence is fixed: running (working, degraded, idle, running), then starting, failed, daemon, stopped. A start in flight wins over a stale failure. `residentBytes` is a deliberate stub (always nil), and `MenuBarMemoryReading.budgetBytes` is a stub (nil) so the bar scales against physical RAM [MenuBarServerState.swift:MenuBarServerPhase].
- A chat session already served is not listed twice in the Loaded list, and an image pipeline that is open takes precedence over served image ids [MenuBarServerState.swift:MenuBarSlots].
- Daemon verbs run on a detached task guarded by `daemonBusy`. Success and failure are reported from the real outcome [ServerMenuBarView.swift:runDaemonVerb].
- The playground validates before sending (unique non-empty question ids, instructions, paired true/false criteria, at least two score levels, `__none__` option handling, JSON state shape). Score level rows use a bounds-checked binding because removal can leave a row alive for one pass [TypeSafePlaygroundCards.swift:TypeSafeQuestionCard].
- The Audio pane requires an explicit consent toggle and a reference WAV for MegaTTS3 native cloning, and choosing a new reference resets consent. Synthesize stays disabled until both are set [AudioAPIPaneView.swift:chooseReference].
- Status words that the model layer stores as English strings are mapped through the catalog via `APIStatusText` or the views, and unknown text renders verbatim. Any new UI string needs a catalog entry (`Text(..., bundle: .module)` or `AppStrings.localized`) for the localization parity gates.

## Side Effects
- Keychain: `SecItemCopyMatching`, `SecItemAdd`, `SecItemUpdate`, `SecItemDelete` on generic-password items `TurboSpark.server` and `TurboSpark.typesafe` [ServerKeychain.swift:saveKeyStatus].
- UserDefaults: the two opt-in keys, and `@AppStorage("TurboSpark.server.consoleHeight")` [ServerPaneView.swift:ServerPaneView].
- Network: `URLSession.shared` test requests to the local server (`/v1/audio/speech`, `/v1/images/generations`, `/v1/embeddings`) with the Bearer key [AudioAPIPaneView.swift:runTest]. TypeSafe requests go through `service.client.evaluate` [TypeSafePlaygroundView.swift:runRequest]. HF endpoint changes call `TurboSparkCatalog.setHfEndpoint`.
- Process spawn: `/usr/bin/osascript` to script Terminal.app [AgentTerminalLaunch.swift:openInTerminal]. The daemon verbs call `TurboSparkDaemon.start`, `stop`, `restart` and `status` through FFI [ServerMenuBarView.swift:startDaemon].
- Disk: the 0600 key file in `NSTemporaryDirectory()` [AgentTerminalLaunch.swift:writeAPIKeyFile]. Audio test reads the chosen reference WAV. Image and audio test results are saved or played in memory, and `saveAPIImageToGallery` writes the image to the gallery. `AudioCatalog.installed()` reads the audio install registry [AudioAPIPaneView.swift:refreshInstalled].
- Clipboard: `NSPasteboard` writes for addresses, keys, snippets, console rows and diagnostics.
- Timers and tasks: a 0.1 s `Timer` for audio playback [AudioAPIPaneView.swift:togglePlayback], a 2 s polling `.task` for TypeSafe models [TypeSafeAPIPaneView.swift:TypeSafeAPIPaneView], sleeps for copy feedback [ServerConsoleView.swift:copyAll], and `model.startServerPolling()` on appear [ServerPaneView.swift:ServerPaneView].
- Mach: `task_info(TASK_VM_INFO)` for the process physical footprint [ServerLiveHistory.swift:currentPhysFootprintBytes].
- App state: mutates `AppModel` (server, favorites, settings persistence via `persistSettingsDebounced`, toasts, window navigation, attach and detach calls) and persists favorites in the settings store [ServerFavorite.swift:saveServerFavorite].
- No NotificationCenter use in this directory.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
- UI changes here also need `make design-check` (see the root AGENTS.md).
