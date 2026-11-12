# BrowserAutomation

## Overview
Agent-driven browser control for the macOS app. The directory has four layers:

- A backend-neutral command protocol (closed command, result, and error enums) and a backend manifest.
- A single-lease automation session actor that serializes one command at a time through a tab-bound port.
- An app-owned WebKit engine that owns one `WKWebView` per tab, gates every navigation through an exact-origin authorizer, and routes JavaScript dialogs, popups, and user takeover.
- A DOM snapshot bridge (Swift service plus an injected JavaScript walker) that mints per-document element references and performs actions.

Around those sit origin grants stored in project permissions, one-shot approvals, private-network isolation for agent tabs, and an allowlisted diagnostics projection. `AppBrowserAutomationCoordinator` wires engine, tab store, and session together for the app shell. `BrowserTabStore`, `BrowserTabID`, `BrowserAgentControlToken`, `BrowserToolRuntime`, `BrowserToolDefinitions`, and `BrowserDialogPolicy` are used here but defined elsewhere.

## Interface
- Command protocol [BrowserControlCommand.swift]
  - `BrowserControlCommandKind`, `BrowserControlProtocol` (versioned vocabulary, `currentVersion` 1), `BrowserControlCommand` (navigate, click, type, pressKey, scroll, screenshot, readState, waitFor, setViewport), `BrowserControlRequest`, `BrowserWaitTarget`, `BrowserLoadState`, `BrowserScrollDirection`, `BrowserSnapshotScope`, `BrowserElementWaitCondition`.
- Port, results, errors [BrowserControlPort.swift]
  - `BrowserControlPort` protocol (`perform`, `isWaitConditionSatisfied`, `cancel`), `BrowserBackendManifest`, `BrowserControlResult`, `BrowserControlValue`, per-command result structs (`BrowserNavigationResult`, `BrowserSnapshotResult`, `BrowserScreenshotResult`, and others), `BrowserControlError` and `BrowserControlErrorKind`, `BrowserNavigationFailureReason`, `BrowserInputErrorReason`.
- Session [BrowserAutomationSession.swift]
  - `BrowserAutomationSession` actor: `attach(to:)`, `perform(_:using:timeoutSeconds:)`, `detach()`, `userTookOver(tabID:)`, `tabClosed(tabID:)`, `cancelCurrentCommand()`, `attachedTabID()`.
  - `BrowserAutomationSessionError`, `BrowserAutomationClock`, `SystemBrowserAutomationClock`, `BrowserControlPortFactory`. `defaultTimeoutSeconds` is 10 and `maximumTimeoutSeconds` is 60.
- WebKit port [WebKitBrowserControlPort.swift]
  - `WebKitBrowserControlPort` (backend id `webkit-macos`, `supportedCommands`, `manifest`). It supports every command except `setViewport`, rejects full-page screenshots, and implements `waitFor` through the session rather than `perform`.
- WebKit engine [WebKitBrowserEngine.swift]
  - `WebKitBrowserEngine`: `createTab`, `navigate`, `navigateAsUser`, `prepareForDirectUserInput`, `stopLoading`, `closeTab`, `recoverCrashedTab`, `captureScreenshot`, `webView(for:)`, `resolvePendingDialog`, `cancelPendingDialog`, published `tabStates` and `pendingDialogs`, `onDialogResolved`.
  - Navigation authorization types: `BrowserNavigationAuthorizer`, `BrowserNavigationAuthorizationRequest`, `BrowserNavigationAuthorizationDecision`, `BrowserNavigationCheckPhase`, `BrowserNavigationDestination`, `BrowserUserTakeoverHandler`.
  - Dialog types: `BrowserDialogRequest`, `BrowserDialogDecision`, `BrowserDialogResolution`, `BrowserDialogResult`, `BrowserJavaScriptDialogKind`, `BrowserDialogCancellationReason`, `BrowserDialogPolicyProvider`.
  - State and errors: `BrowserEngineTabState`, `WebKitBrowserEngineError`.
- DOM snapshot bridge [DOMSnapshotService.swift, DOMSnapshotTypes.swift, DOMSnapshotScript.swift]
  - `DOMSnapshotService`: `snapshot()`, `committedGeneration()`, `resolve(locator:generation:)`, `perform(_:on:generation:)`, `pickCandidate(at:)`, `isWaitConditionSatisfied(reference:condition:)`, `navigationStarted`, `navigationCommitted`, `invalidate()`.
  - `DOMSnapshotEnvelope`, `DOMSnapshotNode`, `DOMSnapshotBounds`, `DOMSnapshotLocator`, `DOMSnapshotAction`, `DOMSnapshotActionOutcome`, `DOMSnapshotResolvedTarget`, `DOMSnapshotPickCandidate`, `DOMSnapshotServiceError`.
  - `DOMSnapshotService.walkerScript`, the injected JavaScript object `__tsSnapshotV1` (`snapshot`, `resolve`, `waitCondition`, `resolveLocator`, `perform`, `pickAt`).
- Permissions [BrowserPermissionRuleStore.swift, BrowserPermissionApprovalCoordinator.swift]
  - `BrowserOrigin` (canonical HTTP(S) origin), `BrowserPermissionContext`, `BrowserPermissionOwner`, `BrowserPermissionRuleDecision`, `BrowserPermissionGrantResult`, `BrowserPermissionRuleStore` (`decision`, `grant`, `revoke`, `maximumGrantCount` 256).
  - `BrowserPermissionApprovalRequest`, `BrowserPermissionApprovalCoordinator` (`authorizeOnce`, `allowsOnce`, `cancel`), `BrowserPermissionApprovalActionResult`.
  - `AppModel` extension: `browserPermissionContext(for:project:)`, `browserPermissionApprovalRequest(for:)`, `approvePendingBrowserToolCall(id:alwaysAllowOrigin:)`, `denyPendingBrowserToolCall(id:)`.
- Isolation [AgentBrowserIsolation.swift]
  - `AgentBrowserNetworkRules.shared` (`withRules`, `compiledRules`, `privateHostPatterns`, `encodedRules`) and `AgentBrowserDataStore.store`.
- App shell wiring [AppBrowserAutomationCoordinator.swift]
  - `AppBrowserAutomationCoordinator` (`openPane`, `closePane`, `paneWillClose`, `runtime(for:project:availability:)`, `userTookOver(tabID:)`) and `BrowserRuntimeActionScope`.
- Diagnostics [BrowserAutomationLogger.swift, BrowserAutomationSupportReportAdapter.swift]
  - `BrowserAutomationDiagnosticEvent`, `BrowserAutomationOutcome`, `BrowserAutomationLogCategory`, `BrowserAutomationLogLevel`, `BrowserAutomationLogSink`, `BrowserAutomationLogger`, `BrowserAutomationSupportReportAdapter.project(call:result:)`.

No child directory in this folder has an EGM.md.

## Invariants
- Session lease [BrowserAutomationSession.swift:BrowserAutomationSession]
  - At most one tab attachment and one active command. A second `perform` throws `commandInFlight`, and `attach` throws `commandInFlight` while a cancellation is in progress.
  - `attach` only accepts agent-owned tabs and checks the ownership token against the store (`ownershipLost`).
  - A stale attachment to a tab the store no longer has is dropped before any new attach.
  - Every command gets a deadline (default 10 s, capped at 60 s, non-finite or non-positive values throw `invalidInput(.invalidTimeout)`). A `waitFor` that expires surfaces as `waitConditionNotMet`, any other expiry as `timeout`.
  - `cancelActive` clears `activeCommand` and sets `cancellationInProgress` before awaiting `port.cancel`, then resumes the continuation exactly once. `complete` ignores a result whose request id is not the active one, and a value with a mismatched request id becomes `unexpectedResponse`.
- Result and protocol shape [BrowserControlPort.swift:BrowserControlResult, BrowserControlCommand.swift:BrowserControlProtocol]
  - `BrowserControlResult.init` preconditions that the value kind matches the request command kind.
  - `screenshotPNGData` is excluded from Codable output and decoding.
  - Command vocabulary is versioned. Add a new snapshot and advance `currentVersion` instead of editing version 1.
  - Dialog resolution is deliberately not a command and stays in app-owned UI.
- Port threading [WebKitBrowserControlPort.swift]
  - Cancelled command ids are remembered under `NSLock`. A cancelled id can never be registered again and `cancel` is idempotent.
  - WebKit work hops to `@MainActor`, and `cancel` dismisses the tab's pending dialogs and stops loading for `navigate`.
  - Element actions use `committedGeneration()` rather than a fresh snapshot, so references come from an earlier `readState`.
  - `retainingReferencedNodes` must keep `parentIndex` pointing at the nearest retained ancestor. Its hop bound guards against cyclic snapshots.
- Origin parsing [BrowserPermissionRuleStore.swift:BrowserOrigin]
  - Only http and https, with no userinfo, path, query, fragment, wildcard, or empty port suffix.
  - A trailing dot on the host is stripped and the port is normalized, so equivalent spellings compare equal through `canonicalString`.
  - The allowlist is capped at 256 entries and is re-normalized on every grant and revoke (invalid and duplicate entries dropped).
- Permission decision order [BrowserPermissionRuleStore.swift:decision]
  - User-owned tabs return `userOwnedNavigation` before any agent policy.
  - Read-only mode or `browser == .deny` denies.
  - A nil origin asks.
  - An approved current action or an allowlisted origin allows. Everything else asks.
- One-time approvals [BrowserPermissionApprovalCoordinator.swift]
  - An approval is keyed by call id and matches only the exact project id and origin it was created for.
  - `browserPermissionContext` must stay side effect free because SwiftUI evaluates it while drawing approval cards, so it never uses the registry provider that opens the pane.
  - `approvePendingBrowserToolCall` refuses while `generating` or `submitting`, and persists an always-allow grant only if `grant` returns `added` or `alreadyGranted`.
- Coordinator ownership [AppBrowserAutomationCoordinator.swift]
  - A stale agent tab (closed, or no longer agent-owned) is released from the session before a replacement tab is created. Otherwise `attach` would fail with `alreadyAttached` on every later call.
  - `activeAction` guards re-entry and is set with no `await` between the check and the assignment.
  - Takeover clears `agentTabID`, the token, and the active action. A new tab needs a new token.
  - `runtime(for:)` returns nil when the browser tool is disabled or no origin is known, so the call fails closed.
- Navigation authorization [WebKitBrowserEngine.swift:WebKitBrowserEngine]
  - Every main-frame action, form submission, redirect, main-frame response, subframe action, and subframe response passes the injected authorizer for agent-owned tabs. User-owned tabs are allowed without it.
  - A destination that is not an HTTP(S) origin is cancelled. An `about:` subframe inherits its creator's origin.
  - A fragment-only change skips load bookkeeping, since it never commits.
  - `pendingPolicyGates` are all resolved `false` on takeover, close, and content-process crash.
  - `authorizedFrameOrigins` is recorded only for agent-owned authorization. A main-frame document resets it, a subframe adds to it.
  - A screenshot is refused (`unapprovedFrameOrigin`) when any iframe origin in the document is not in that set. It is retried up to 3 times on blank or changed-URL captures.
  - `navigate` bumps `screenshotRevision`, so stale captures are discarded.
  - Ownership never returns to the agent. Direct input transfers the tab to the user, calls `onUserTakeover`, dismisses dialogs with `userTakeover`, and removes the content rules.
  - Popups from agent tabs are denied by `BrowserTabStore.requestPopup` (`deniedAgentPopup`). Allowed popups open as a new managed tab and navigate through the same gate.
- Agent tab isolation [AgentBrowserIsolation.swift]
  - An agent tab loads nothing until `AgentBrowserNetworkRules` is installed on its webview, and navigation fails with `navigationFailure(.unknown)` if compilation failed (fail closed).
  - Each private host form is a separate rule for both `https?` and `wss?` and both plain and userinfo spellings. Rules are `unless-top-url` private, so a private top document may load its own private subresources.
  - Numeric host spellings cannot be expressed in the rules. Only the navigation delegate stops those frames.
  - Agent tabs use `AgentBrowserDataStore.store`, never the user's default store. Under `AppStorageRoot.isRunningTests` it is non-persistent.
- Dialogs [WebKitBrowserEngine.swift:handleJavaScriptDialog]
  - The policy provider decides: auto-accept, auto-dismiss, or ask. An ask-policy dialog stays pending until resolved, cancelled, or timed out. The timeout is capped at 60 s.
  - Each gate resolves its WebKit completion at most once.
- Snapshot bridge [DOMSnapshotService.swift]
  - Snapshots, references, and actions are tied to a document generation. `navigationStarted` mints a new generation and deactivates the old one immediately, and `navigationCommitted` only activates the current generation.
  - Every `await` on WebKit is followed by a re-check of `isInstalled`, `isActive(generation:)`, and `expectedOrigin`.
  - The bridge runs in a named `WKContentWorld`. `userContentController(_:didReceive:)` rejects any message not from the main frame, the expected origin, the same webView, or the current generation, and enforces exact key sets per message type.
  - The snapshot byte budget (default 64,000 bytes, 500 nodes) is measured with `withoutEscapingSlashes`. JS and Swift must keep the same encoding, and bounds are rounded to integers for that reason.
  - References are 32 lowercase hex characters. Locators, key names, text (4096 bytes), scroll amount (1 to 5000), and pick candidates are validated on both the Swift and JS sides. Keep the two validators in step.
  - A detached or replaced reference satisfies `hidden` only.
  - Password inputs are named "Password field" in snapshots and are not read.
  - The locator walk is limited to 5000 nodes and reports ambiguity rather than failing once a first match exists.
- Diagnostics [BrowserAutomationLogger.swift, BrowserAutomationSupportReportAdapter.swift]
  - The event shape is closed: action kind, canonical origin, outcome, and duration. Decoding rejects a non-canonical origin.
  - The adapter never reads transcript output or image references. It drops running and pending states, non-finite or negative durations, and non-canonical origins.
  - Sink write failures are counted, never thrown.

## Side Effects
- WebKit [WebKitBrowserEngine.swift]
  - Creates `WKWebView` instances and loads URLs, installs and removes content rule lists, and runs JavaScript (frame-origin scan) in the page.
  - Takes snapshots and installs KVO observations on progress, title, and url.
  - Starts a `Task.sleep` timeout per ask-policy dialog.
- Content rule compilation [AgentBrowserIsolation.swift:AgentBrowserNetworkRules]
  - Compiles into `WKContentRuleListStore.default()` under identifier `AgentBrowserPrivateNetworkBlock`, and the engine init warms it.
- Disk [AgentBrowserIsolation.swift:AgentBrowserDataStore]
  - Reads and writes `agent_browser_store_id` under `AppStorageRoot`, which names a persistent `WKWebsiteDataStore`. Cookies and sessions in agent tabs persist there.
- Settings read [WebKitBrowserEngine.swift:init]
  - The default dialog policy provider calls `MacAppSettingsFileStore.load()` each time a dialog opens.
- Page bridge [DOMSnapshotService.swift]
  - Registers a script message handler and a user script in a private content world. `invalidate()` removes the handler.
  - Page-side JavaScript dispatches clicks, input, change, keyboard, and wheel events, sets input values, focuses elements, scrolls, and calls `form.requestSubmit()`.
- Session [BrowserAutomationSession.swift]
  - Spawns an operation task and a deadline task per command and sleeps on the injected clock. `detach` transfers tab ownership to the user in the store.
- Project state [BrowserPermissionApprovalCoordinator.swift]
  - `approvePendingBrowserToolCall` writes the origin grant through `updateProject` and mutates `pendingToolCallProject`.
  - It then approves or denies the pending tool call, which triggers tool execution.
- Coordinator [AppBrowserAutomationCoordinator.swift]
  - Creates browser tabs (the user's pane tab and the agent tab) and publishes `ObservableObject` changes.
- Diagnostics [BrowserAutomationLogger.swift]
  - Calls the injected `BrowserAutomationLogSink.record` for each transition. No concrete sink lives in this directory.
- Network, Keychain, UserDefaults, NotificationCenter, FFI, and process spawn
  - None are issued directly here. Page loads go through WebKit.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
