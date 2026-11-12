# Browser

## Overview
UI and state layer for the in-app browser and its Settings. The WebKit engine (`WebKitBrowserEngine`, `BrowserEngineTabState`, `DOMSnapshotService`, `BrowserPermissionRuleStore`, `ChromeBookmarkImporter`) lives outside this directory. Here live: the thread-safe tab metadata store that arbitrates user versus agent ownership of tabs, the SwiftUI pane (tab strip, toolbar, address field, failure and loading overlays, fixed-viewport host), the element picker that turns a clicked page element into a chat attachment, the page-dialog approval overlay, the persisted browser settings model, and the Chrome bookmark import preview flow.

## Interface
- Tab metadata and agent control [BrowserTabStore.swift:BrowserTabStore]
  - Value types: `BrowserTabID`, `BrowserTabOwner` (`user`/`agent`), `BrowserTabLoadState`, `BrowserTab`, `BrowserPopupRequestResult`, `BrowserTabStoreError` [BrowserTabStore.swift:BrowserTab]
  - `BrowserAgentControlToken`: opaque lease token, only `BrowserTabStore` can mint one [BrowserTabStore.swift:BrowserAgentControlToken]
  - Store API: `createTab`, `selectTab`, `closeTab`, `startNavigation`/`completeNavigation`/`failNavigation`, `markCrashed`/`restoreCrashedTab`, `transferOwnershipToUser`, `acquireAgentControl`, `agentControlledTab(for:)`, `requestPopup` [BrowserTabStore.swift:BrowserTabStore]
- Persisted settings (`browser` object in settings.json) [BrowserSettings.swift:BrowserSettings]
  - `BrowserDialogPolicy` (`auto-accept`, `auto-dismiss`, `ask`), `BrowserViewportPreference`, `BrowserBookmark`, `BrowserBookmarkFolder`, `BrowserBookmarkTree` [BrowserSettings.swift:BrowserDialogPolicy]
  - `BrowserSettings.replaceBookmarks` and `bookmarksSizeLimit` (2 MiB) [BrowserSettings.swift:BrowserSettings]
- Settings pane [BrowserSettingsPaneView.swift:BrowserSettingsPaneView]
  - `BrowserSettingsPaneViewModel` with `BrowserFeatureEnableRequest` (first-enable onboarding gate) [BrowserSettingsViewModel.swift:BrowserSettingsPaneViewModel]
  - `BrowserProjectPermissionsSettingsViewModel` with `BrowserOriginSettingsFeedback` (project origin allowlist edits) [BrowserSettingsViewModel.swift:BrowserProjectPermissionsSettingsViewModel]
- Pane [BrowserPaneView.swift:BrowserPaneView]
  - `BrowserPaneModel` (selection, address draft, navigation intents, system-browser handoff), `BrowserTabView`, `BrowserTabPresentation` (`presentation(for:)` maps engine state to content/loading/failed/crashed) [BrowserPaneView.swift:BrowserPaneModel]
  - `BrowserPaneView.init(engine:)` and `init(engine:appModel:)`; `makeViewportController(appModel:)` [BrowserPaneView.swift:BrowserPaneView]
- Viewport [BrowserViewportController.swift:BrowserViewportController]
  - `BrowserViewportMode`, `BrowserViewportLayout`, ranges `widthRange` 320...3840, `heightRange` 240...2160, `zoomRange` 0.5...3.0, `cssPoint(viewPoint:scrollOffset:scale:)`, `apply(to:containerSize:)` [BrowserViewportController.swift:BrowserViewportController]
  - `BrowserViewportWebView` wraps the `WKWebView` in an `NSScrollView` host [BrowserViewportController.swift:BrowserViewportWebView]
- Element picker [ElementPickerController.swift:ElementPickerController]: `startPicking`, `updateCandidate`, `selectCandidate`, `attachSelectedCandidate`, `cancel`, `canStartPicking`, `canConfirmSelection`; injected closure typealiases `CandidatePicker`, `CandidateValidator`, `TabReadiness`, `UserTakeover`, `AttachmentSink`.
- Dialog approval [BrowserDialogApprovalView.swift:BrowserDialogApprovalView]: `BrowserDialogApprovalActions`, `originLabel(_:)`.
- Chrome bookmark import [ChromeBookmarkImportPreviewView.swift:ChromeBookmarkImportPreviewViewModel]: `ChromeBookmarkImportPreviewViewModel`, `ChromeBookmarkImportPreviewView`, `ChromeBookmarkImportStage`, `ChromeBookmarkImportFeedback`, `ChromeBookmarkPreviewRow`.
- Subdirectories: none.

## Invariants
- `BrowserTabStore` state is guarded by one `NSRecursiveLock` because the automation actor reads it off the main thread while WebKit callbacks and UI mutate it on main. Recursion is required because `requestPopup` calls `createTab`. Every read and write, including the lease check-then-act, must go through the lock [BrowserTabStore.swift:BrowserTabStore].
- At most one agent-control lease exists. `acquireAgentControl` requires an agent-owned tab, returns the existing token when the same tab reacquires, and throws `agentControlAlreadyAssigned` for another tab. Closing the tab or `transferOwnershipToUser` clears the lease. `agentControlledTab(for:)` fails with `invalidAgentControlToken` once the lease is gone [BrowserTabStore.swift:acquireAgentControl].
- Popups from agent-owned tabs are denied without creating a tab; only user-owned sources open a new user tab [BrowserTabStore.swift:requestPopup].
- Load transitions are guarded: `startNavigation` refuses a crashed tab, `completeNavigation` and `failNavigation` require `.loading`, `restoreCrashedTab` requires `.crashed`. `.restored` means metadata only, not that page state survived [BrowserTabStore.swift:BrowserTabLoadState].
- Closing the active tab selects the tab now at the same index, or the last one [BrowserTabStore.swift:closeTab].
- Any direct user action in the pane (back, forward, reload, stop, system-browser open, retry, viewport edit, element pick start) calls `engine.prepareForDirectUserInput` first so the engine takes the tab over from the agent and does not treat the action as an agent navigation [BrowserPaneView.swift:BrowserPaneModel].
- Address submission only reaches `engine.navigateAsUser`; `BrowserNavigationDestination` validation failure sets `isAddressInvalid`. The address field is not overwritten by `refresh()` while `isEditingAddress` is true [BrowserPaneView.swift:submitAddress].
- `BrowserPaneView` reads `engine.$tabStates` through `receive(on: DispatchQueue.main)` because `@Published` emits in willSet; a synchronous refresh would see stale state [BrowserPaneView.swift:BrowserPaneView].
- A pending page dialog forces its tab selected before the overlay shows, and the overlay is keyed by `request.id` so prompt text never carries to the next dialog. The dialog always shows the source origin, `?` when unknown [BrowserPaneView.swift:BrowserPaneView, BrowserDialogApprovalView.swift:originLabel]. Prompt text is sent only for `.prompt` kind [BrowserDialogApprovalView.swift:accept].
- Viewport popover edits are per pane session and never persisted; the initial value comes from `BrowserSettings.defaultViewport` [BrowserViewportController.swift:init]. `updatePreference` clamps to the ranges and rounds zoom to 0.01. In responsive mode the scroll offset reported is zero and zoom is 1. `cssPoint` clamps scale to at least 0.01 [BrowserViewportController.swift:normalized].
- `BrowserViewportPreference` falls back to 1280x800 at 1x for non-positive or non-finite values in its initializer, but its Decodable init throws on invalid persisted values, so `decodeLenient` in `BrowserSettings` replaces them with `.standard` [BrowserSettings.swift:BrowserViewportPreference].
- `BrowserSettings` decodes leniently; a legacy `viewport_pref` is ignored. Bookmarks over 2 MiB of encoded JSON decode to empty and `replaceBookmarks` refuses them (returns false, leaves old value). `bookmarks` is `private(set)` so the cap cannot be bypassed [BrowserSettings.swift:replaceBookmarks]. JSON keys: `dialog_policy`, `default_viewport`, `onboarding_done`.
- `BrowserSettingsPaneViewModel.update` applies edits on top of `latest()` (the live `AppModel` value), not its own copy, so a concurrent bookmark import is not wiped by a stale write. First enable requires `onboardingDone`; only `confirmFirstEnable` sets `enabled` and `onboardingDone` together [BrowserSettingsViewModel.swift:update].
- Project origin grants go through `BrowserPermissionRuleStore` (`normalizedAllowlist`, `grant`, `revoke`) so the settings UI and runtime checks share one canonical store; the pane text states 256 grants per project [BrowserSettingsViewModel.swift:BrowserProjectPermissionsSettingsViewModel].
- Element picker: disabled while the app is generating or submitting, and without a selected chat. Hover results are debounced 35 ms and discarded unless the request id, `isPicking`, and an empty selection still match. `candidate` is cleared before every hover lookup so a click cannot confirm the previous element. Attach re-validates the selection with the snapshot service (`resolve(reference:generation:)`) and rechecks session, confirmation, tab, chat and selection identity after each await; failure sets `.staleCandidate` and clears selection. Picked element text is HTML-escaped and wrapped in `<untrusted_browser_element>` because page content is untrusted [ElementPickerController.swift:attachment].
- Chrome import: previews start with nothing selected; selecting a folder selects its whole subtree, and deselecting a descendant also removes ancestor folder ids because folder ids select whole subtrees at import time. Only ids in `selectableIdentifiers` toggle. `confirmImport` is idempotent after a successful commit (`didCommitImport`), refuses an empty result, and `onSave` can refuse over the 2 MiB cap. Saved value is `existingBookmarks + report.trees`, an append not a replace [ChromeBookmarkImportPreviewView.swift:confirmImport].
- UI strings use `Text(..., bundle: .module)` or localization keys (see `localizationKey` in `BrowserOriginSettingsFeedback`); status uses theme styles.

## Side Effects
- Opens a URL in the default system browser through `NSWorkspace.shared.open`, injectable via `externalBrowserOpener` [BrowserPaneView.swift:BrowserPaneModel.init].
- Reads Settings default viewport from disk via `MacAppSettingsFileStore.load()` in `BrowserPaneView.init(engine:)` [BrowserPaneView.swift:BrowserPaneView].
- Writes settings: mutates `AppModel.browserSettings` and calls `persistSettingsDebounced()` from the settings pane and bookmark import [BrowserSettingsPaneView.swift:init, ChromeBookmarkImportPreviewView.swift:init(model:)].
- Reads Chrome profile bookmark files through `ChromeBookmarkImporter` (`enumerateProfiles`, `preview`, `importConfirmed`) [ChromeBookmarkImportPreviewView.swift:scanProfiles].
- Adds a prompt attachment to the selected chat via `AppModel.addPromptAttachment` and cancels/prepares engine user input [ElementPickerController.swift:init].
- Mutates WebKit state: `pageZoom`, `setFrameSize`, reparenting the `WKWebView` into a scroll document view [BrowserViewportController.swift:apply].
- Registers an `NSView.boundsDidChangeNotification` observer on the scroll view's clip view, removed in the coordinator `deinit` [BrowserViewportController.swift:BrowserViewportHost].
- Spawns `Task`s: hover debounce and attach confirmation in the picker [ElementPickerController.swift:updateCandidate]; Combine subscription to `AppModel.$generating` and `$submitting`.
- No network, Keychain or UserDefaults access in this directory.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
