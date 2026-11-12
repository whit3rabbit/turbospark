# Chrome

## Overview
The persistent window chrome of the macOS app: the always-visible navigation rail, the optional section panel beside it, the top bar (toggles, phase indicator, git pill, machine telemetry, unload and browser/inspector toggles), the bottom status strip (context fill, throughput, tokens, thermal, memory pressure), the chat model loader pill, the fan control popover and the profile/appearance menu. Everything here is SwiftUI reading `AppModel`; it owns almost no state of its own beyond hover, popover and polled-sample `@State`. Section panels themselves (`ChatSidebarView`, `AudioSidebarPanel`, `InstalledSidebarPanel`, `APISidebarPanel`, `ImagesSidebarPanel`) live in other directories and are hosted by `AppSidebarView`.

## Interface
- Left chrome
  - `AppSidebarView(model:isPanelVisible:isPanelPreferred:)`: rail plus a panel switched on `model.activeSection` [AppSidebarView.swift:AppSidebarView]
  - `NavigationRailView`: icon rail, `hasDestinationMenu(_:)`, settings link, hosts `ProfileMenuButton` [NavigationRailView.swift:NavigationRailView]
  - `ProfileMenuButton`: avatar menu for active profile, Mode, Text Size, Language [ProfileMenuButton.swift:ProfileMenuButton]
  - `SidebarNavigationGroup` (creation, models, utilities) and the `sidebarScale`, `sidebarRowHeight`, `sidebarActionSize` theme extensions, `SidebarDivider`, `SidebarActionButtonStyle` [SidebarPresentation.swift:SidebarNavigationGroup]
  - Shared panel building blocks: protocol `SidebarPanelMode`, `SidebarPanelList`, `SidebarPanelRow`, `SidebarModePanel`, `SidebarModeMenu` [SidebarPanelList.swift:SidebarPanelMode]
- Top and bottom bars
  - `TopBarView` [TopBarView.swift:TopBarView]
  - `ChromeTelemetryView` (memory, CPU, GPU wait) [ChromeTelemetryView.swift:ChromeTelemetryView]
  - `GenerationPhaseIndicator` [GenerationPhaseIndicator.swift:GenerationPhaseIndicator]
  - `StatusBarView` and `SampleRateGate` (internal, rate limits sparkline samples) [StatusBarView.swift:StatusBarView]
  - `FanReadoutView` (ThermalForge fan readout and popover) [FanReadoutView.swift:FanReadoutView]
- Model loader
  - `ModelLoaderControl` with `Density` (`regular`, `compact`) and `LoaderState` (`noneInstalled`, `notConnected`, `noneSelected`, `notLoaded`, `installing`, `opening`, `ready`) [ModelLoaderControl.swift:ModelLoaderControl]
  - `ModelStatusDot`, `ModelLoadProgressBar` [ModelLoaderIndicators.swift:ModelStatusDot]
- Private helpers: `MiniSparklineView`, `MeterBar` [StatusBarView.swift], `TSMeterBar` [ChromeTelemetryView.swift], `PulsingIndicatorDot` [GenerationPhaseIndicator.swift].
- Subdirectories: none.

## Invariants
- Navigation never hides: the rail is present in every section and state. The toggle hides only the panel. The rail takes the panel's background while a panel is visible and draws the hairline between them, so the pair reads as one surface [NavigationRailView.swift:NavigationRailView].
- `isPanelVisible` must come from `RootView`, which also feeds the window minimum width. Do not recompute it from the raw preference [AppSidebarView.swift:AppSidebarView].
- The panel never repeats the section list. Chat's panel is its conversation list; every other panel lists the section's modes. A section added to `hasSidebarPanel` needs its own case in `AppSidebarView.panel` and in `NavigationRailView.railMenu`, otherwise it falls to the unreachable `EmptyView` [AppSidebarView.swift:panel, NavigationRailView.swift:railMenu].
- When the panel is not preferred, a section with a destination menu (all panel sections except Chat) renders as a `Menu` listing the same modes, so hiding the panel never strands them. Menu selection sets the mode on `AppModel` and then `activeSection` [NavigationRailView.swift:railMenu, SidebarPanelList.swift:SidebarModeMenu].
- Rail ordering comes from `SidebarNavigationGroup` so both presentations keep the same order and shortcuts [SidebarPresentation.swift:SidebarNavigationGroup].
- Layout scales with the resolved UI font via `ResolvedAppTheme.sidebarScale` (min 1), not a fixed pixel size [SidebarPresentation.swift].
- Menu styles with a custom label must use `.menuStyle(.button)` plus `.buttonStyle(.plain)`, not `.borderlessButton`, which flattens the label on the macOS 14 SDK. Views using `body` isolation carry the explicit `@MainActor` annotation (Gotcha 45 cited in comments) [ModelLoaderControl.swift, NavigationRailView.swift, ProfileMenuButton.swift].
- `TopBarView` has no `.keyboardShortcut` on the sidebar or inspector toggles; the View menu commands own Cmd+Ctrl+S and Cmd+Shift+I, and duplicate shortcuts are undefined in SwiftUI. The inspector handler also dismisses an open attachment preview, which a direct button would skip [TopBarView.swift:sidebarToggle, inspectorToggle].
- The trailing and leading top-bar groups keep `frame(maxWidth: .infinity)` so the fixed-size telemetry centre stays still. Telemetry shows only when the bar is at least 760 scaled points wide [TopBarView.swift:body].
- The browser toggle appears only when `browserSettings.enabled` and the section does not hide chat panes; the inspector toggle follows `hidesChatPanes` [TopBarView.swift:body].
- Telemetry and status polling use a `.task` loop with `Task.sleep(2s)`, not a `Timer` stored on the struct, because struct re-creation (about every 25 ms while generating) restarted the timer and the poll never fired [StatusBarView.swift:body, ChromeTelemetryView.swift:body]. Per-token updates only feed the throughput sparkline through `SampleRateGate` (default 0.5 s interval; `refreshMetrics` passes `minInterval: 0`). The telemetry round trip stays on the 2 s timer [StatusBarView.swift:body].
- Telemetry honesty: no GPU percentage is invented; GPU wait comes from `diagnostics.phases.gpuWaitMs` and shows an em dash unless `phases.calls > 0`. Fan readout shows nothing (and no divider) when `FanController.isAvailable` is false [ChromeTelemetryView.swift:gpuWaitText, FanReadoutView.swift:body].
- Model loader: load and eject are separate hit targets, since eject discards the warm session. The glyph is `eject.fill`, not pause. "ctx" shows `model.resolvedContextTokens.formatted()` unabbreviated (matching `ContextLadderPicker`), not `info.maxContext`. `loaderState` priority: installing (only if no session), opening, empty loadable list (`notConnected` vs `noneInstalled`), no selection, then session state. The reasoning picker appears only when `model.reasoningPickerEnabled`; load shows only when `model.canLoadModel` and no session [ModelLoaderControl.swift:loaderState].
- Only the loadable model list is offered in the chooser; a model on an absent drive shows `notConnectedStatement` instead [ModelLoaderControl.swift:installedModelsSection].
- Auto-compact readout mirrors `AppChatCompaction.triggerNumerator/Denominator` over the usable window (context minus `maxNewTokens`), shown only within 40 percent of the trigger, urgent at 15 percent or less [StatusBarView.swift:compactionHint].
- Motion honours `AppearanceManager.shouldReduceMotion(systemReduceMotion:)` over the system value in the dot, progress bar and phase indicator. The pulse is a modifier so its `phaseAnimator` is torn down when the state changes [ModelLoaderIndicators.swift, GenerationPhaseIndicator.swift].
- Phase indicator renders nothing while idle (`statusText` nil) [GenerationPhaseIndicator.swift:statusText].
- Fan pin toggle writes both `FanController.keepFansPinnedOnQuit` (read by the quit path) and `AppModel.keepFansPinnedOnQuit` (persisted), then `persistSettingsDebounced()`; both must stay in sync [FanReadoutView.swift:controls].
- User-facing strings use `Text(..., bundle: .module)` or `AppStrings.localized`; identifiers (aliases, family, units) use `verbatim` [ModelLoaderControl.swift, StatusBarView.swift].

## Side Effects
- Posts `Notification.Name.toggleInspector` via `NotificationCenter.default` from the git pill [TopBarView.swift:gitPill].
- Mutates `AppModel`: `activeSection`, `imagesMode`, `installedModelKind`, `apiWorkspaceTab`, `setBrowserPaneOpen`, `unloadModel`, `unloadImageModel`, `unloadActiveOrAnyModel`, `loadModel`, `reloadModel`, `openChatWithModel`, `setReasoning`, `persistSettingsDebounced` [TopBarView.swift, ModelLoaderControl.swift, NavigationRailView.swift, FanReadoutView.swift].
- Calls `model.audioWorkspace.unloadModel()` [TopBarView.swift:unloadButton].
- Opens a model folder chooser through `ModelLocationPicker.choose(for:)` [ModelLoaderControl.swift:chooser].
- Process and system reads: `model.currentProcessMemoryBytes`, `currentProcessCPUUsage`, `refreshTelemetry()` on a 2 s `Task` loop [ChromeTelemetryView.swift:refresh, StatusBarView.swift:refreshMetrics].
- Spawns tasks to drive `FanController.shared.pinMax()` and `restoreAuto()`, which bridge to the `thermalforge` CLI (process spawn lives in `FanController`, outside this directory) [FanReadoutView.swift:controls].
- Writes appearance state: `AppearanceManager.shared.appearance`, `textSize`, `statusBarViewMode`, and `@AppStorage(AppLanguage.storageKey)` (UserDefaults) from the profile menu and status bar toggle [ProfileMenuButton.swift, StatusBarView.swift:viewModeToggleButton].
- Reads `UserProfileStore.active`, `NSFullUserName()` for the avatar [ProfileMenuButton.swift:profileDisplayName].
- No network, Keychain or direct disk access.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
