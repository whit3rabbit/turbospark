# Diagnostics

## Overview
The right-hand "Model Settings" inspector of the macOS app. `InspectorView` stacks a SwiftUI grouped `Form` of sections that all read and write one `AppModel`: model picker and load/unload, Essentials (thinking level and context window), open/architecture options, directional steering, generation sampling, session telemetry, last-run diagnostics, and a recorded token-usage dashboard. The directory also holds `MetricFormat`, the shared formatter for bytes, durations and rates used by the inspector and by other app views. It renders engine facts (reasoning levels, context ladder, speculation, steering, telemetry) as sent by the loaded session and computes none of them itself.

## Interface
- Container: `InspectorView` (`@MainActor`, takes `AppModel`); its sections are mostly `extension InspectorView` computed properties so they can share `model` [InspectorView.swift:InspectorView].
- Sections declared as extensions of `InspectorView`:
  - `modelSection`, `memoryAndPowerSection` ("Open & Architecture Options": cache slots, power profile, speculation, drafter, rate cap, Apply & Reload), `steeringSection` [InspectorOptionsSection.swift:modelSection].
  - `telemetrySection` (hardware RAM, chip, thermal, plus `SessionInfo` dialect, vocab, context, slots, speculation, steering, stop tokens) [InspectorTelemetrySection.swift:telemetrySection].
  - `usageDashboardSection` (wraps private `UsageDashboardSectionView`: totals, 12-week heatmap, 30-day Charts bar chart, top 5 chats) [UsageDashboardSection.swift:usageDashboardSection]. `InspectorView.weekday(of:)` is the testable weekday helper [UsageDashboardSection.swift:weekday].
- Sections that are their own structs (they hold `@State`):
  - `InspectorEssentialsSection` (thinking level, context window) [InspectorEssentialsSection.swift:InspectorEssentialsSection].
  - `ContextLadderPicker` plus `static ladderKey(path:slots:loadGuard:)` [InspectorEssentialsSection.swift:ContextLadderPicker].
  - `ContextSelectionButton`, which opens the file-private priced popover [ContextSelectionPopover.swift:ContextSelectionButton].
  - `GenerationSamplingSection` (scope picker, sampling knobs, preset registry) [GenerationSamplingSection.swift:GenerationSamplingSection].
  - `RunnerDiagnosticsSection` with `stopReasonText(_:)`, `compactionBoundaryRows(for:)` and value type `RunnerDiagnosticValue` [RunnerDiagnosticsSection.swift:RunnerDiagnosticsSection].
- Formatting: `MetricFormat` (`seconds`, `milliseconds`, `rate`, `percent`, `memory`, `storage`, `transferBytes`, `fileSize`); nil durations and memory render as an em-dash character `\u{2014}` [MetricFormat.swift:MetricFormat].
- Subdirectories: none.

## Invariants
- Reasoning levels come only from `model.availableReasoningLevels` (probed from the checkpoint's template at open), never `Reasoning.allCases`. Offering a level the template rejects (for example `.high` on Qwen 3.8) fails the turn. When unavailable, the section explains whether the model is not loaded or the template has no levels [InspectorEssentialsSection.swift:ThinkingLevelControl].
- The context ladder is rendered, never computed: no figure is scaled from another, because KV cost is not linear in the window (sliding-window rings stop growing). An empty ladder shows no priced rows and no zeros, and falls back to the unpriced `AppContextLengthOption` presets. Auto and Custom are always offered [InspectorEssentialsSection.swift:ContextLadderPicker, ContextSelectionPopover.swift:ContextSelectionPopover].
- Rungs with verdict `.refused` are disabled in the popover; rungs past the trained window only warn, they are not refused [ContextSelectionPopover.swift:rungRow, InspectorEssentialsSection.swift].
- `maxContextTokens == 0` means Auto. `isCustom` is a sticky stored flag: while Custom and tokens are non-zero, `syncIsCustom` never clears it. Every explicit row tap clears it. The custom slider snaps to 256/512/1024 steps and clamps to 512...131072 [InspectorEssentialsSection.swift:syncIsCustom, customSizeRow].
- The ladder task id must include model path, cache slots and load guard (`ladderKey`), otherwise a slots or guard change leaves a stale ladder. The ladder is priced with the same trio the loader uses, so verdicts match the open decision [InspectorEssentialsSection.swift:ladderKey, loadLadder].
- Sampling edits go through one `scopeSettings` binding: scope `.thisChat` writes `setChatSamplingOverride`, `.appDefaults` writes `applyGlobalSampling`. Reasoning stays app-wide, since overrides cover sampling only. A preset is applied to the scope being edited, never the other. Saving under an existing name replaces that preset, a new name mints a new `UUID` [GenerationSamplingSection.swift:scopeSettings, applyPreset, savePreset].
- Scope re-syncs on appear and on `selectedChatID` change: a chat with an override opens on "This chat" [GenerationSamplingSection.swift:syncScopeWithSelection].
- Controls are disabled while `model.isRunning` (sampling, open options, steering) and, for model and Essentials sections, also while `isInstallingModel` [InspectorOptionsSection.swift, GenerationSamplingSection.swift, InspectorEssentialsSection.swift].
- Steering resolves once at open: editing fields does not affect a loaded model, so `steeringNeedsReload` shows a Reload affordance, and an unsupported family (`steeringFamilySupported == false`) is shown with the engine's own reason rather than hidden [InspectorOptionsSection.swift:steeringSection].
- Rate cap `0` is the uncapped sentinel; the clamp target stepper intentionally has no range because the value is control-vector dependent [InspectorOptionsSection.swift:memoryAndPowerSection, steeringSection].
- Usage dashboard shows recorded usage only (`totals.turns == 0` shows an empty message, no estimates). Day keys are UTC `yyyy-MM-dd` and the weekday is computed in a UTC calendar, so the heatmap's first column aligns; unparseable keys land on Monday (row 1). Ghost chats and chats without `usage` are excluded from Top chats [UsageDashboardSection.swift:weekday, topChats, DateFormatter.utcDay].
- Engine-sent strings (steering summary and reason, speculation reason, dialect, chip, unknown thermal level) are shown verbatim; only our own fallbacks are localized. Stop reasons and thermal levels are mapped to catalog `Text` rather than shown as wire identifiers, and compaction boundary labels use `AppStrings.localized` [RunnerDiagnosticsSection.swift:stopReasonText, compactionBoundaryRows, InspectorTelemetrySection.swift:thermalLevelText].
- `MetricFormat.storage` floors at MB (model installs); use `fileSize` for single documents. `transferBytes` clamps non-finite, negative and overflowing rates [MetricFormat.swift].
- `InspectorView` is explicitly `@MainActor` because only `body` is isolated by the protocol on the macOS 14 SDK. Theme-dependent bar color uses `theme.isDark` from the environment, not `NSApp.effectiveAppearance` [InspectorView.swift].
- Every user-visible string is a bundled `Text(..., bundle: .module)` catalog key (localization parity gates apply).

## Side Effects
- FFI: `TurboSparkCatalog.contextLadder(modelPath:expertCacheSlots:loadGuard:)` walks the install directory, run in `Task.detached(priority: .userInitiated)` [InspectorEssentialsSection.swift:loadLadder].
- stderr: a failed ladder load writes "context ladder unavailable for <path>" to `FileHandle.standardError` [InspectorEssentialsSection.swift:loadLadder].
- Pasteboard: the copy-path button clears and sets `NSPasteboard.general` [InspectorOptionsSection.swift:modelSection].
- Panel: `ModelLocationPicker.choose(for:)` opens a folder picker [InspectorOptionsSection.swift:modelSection].
- Model mutations through `AppModel`: `selectModel`, `loadModel`, `unloadModel`, `reloadModel`, `reloadForSteering`, `setReasoning`, `maxContextTokens`, `runtimeOptions.*`, sampling apply/override/preset upsert/delete (persistence is debounced inside `AppModel+Sampling`, not here), and `selectChat` from Top chats [various files in this directory].
- No UserDefaults, Keychain, network, process spawn or timers in this directory.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
