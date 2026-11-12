# Theme

## Overview
The app's design-system implementation. It resolves appearance settings into one environment value (`ResolvedAppTheme`), supplies a named type scale, status and accent colors with contrast math, shared controls and surfaces, a motion vocabulary, language and localization helpers, and right-column and chrome layout constants. The architecture has four parts:

- `AppearanceManager` (a `@MainActor` singleton) holds every user-facing appearance preference and persists it to `appearance.json`.
- `AppThemeInjector` (via `.appThemed()`) turns the manager plus the SwiftUI environment (color scheme, Increase Contrast, Reduce Transparency, Dynamic Type) into a `ResolvedAppTheme` and injects it as `\.appTheme`, so views read the theme from the environment and never from `NSApp`.
- `AppThemeColor` and the `.app*` ShapeStyles, `themedFont`, and the control and surface modifiers read `\.appTheme` to draw.
- Pure helpers (`AppContrast`, `AppAccentText`, `AppStatusPalette`, `AppFontCatalog`, `AppRightColumnClaimant`, `AppChromeLayout`) are free of UI state so tests can pin them.

DESIGN.md and `docs/design-system/` own the rules, and this directory owns the values.

## Interface
- Appearance state and persistence
  - `AppearanceManager` (shared; published appearance, text size, light and dark `ThemeModeConfig`, fonts, motion, pointer cursors, dock icon, diff markers, status bar mode, saved themes). It exposes `applyPreset`, `resetToDefaults`, `reloadFromProfile`, `makeTextBigger`, `makeTextSmaller`, `resetTextSize`, `setUIFont`, `setCodeFont`, the font descriptor and font builders, `activeConfig`, `activeAccentColor`, `activeBackgroundColor`, `activeForegroundColor`, `shouldReduceMotion`, and `updateDockIcon` [AppearanceSettings.swift:AppearanceManager].
  - `AppearanceArchive`, `AppearanceFileStore` (`load`, `save`, `removeLegacyKeys`) [AppearanceStore.swift].
  - Preference enums `AppAppearance`, `AppTextSize`, `AppLanguage`, `StatusBarViewMode`, `DiffMarkerPreference`, `ReduceMotionPreference`, `AppDockIcon` [AppAppearance.swift, AppTextSize.swift, AppLanguage.swift, AppearanceTypes.swift].
  - `AppDockIconRenderer.applyDockIcon` [AppDockIconRenderer.swift].
- Theme model
  - `ThemeModeConfig` (with `fromClipboardJSON`, `defaultLight`, `defaultDark`, `codexLight`, `codexDark`), `ThemePreset.presets`, `AccentOption`, and the `Color(hex:)`, `toHex`, `isLight`, `contrastForeground`, and `Font.Weight.fromName` helpers [AppearanceTypes.swift].
  - `SavedTheme`, `ThemeLibraryError`, `ThemePreset.additionalPresets`, `ThemeModeConfig.matchesPalette`, `textContrastRatio`, and the `AppearanceManager` theme-library methods (`availableThemes`, `currentThemeID`, `saveTheme`, `renameTheme`, `deleteTheme`, `importTheme`, `exportTheme`) [ThemeLibrary.swift].
  - `ResolvedAppTheme` (`resolve`, `fallback`, `ui`, `code`, `ui(fitting:)`, `surfaceOpacity`, `composerBackground`), `AppThemeInjector`, `View.appThemed()`, `EnvironmentValues.appTheme` [ResolvedAppTheme.swift].
  - `TurboSparkTheme`, the mode-explicit chrome color accessors for non-view contexts [TurboSparkTheme.swift].
- Colors, contrast, status
  - `AppThemeColor` and the `.appPage`, `.appSurface`, `.appElevated`, `.appText`, `.appSecondary`, `.appBorder`, `.appAccent`, `.appAccentText`, and the status and tint styles; `ResolvedAppTheme` surface, border, selection, and secondary-text colors [AppThemeColor.swift].
  - `AppStatusPalette`, `AppAccentText`, `ResolvedAppTheme.success`, `warning`, `danger`, the tints, `accentText`, `focusEdge`, and `View.appLinkTint()` [AppStatusColors.swift].
  - `AppContrast` (`ratio`, `luminance`, `blend`, `hex(of:)`) [AppContrast.swift].
- Typography
  - `AppFontStep`, `AppFontDescriptor`, `AppFontCatalog` [AppFontCatalog.swift]; `AppFontRegistrar` [AppFontRegistrar.swift]; `View.themedFont`, `themedFont(fitting:)`, and `themedCode` [ThemedFont.swift].
- Controls and surfaces
  - `AppMetrics`, `AppActionRole` with `appAction`, `appMenuAction`, `appEditorSurface`, `AppSearchField`, `AppIconButton`, `AppAdaptiveHeader`, `AppSectionHeader`, `AppEmptyState`, `AppChip` [AppControls.swift].
  - `appCardSurface`, `appComposerSurface`, `appFocusGlow`, `appPrimaryGlow` [AppSurfaces.swift].
  - `AppPillPicker` and `AppPillItem` [AppPillPicker.swift]; `AppProgressBar` [AppProgressBar.swift]; `AppLevelWaveform` and `AppFillWaveform` [AppWaveform.swift].
  - `appAttention` and `AppAttentionNote` [AppAttention.swift]; `appPointerCursor` [PointerCursorModifier.swift]; `AppearanceModeOptions`, `TextSizeOptions`, `LanguageOptions` [AppPreferenceOptions.swift].
  - Settings previews `ThemeCodePreviewView` and `ThemeTypographyPreviewView`.
- Motion [TurboSparkMotion.swift]
  - `TSMotion` (named animations, `animate`), `TSMotionContext`, `tsHoverLift`, `tsEntrance`, `tsPulse`, `tsAnimation`, `TSPressScaleStyle` (`.tsPress`), `TSIdlingSparkView`.
- Layout and chrome
  - `AppRightColumnClaimant` (`resolve`, `isVisible`, `isArtifact`, `isPreviewPane`) and the `AppChromeLayout` extension (`rightColumnWidth`, `minimumWindowWidth`, `browserPanelWidth`, `artifactPanelWidth`) [AppRightColumnClaimant.swift].
  - `AppChromeLayout` (widths, heights, `leftChromeWidth`), `AppSidebarKind`, `AppSidebarControlPresentation` [AppChromePresentation.swift].
- Strings and resources
  - `AppStrings.localized` and `bundle(for:)` [AppStrings.swift]; `Bundle.module` shim over `AppResources.bundle` [ResourceBundle.swift].
  - `LanguageDetector.currentKeyboardLayoutName` [LanguageDetector.swift].

No child directory has an EGM.md.

## Invariants
- Dark-mode resolution
  - Take `isDark` from `@Environment(\.colorScheme)` through `AppAppearance.isDark(systemColorScheme:)`, not from `NSApp.effectiveAppearance`. The application property is not moved by `.preferredColorScheme`. The static `TurboSparkTheme` accessors cover forced Light and Dark but still read `NSApp` for `.system`, so views should prefer `\.appTheme` [AppAppearance.swift:isDark, TurboSparkTheme.swift].
  - Every scene root needs its own `.appThemed()`. The Settings scene is a separate window and gets nothing from the main window [ResolvedAppTheme.swift:AppThemeInjector].
  - `ResolvedAppTheme.fallback` and `AppThemeKey.defaultValue` must not read `AppearanceManager`. They are nonisolated and use only the `Codable` defaults. `composerBackgroundColor(isDark:background:)` is `nonisolated` for the same reason [ResolvedAppTheme.swift, TurboSparkTheme.swift].
  - Increase Contrast raises the resolved `contrast` to at least 95. Reduce Transparency forces structural surfaces opaque (`surfaceOpacity`, and the `reduceTransparency` parameter on sidebar and rail colors).
- Persistence [AppearanceStore.swift, AppearanceSettings.swift]
  - Appearance lives in `appearance.json` under the store root, not in UserDefaults. `AppearanceArchive.init(from:)` uses `decodeIfPresent` with defaults for every field, so older files load. Saved themes that fail `validated()` are silently dropped on decode.
  - Migration from the eleven legacy UserDefaults keys is one-way. The legacy keys are removed only after the first successful `AppearanceFileStore.save` (`needsLegacyCleanup`), so a crash mid-migration loses nothing.
  - Every published property's `didSet` calls `persist()`. `reloadFromProfile` sets `isReloadingProfile` so partial field sets are never written back to the vault.
  - `persist()` writes `currentThemeID` (nil unless the palette still matches the selected preset), not the raw `selectedThemeID`.
  - Missing `selectedThemeID` in an old file is inferred by `matchesPalette` against the built-in presets.
  - Fonts are global. Setters write both light and dark configs, and the font accessors read `lightConfig` only. Keep that invariant if you add a writer.
  - `AppFontCatalog.factoryUISize` (14) is deliberately not `defaultUISize` (16). The latter is the layout reference that `textScale` and `ui(fitting:)` divide by.
- Themes [ThemeLibrary.swift]
  - `SavedTheme.validated()` requires version 1, a trimmed name of 1 to 80 characters, `#RRGGBB` for accent, background, and foreground, and contrast in 0...100. `importTheme` always replaces the incoming id with a fresh UUID so an import cannot overwrite a built-in or another saved theme.
  - `ThemePreset.presets` ids must stay distinct from `defaultLight` and `defaultDark` except `sparkblue`. The code documents a test for this, and Codex values are named literals, not aliases of the default.
  - `applyPreset` preserves the current font choices. A single-mode apply clears `selectedThemeID`.
- Type scale [AppFontCatalog.swift, ThemedFont.swift]
  - Text size is by `AppFontStep` role only. A numeric size is allowed only as a layout-derived value through `themedFont(fitting:)`. A repeated constant belongs as a new step.
  - `AppFontDescriptor.customFamilyName` is nil for `System default`, `SF Pro`, and `SF Mono`. Those go through `Font.system(design:)` and must never reach `Font.custom`.
  - `resolveFamily` falls back to the system face by name when a persisted family is not installed, so the picker shows what is drawn.
  - `installedFamilies()` is memoized for the process (it runs inside a per-token body), so a font installed mid-run appears after relaunch.
- Contrast and status [AppStatusColors.swift, AppContrast.swift]
  - Status hexes are fixed per mode and per Increase Contrast, and are not derived from the accent. They must clear AA on every preset surface. `AppStatusColorTests` pins this, and DESIGN.md plus `derive_tokens.py` must change with any status hex. Status is never color alone, so pair it with a glyph or a word.
  - Surfaces for contrast must be read from the drawn color with `AppContrast.hex(of:)`, never rebuilt with `blend`, because `NSColor.blended` is not an sRGB mix.
  - `accentText` darkens the light-mode accent toward black in 2% steps until 4.5:1 on the composer fill. Spark Blue light has a named companion (`#1A64CC`). The focus edge uses the accent only if it reaches 3:1 on every surface, else the foreground.
- Motion [TurboSparkMotion.swift, AppAttention.swift, AppProgressBar.swift]
  - Every animation resolves Reduce Motion through `AppearanceManager.shouldReduceMotion` (the app setting first, the system setting for `.system`). Direct `withAnimation` calls bypass it, so use `TSMotion.animate` and `tsAnimation`.
  - Looping animations use `phaseAnimator` or `keyframeAnimator` owned by the view, so they stop when the view leaves.
  - The attention edge is the state, and the ping is only a pointer. Reduce Motion replaces rings with one held ring (2.7 s).
  - A paused `AppProgressBar` (`animated: false`) holds still. The bar flips to left-to-right geometry and mirrors as a whole.
- Layout [AppRightColumnClaimant.swift, AppChromePresentation.swift]
  - The right column is one slot. `resolve` precedence is artifact, HTML preview, file preview, browser, inspector, project summary. The setters keep at most one of the first three set, and the ladder is only a fallback.
  - `minimumWindowWidth` always counts the rail. The panel counts only when it is actually on screen, not merely preferred. The widths for artifact (420) and browser (460) differ from the inspector (320) on purpose, so a width test can fail.
- Controls [AppControls.swift, AppPillPicker.swift]
  - Controls wrap native `Button`, `Menu`, `TextField`, and `Toggle`, so keyboard, focus, and VoiceOver semantics stay native. Icon-only controls take a `Text` label that becomes the accessible name and tooltip. Search prompts are passed as `Text` so the localization source scan sees the literal.
  - `AppAdaptiveHeader` has three `ViewThatFits` layouts and the last always accepts, so nothing clips.
  - `AppPillPicker` must not fall back to a menu inside `ViewThatFits`. That caused continuous window re-measurement. Arrow keys clamp at the ends and mirror under RTL.
  - `AppChip` shows an on/off indicator as a differently shaped glyph, not only a different hue.
- Localization [AppStrings.swift, AppLanguage.swift, AppPreferenceOptions.swift]
  - UI strings outside `Text` use `AppStrings.localized`, not `String(localized:)`, so they follow the in-app language picker and not the process language. Language names are intentionally untranslated. Only the System entry is a UI string.
  - `AppLanguage.system` resolves through `Locale.preferredLanguages` and the catalog, and falls back to English. RTL follows the resolved catalog language. Compiled `.lproj` names are lowercased, so matching is case-insensitive.
  - `writeProcessLanguages` mirrors the picker into `AppleLanguages` and takes effect at the next launch.
- Resources [ResourceBundle.swift, AppFontRegistrar.swift]
  - This target declares no resources, so `Bundle.module` is a hand-written shim over `AppResources.bundle`.
  - Bundled fonts are registered with Core Text at `.process` scope, not through an Info.plist key. Font URLs fall back from the `Fonts` subdirectory to the bundle root, and the code uses an emptiness check rather than `??`. `alreadyRegistered` counts as success.

## Side Effects
- Disk [AppearanceStore.swift]
  - `AppearanceFileStore.save` writes `appearance.json` through `AppJSONStore` on every appearance change. `load` reads it, and `AppJSONStore` quarantines corrupt files.
- UserDefaults
  - `AppStrings.currentLanguage` reads `TurboSpark.language` [AppStrings.swift].
  - `AppLanguage.writeProcessLanguages` writes or removes `AppleLanguages` in the app domain [AppLanguage.swift].
  - Legacy appearance keys are read for migration and removed after the first successful save [AppearanceStore.swift].
- Process and AppKit
  - `AppDockIconRenderer.applyDockIcon` sets `NSApplication.shared.applicationIconImage`, loading `AppIcon.png` from `Bundle.module` for the default icon. `AppearanceManager.init` and `reloadFromProfile` call it [AppDockIconRenderer.swift, AppearanceSettings.swift].
  - `AppPointerCursorModifier` pushes and pops `NSCursor.pointingHand`, tracking its own push so the stack stays balanced [PointerCursorModifier.swift].
  - `AppFontRegistrar` registers bundled `.ttf` files with Core Text and writes failures to standard error. `AppFontCatalog.installedFamilies` queries `NSFontManager` [AppFontRegistrar.swift, AppFontCatalog.swift].
  - `LanguageDetector` calls Carbon `TISCopyCurrentKeyboardInputSource` [LanguageDetector.swift].
  - `TSMotion.animate` reads `NSWorkspace.shared.accessibilityDisplayShouldReduceMotion` [TurboSparkMotion.swift].
- Network, process spawn, Keychain, NotificationCenter, FFI
  - None.
- Tasks and timers
  - `AppAttentionModifier` sleeps for the hold duration under Reduce Motion [AppAttention.swift].
  - `AppFillWaveform` runs a `TimelineView` at up to 30 fps only while `rippling` and motion is allowed [AppWaveform.swift].
- Shared state
  - `AppearanceManager.shared` and the `WelcomeCharacterAssetCache.shared` lookup in `TSIdlingSparkView` (defined elsewhere) are read by views [TurboSparkMotion.swift].

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
- App UI changes also run `make design-check` (and `make design-check-full` for the contrast and ratchet tests), per the repository root instructions.
