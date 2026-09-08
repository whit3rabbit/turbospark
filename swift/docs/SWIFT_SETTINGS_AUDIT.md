# Swift app settings audit

What the TurboSparkApp settings surface promises against what it delivers:
every persisted key traced to its consumers, every pane control checked for
a real effect. Most findings below are fixed; this page keeps the ones
still open, the root-cause explanations worth not re-deriving, and the
method for re-running the audit. Read it before adding a setting, before
trusting a control that looks wired, and before proposing another pass at
the font system.

The method: grep for the ACCESSOR, never for the setting. Section 5 spells
it out.

Two stores hold everything. `State/MacAppSettings.swift` writes
`settings.json` and `Theme/AppearanceSettings.swift` writes
`appearance.json`, both under `AppStorageRoot`
(`swift/docs/SWIFT_STORAGE.md`). Thirteen tabs edit them.

Two other surfaces moved off their old stores (2026-09-05/06) and neither
is tested the same way. The server API key lives in the login Keychain
(`ServerKeychain.swift`, service `TurboSpark.server`,
`kSecAttrAccessibleAfterFirstUnlock`) rather than in `settings.json`; an
empty string deletes the item, and a Keychain error degrades to "no key
restored" rather than failing the load. `ServerKeychain` itself is
UNTESTED, because a round-trip test would write into the host user's real
keychain -- stated here so its absence reads as a decision. And
`AppearanceManager` moved from `UserDefaults` (11 keys and 2 JSON blobs) to
`appearance.json`, with a one-way migration that reads the legacy keys
only when the file is absent and removes them only after the first
successful save.

## 1. Font propagation

Fixed 2026-09-06. The root cause: 81 files used raw `.font(...)` calls that
never read `\.appTheme`, so the font/theme injector's container-level
`.font(theme.uiFont)` was overridden by any explicit child `.font(...)`.
`Installation/`, `Diagnostics/`, `Files/` and `Server/` had zero themed
calls; the Skills, Permissions, MCP, Hooks and Agents panes were entirely
hardcoded. Text Size was ALSO applied by three mechanisms that disagreed
(themed sites moved by `AppTextSize.scale`, semantic sites moved about one
point through `.dynamicTypeSize`, fixed-size sites did not move at all).

`Theme/ThemedFont.swift` is the fix: a modifier family
(`.themedFont(.small)`, `.themedCode(.small)`) that reads `\.appTheme`
itself, so a call site needs no environment declaration. 972 hardcoded
sites converted; 14 remain by design (the two per-mode theme preview cards
and the font-family picker's own preview rows, which must show a SPECIFIC
mode or a candidate the user has not chosen yet, not the ambient theme).
The scene-level `.dynamicTypeSize` modifier is gone; every themed site
scales through `AppFontDescriptor`'s own size now, the more precise of the
two mechanisms.

`FontPropagationTests.swift` is the regression guard: it walks every file
under `Sources/TurboSparkApp`, and any file whose text contains `.font(`
without also containing `themedFont`, `themedCode` or `theme.` fails the
build. The exempt list is explicit in the test, by relative path and by
reason, so a change to one of those files is a visible diff rather than a
silent widening of the exemption.

**Two zero-caller theme accessors this audit found were a second copy of
the SAME `NSApp.effectiveAppearance` bug the resolution fix below closes.**
`TurboSparkTheme.metadataForeground(contrast:)` and
`.borderStrokeOpacity(contrast:)` were deleted rather than wired up, since
the correct replacements (`ResolvedAppTheme`'s instance members of the same
names) already had every real call site.

### The `NSApp.effectiveAppearance` resolution bug

`NSApp.effectiveAppearance` is APPLICATION-level, while
`.preferredColorScheme` sets the WINDOW's, so all seven `TurboSparkTheme`
accessors branching on the former drew light chrome out of `darkConfig`
when the app was forced to Light on a dark system -- dark accent, dark
background, dark contrast. `ThemeCodePreviewView` had the identical bug
with `isDark` ALREADY IN SCOPE, so both its Light and Dark cards showed
whichever mode happened to be active. The fix is `ResolvedAppTheme`, an
`Equatable` environment value injected once by `RootView`; being in the
environment is what makes it reactive, since a static computed var
re-reads its inputs but tells SwiftUI nothing changed. `effectiveIsDark`
(the getter behind the four font accessors, carrying the same bug) is
deleted; the getters read `lightConfig`, documented as always equal.

### Two packaging traps, and both fail silently

`Package.swift` declares `.process("Resources")`, and that rule FLATTENS
subdirectories: bundled fonts land at the resource bundle's root, not
under `Fonts/`, exactly as `Resources/Logos/*.svg` do (measured in both the
`.build` bundle and the shipped `.app`). That also rules out an
`Info.plist` `ATSApplicationFontsPath`, which only looks directly under
`Contents/Resources` and never inside the nested `.bundle` -- and which
`swift run` has no plist for anyway (`swift/CLAUDE.md` Gotcha 12).
Register through `CTFontManagerRegisterFontsForURL` at `.process` scope
instead.

`??` is the wrong operator for the subdirectory fallback:
`Bundle.urls(forResourcesWithExtension:subdirectory:)` returns an EMPTY
ARRAY rather than nil for a missing subdirectory, so a nil-coalescing
chain never falls through, zero faces register, and nothing logs an
error.

**The test for that skipped twice before it could fail.** Its `XCTSkipIf`
was keyed first on `registerBundledFonts()`'s return value and then on
`bundledFontURLs` -- both computed by the code under test, so a broken
lookup and an empty directory were the same zero and the case went GREEN
(skipped) under mutation both times. It keys on the checked-in files via
`#filePath` now, and the mutation reddens. General rule: a skip condition
derived from the thing under test cannot tell "nothing to do" from "it is
broken."

## 2. Persisted settings, field by field

Every `MacAppSettings` field was traced from `AppModel+Persistence.swift`'s
`loadSettings()` to a consumer outside the store and outside the pane that
edits it. The 47 fields not listed below all reach one: the sampling
fields reach `AppModel+Generation.swift`'s `GenerateOptions`, the runtime
fields reach `buildOpenOptions` in `AppModel+Models.swift`, the steering
fields funnel through `resolvedSteeringPreset`, and the server, compaction,
ghost, plugin and guardrail fields each have a named reader. Three did
not, all fixed 2026-09-06:

| Field | Finding | Disposition |
|---|---|---|
| `prefillEnabled` | No consumer; `OpenOptions` in the binding has no prefill field. | Deleted. |
| `modelsDirectory` | Edited by the Models pane's "Change..." button and read only by that pane; catalog installs have no configurable destination. | Deleted. The pane shows the store path read-only. |
| `commandAdvisoryVeto` | Consumed by `CommandGate` and reachable from nowhere but a hand edit of `settings.json`. | Toggle added, off by default (see `swift/docs/SWIFT_TOOLS.md`). |

`expertCacheSlots` and the six raw steering knobs are editable only in the
Inspector, not in any Settings tab. `SubagentRunner` and
`AppChatCompaction` each build their own `GenerateOptions` with a fixed
temperature and budget; compaction does that on purpose.

Per-chat sampling (added 2026-09-07): `samplingPresets` in `settings.json`
(app-wide presets, CRUD through `AppModel+Sampling.swift`) and
`samplingOverride` in each chat row of `chats_archive.json`
(`AppSamplingSettings?`, nil meaning "use the app-wide fields").
`executeGenerationTurn` reads `effectiveSamplingSettings(chatID:)`. Both
clamp on decode (`AppSamplingSettings.clamped()`); the per-chat path
reaches the `UInt32` conversion without the app-wide path's load-guard
clamp, so an out-of-range value in a hand-edited archive would trap the
process rather than error.

## 3. Appearance settings, field by field

| Field | Finding | Disposition |
|---|---|---|
| `dockIcon` | Persisted and rendered with no picker anywhere. | Picker added. |
| `reduceMotion` | Honoured by `RootView`'s pane transitions only; two other surfaces read the system flag directly. | Both read the app preference now. |
| per-mode font rows | `setUIFont`/`setCodeFont` write both configs, so a per-mode font was never representable. | Rows removed. One set of font controls. |
| theme Import | A non-theme clipboard silently applied the TurboSpark preset. | Reports "Not a theme," changes nothing. |
| `installedFamilies()` | Enumerated `NSFontManager` on every injector body pass, once per streamed token. | Memoized. |
| reset | No way back to the factory theme short of hand-editing `appearance.json`; `resetTextSize()` covered sizes only. | `resetToDefaults()` (2026-09-07) restores every published field from a zero-arg `AppearanceArchive()` behind a confirmation dialog at the bottom of the pane; `testResetToDefaultsRestoresEveryField` moves all eleven fields off their defaults first so a partial reset cannot pass. |
| `translucentSidebar`, `usePointerCursors`, `statusBarViewMode`, `diffMarkers`, `contrast`, the three hex colors | Each has a control and a reader. | No action. |
| `ResolvedAppTheme.borderStrokeOpacity`, `.background`, `.textSize` | Zero callers. | Left in place; the font conversion is the likely consumer. |

### The default preset is Spark Blue, and a preset must not alias the default

`ThemePreset.presets`' first entry used to be

    ThemePreset(id: "codex", light: .defaultLight, dark: .defaultDark)

so "Codex" was not a preset at all, it was a name for whatever the default
happened to be. Changing the default (2026-09-07, to Spark Blue -- the cyan
taken from `WelcomeCharacter.png`, on a `#12151B` dark ground rather than a
neutral `#181818`, so a cyan accent has something to sit on) would have
silently rewritten Codex into the new default and left the app with two
identical entries.

`ThemeModeConfig.codexLight` / `.codexDark` are Codex's own values now, and
`.defaultLight` / `.defaultDark` carry Spark Blue. **A preset names a set of
colours; it never points at `default*`.** Only a fresh install moves: the
appearance store decodes a persisted config first and falls back to the
defaults, so an existing user keeps whatever they had.

## 4. Pane findings

All fixed 2026-09-06 unless noted. Compact facts only; grep the pane name
if the mechanism matters.

- **Engine / Inspector**: seven `TextField`s passed a title with no
  `.labelsHidden()`, which on macOS is a visible label rather than a
  placeholder. All hidden now.
- **Server Advanced**: the memory guard picker hardcoded four of five
  tiers (no Custom) and rendered blank after choosing Custom elsewhere; it
  iterates the enum now. The reasoning picker bypassed `setReasoning` and
  offered levels the checkpoint refuses; it shares the Engine pane's
  binding now (`swift/docs/SWIFT_SESSION_CAPABILITIES.md`). The port field
  turned "8O80" and "70000" into 0 captioned "automatic";
  `ServerPortInput.parse` refuses both with a reason.
- **Keyboard Shortcuts**: the rail advertised Cmd-5 for Server with
  nothing bound (fixed), and omitted New Temporary Chat and Add Files. The
  rows are derived from `KeyboardShortcutCatalog` against a test now.
  Alternate chords (unsloth compatibility, 2026-09-07) are documented in
  `swift/docs/KEYBOARD_SHORTCUTS.md`.
- **Models and Storage**: a "Configure Custom Path" toggle moved only its
  own caption; a `@State` path input was written once and never read.
  Both removed.
- **Skills**: the detail pane advertised "Allowed Tools and Permissions"
  enforcement that does not exist (`state#48`); removed. A disabled skill
  had no re-enable path outside the composer menu; an Enabled switch sits
  in the detail header now.
- **Agents**: a "Model Override" row displayed a field nothing reads; gone.
- **Hooks**: an empty state read `No hooks found matching ''`; now says
  "No hooks configured."
- **MCP**: a header/tab title mismatch, three non-interactive count pills
  tinted one as selected, and an icon ternary with identical arms; all
  fixed.
- **Plugins**: option fields drew their title twice; hidden on the field.
- **Localization**: the language picker's root cause and fix are the
  subject of `swift/docs/SWIFT_LOCALIZATION.md`, not restated here.
- **Settings search**: cross-referenced every pane's real control labels
  against its tab's `keywords`; the newest control at audit time
  (`commandAdvisoryVeto`) had no keyword entry, which is the general
  failure mode -- a keyword list drifts the moment a control is added.
  `ServerAndAppearanceStoreTests.testSettingsKeywordsCoverRecentlyAddedControls`
  pins it. Search still matches only the tab title and this hand-typed
  list, never rendered pane content; real content-indexing search is a
  separate, larger feature.
- **Subagent sampling**: every subagent turn ran at a hardcoded
  `temperature: 0.2` regardless of the Engine pane's setting.
  `AppModel.samplingOptions()` is now the one place that reads sampling
  settings from the user's live values, shared by both `AppModel`-context
  call sites and, through `AppToolRegistry.subagentSamplingOptionsProvider`,
  the two `AppToolRegistry` call sites. `maxNewTokens` stays a
  subagent-owned `2048` on purpose: a tool-use turn budget is an
  architectural choice, not a sampling preference.
- **`LanguageDetector`**: `currentKeyboardLanguage()`, `detectTextLanguage(_:)`
  and `isRTL(languageCode:)` had zero production callers and are deleted.
  `currentKeyboardLayoutName()` stays; `GeneralSettingsPaneView` reads it
  for an informational row.
- **`scanModels` in a view body**: `ModelStorageManager.scanModels` walks
  the target directory with real I/O and ran inline in `body`, re-running
  on every SwiftUI re-evaluation. Both call sites cache the count in
  `@State`, recomputed by a `.task(id:)` keyed on what actually changes it.
- **HF endpoint**: two editors (Server Advanced, the auth-token card) each
  wrote the endpoint their own way, so editing through the first left the
  catalog stale until the next launch. `HfEndpointResolution.effectiveEndpoint(from:)`
  is the one function both call now.
- **Server key generate/copy**: Generate fills the API key field from
  `ServerAPIKeyGenerator.generate()`; Copy writes the trimmed effective key
  to the clipboard.
- **Per-chat sampling controls** (2026-09-07): the Inspector's Generation
  Sampling section gained an "Edits apply to" picker over "App defaults"
  and "This chat," explicit rather than implicit, because the Inspector is
  always open (unlike a per-thread sheet) and a silent chat-local edit
  would read as an app-wide one. A chat override is a COMPLETE snapshot,
  not per-key optionals. Reasoning stays app-wide on purpose (it already
  has per-model memory).

Cleared after checking, so nobody re-derives them: the Profiles caption
about isolation is accurate, `/v1/embeddings` is a real listed route, and
every control in the Safety and Permissions panes reaches a consumer.

## 5. How to re-run this audit

For a persisted field, grep its `AppModel` property name and discard the
hits in `AppModel+Persistence.swift`, `MacAppSettings.swift` and the pane
that edits it. Anything left is a consumer, and nothing left is a dead
setting. For a control, find the bound property and apply the same rule.
For a button, read its action, and for a caption, check the claim against
the code it describes.

The font counts in section 1 come from grepping `theme.ui(`, `theme.code(`
and the `.font(` patterns over `Sources/TurboSparkApp`.
