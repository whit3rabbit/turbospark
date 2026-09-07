# Swift app settings audit

Audited 2026-09-06. This page is the HOME for what the TurboSparkApp settings
surface promises against what it delivers: every persisted key traced to its
consumers, every pane control checked for a real effect, and the font
complaint diagnosed to its cause. Read it before adding a setting, before
trusting a control that looks wired, and before proposing the font rewrite.
The method is `swift/CLAUDE.md` Gotcha 36's rule applied to the whole surface:
grep for the ACCESSOR, never for the setting.

Two stores hold everything. `State/MacAppSettings.swift` writes
`settings.json` and `Theme/AppearanceSettings.swift` writes `appearance.json`,
both under `AppStorageRoot`. Thirteen tabs edit them.

## 1. Why font changes do not apply

The pipeline is correct and this was checked end to end. `ResolvedAppTheme`
takes every font input, is `Equatable` over all of them, and `AppThemeInjector`
observes `AppearanceManager`. Both scenes inject it (`RootView.swift`,
`TurboSparkApp.swift`'s `Settings` scene). The transcript reads it
(`ChatMessageMarkdownView`). All twelve bundled TTFs register and all four
weights resolve, measured at runtime.

The defect is call-site coverage. Counted over `Sources/TurboSparkApp`:

| Pattern | Count |
|---|---|
| `theme.ui(` and `theme.code(` | 232 |
| `.font(.caption)`, `.body`, `.headline` and the other semantic styles | 737 |
| `.font(.system(size:))` at a fixed point size | 235 |
| `NSFont.` | 12 |

81 files contain `.font(` and never read `\.appTheme`, which is 809 of 1,230
font calls. The injector's container-level `.font(theme.uiFont)` is overridden
by any explicit `.font(...)` on a child. `Installation/`, `Diagnostics/`,
`Files/` and `Server/` have zero themed calls. The Skills, Permissions, MCP,
Hooks and Agents panes are entirely hardcoded, so the tabs beside the font
picker do not move when it changes. The rail, top bar, status bar and chat
sidebar are fully themed, which is why some of the window does update and
the rest does not.

Text Size is applied by three mechanisms that disagree:

| Site kind | Family, weight, size stepper | Text Size |
|---|---|---|
| 232 themed sites | move | move by `AppTextSize.scale` (1.0, 1.15, 1.30) |
| 737 semantic sites | no effect | move about one point through `.dynamicTypeSize` |
| 235 fixed-size sites | no effect | no effect |

`AppTextSize.standard` maps to `.xLarge` dynamic type, so the semantic half
already renders one notch above the themed half at the Default setting.

Smaller confirmed items in the same area:

- `MenuBarExtra` has no `.appThemed()`, and `ToastOverlayView` and
  `ProjectMcpApprovalSheet` consume no theme.
- `CodeBlockContainer`'s language tag and Copy button use `.caption2` while
  the block body is themed.
- The export renderers (`ResponseMarkdownRenderer`,
  `InstructionTranscriptDocumentController`) use `NSFont.systemFont`.
- `ChatMessageMarkdownView` forces a full `.id` rebuild on any font change,
  which drops selection and scroll position.
- Only one test, `testMarkdownRenderingWithCustomFont`, asserts that a
  rendered surface follows the font setting.

### Fix, done 2026-09-06

`Theme/ThemedFont.swift` adds the modifier family: `.themedFont(.small)`,
`.themedFont(points: 11, weight: .medium)`, `.themedCode(.small)`, each
reading `\.appTheme` itself so a call site needs no environment declaration.
The 972 hardcoded sites converted by a two-pass script against the fixed map
this section originally proposed (`.caption2` to `.tiny`, `.caption` and
`.subheadline` to `.small`, `.body` and `.callout` to `.base`, `.headline` to
`.base` at semibold, `.title3`/`.title2`/`.title` unchanged, `.largeTitle` to
`.hero`, `.system(size: N, weight: W)` to `points: N`, `.monospaced()` to the
code variant), plus by hand for the dozen sites the script's patterns did not
cover (a top-level ternary switching between two whole font expressions, one
badge view whose `Font`-typed computed property became an `AppFontStep`
one). 245 `theme.ui(`/`theme.code(` sites plus 974 `themedFont(`/`themedCode(`
sites now read the theme. 14 do not, and are the audit's own deliberate
exceptions -- the two per-mode theme preview cards
(`ThemeCodePreviewView.swift`, `ThemeTypographyPreviewView.swift`, which
resolve a SPECIFIC mode's font rather than the ambient one, by design) and
the font-family picker's own preview rows (`AppearancePreferencesCardView.swift`,
which preview a candidate the user has not chosen yet). The 12 `NSFont.`
sites in the export renderers are unchanged: SwiftUI's `Font` has no bearing
on an `NSAttributedString` render target.

The scene-level `.dynamicTypeSize(appearanceManager.textSize.dynamicTypeSize)`
is gone from both scenes in `TurboSparkApp.swift`, and
`AppTextSize.dynamicTypeSize` (and its "one notch above baseline" comment)
is deleted with it: every themed site now scales through
`AppFontDescriptor`'s own size, which was always the more precise of the two
mechanisms this section's table compared. This is item 2 as well as the rest
of item 1.

`FontPropagationTests.swift` is the regression guard: it walks every file
under `Sources/TurboSparkApp`, and any file whose text contains `.font(`
without also containing `themedFont`, `themedCode` or `theme.` fails the
build (mutation-checked: an injected `.font(.caption)` in an otherwise
font-free file reddens it). The exempt list above is explicit in the test,
by relative path and by reason, so a change to one of those four files is a
visible diff rather than a silent widening of the exemption.

Two zero-caller theme accessors this section flagged turned out to be the
exact `NSApp.effectiveAppearance` bug Gotcha 36 (`swift/CLAUDE.md`) already
fixed once, left behind as a second copy: `TurboSparkTheme.metadataForeground(contrast:)`
and `.borderStrokeOpacity(contrast:)` are deleted rather than wired up, since
the correct replacements (`ResolvedAppTheme`'s instance members of the same
names) already have every real call site. `ResolvedAppTheme.background` (a
stored field, never read anywhere including the test suite) is deleted too.
`ResolvedAppTheme.textSize` and the instance `.borderStrokeOpacity` are kept:
both are exercised by `AppearanceSettingsTests`, which is a real consumer
even though no view has wired them into a control yet. This is item 8.

## 2. Persisted settings, field by field

Every `MacAppSettings` field was traced from `AppModel+Persistence.swift`'s
`loadSettings()` to a consumer outside the store and outside the pane that
edits it. The 47 fields not listed below all reach one: the sampling fields
reach `AppModel+Generation.swift`'s `GenerateOptions`, the runtime fields
reach `buildOpenOptions` in `AppModel+Models.swift`, the steering fields
funnel through `resolvedSteeringPreset`, and the server, compaction, ghost,
plugin and guardrail fields each have a named reader. Three did not.

| Field | Finding | Disposition |
|---|---|---|
| `prefillEnabled` | Loaded, saved, declared on `AppRuntimeOptions`, read by nothing. `OpenOptions` in the binding has no prefill field, so it could never reach the engine. | Deleted 2026-09-06. |
| `modelsDirectory` | Edited by the Models pane's "Change..." button and read only by that pane. Catalog installs go to the engine's store and the binding exposes no destination. | Deleted 2026-09-06. The pane shows the store path read-only. |
| `commandAdvisoryVeto` | Consumed by `CommandGate` and reachable from nowhere but a hand edit of `settings.json`. | Toggle added 2026-09-06 in Files & Permissions, with the measured reason it is off by default. |

Two things worth knowing about the fields that are wired. `expertCacheSlots`
and the six raw steering knobs are editable only in the Inspector, not in any
Settings tab. And `SubagentRunner` and `AppChatCompaction` each build their
own `GenerateOptions` with a fixed temperature and budget, so no sampling
setting reaches a subagent turn or a compaction summary. Compaction does that
on purpose. Subagents arguably should inherit the user's settings, which is
an open item below.

## 3. Appearance settings, field by field

| Field | Finding | Disposition |
|---|---|---|
| `dockIcon` | Persisted and rendered (all four variants draw) with no picker anywhere, so it was stuck at the default for the life of the feature. | Picker added 2026-09-06. |
| `reduceMotion` | Honoured by `RootView`'s pane transitions only. `ToastOverlayView` and the top bar's phase indicator read the system `accessibilityReduceMotion` directly. | Both read the app preference since 2026-09-06. |
| per-mode font rows | `ThemeConfigCardView` showed UI and Code font rows on both the Light and Dark cards, but `setUIFont` and `setCodeFont` write both configs, so a per-mode font was never representable. | Rows removed 2026-09-06. The Preferences card holds the one set of font controls. |
| theme Import | On a clipboard that was not a theme, Import silently applied the TurboSpark preset. | Reports "Not a theme" and changes nothing since 2026-09-06. `ThemeModeConfig.fromClipboardJSON` is the pure decoder and is tested. |
| `effectiveIsDark` | Read `NSApp.effectiveAppearance`, the exact bug Gotcha 36 removed from `TurboSparkTheme`, as the getter behind the four font accessors. Harmless only while the both-configs invariant held. | Deleted 2026-09-06. The getters read `lightConfig`, documented as always equal. |
| `installedFamilies()` | Enumerated `NSFontManager.shared.availableFontFamilies` on every injector body pass, once per streamed token. | Memoized 2026-09-06. A font installed while the app runs appears after relaunch. |
| `translucentSidebar`, `usePointerCursors`, `statusBarViewMode`, `diffMarkers`, `contrast`, the three hex colors | Each has a control and a reader. | No action. |
| `ResolvedAppTheme.borderStrokeOpacity`, `.background`, `.textSize`, `TurboSparkTheme.metadataForeground(contrast:)`, `TurboSparkTheme.borderStrokeOpacity(contrast:)` | Zero callers. | Left in place. The font conversion is the likely consumer. |

## 4. Pane findings

Done 2026-09-06:

- Engine: seven `TextField`s passed a title with no `.labelsHidden()`, which
  on macOS is a visible label rather than a placeholder (Gotcha 23). One was
  byte-identical to the "Uncapped" field the Inspector had already fixed. The
  Inspector's stop-sequences field had the same gap. All eight hide their
  labels now, and the section caption no longer claims every field mirrors
  the Inspector.
- Server Advanced: the memory guard picker hardcoded four of the five tiers
  and omitted Custom, so after choosing Custom in Models and Storage it
  rendered blank and any touch discarded the ceiling. It iterates the enum
  now. The reasoning picker bound `$model.reasoning` directly, skipping
  `setReasoning` and the per-model memory from state#96, and offered all five
  levels against Gotcha 9. It uses the Engine pane's binding and option
  source now. The port field turned "8O80" and "70000" into 0 and captioned
  it "automatic". `ServerPortInput.parse` refuses both with a reason, and is
  tested.
- Keyboard Shortcuts: the rail advertised Cmd-5 for Server with nothing
  bound. The binding exists now. The pane omitted New Temporary Chat
  (Shift-Cmd-N) and Add Files (Cmd-U) and misnamed Clear Chat History. The
  rows come from `KeyboardShortcutCatalog`, whose navigation section is
  derived from `AppNavigationSection`, and a test holds it against the rail.
  The pane says shortcuts are fixed, because no rebinding exists.
- Models and Storage: "Configure Custom Path" toggled only its own caption,
  and a `@State` path input was written once and never read. Both removed.
- Skills: the detail pane rendered "Allowed Tools and Permissions" with a
  green shield per entry, one scroll below a comment explaining that
  `allowed-tools` is enforced by nothing (state#48). Removed. The pane greyed
  a disabled skill with no way to re-enable it, since the only toggle was in
  the composer menu. An Enabled switch sits in the detail header now.
- Agents: the "Model Override" row displayed `agent.model`, which nothing
  reads, and is gone. The slash-command hint is shown only for an enabled
  agent, since the command refuses a disabled one (state#93).
- Hooks: a fresh install with no hooks read `No hooks found matching ''`.
  The empty state says "No hooks configured" and offers Add Hook.
- MCP: the header said "Plugins and MCPs" under a tab titled MCP Servers,
  three non-interactive count pills tinted one as selected, and an icon
  ternary had identical arms. All three fixed.
- Plugins: the option fields drew their title twice. Hidden on the field.

Done 2026-09-06 (font propagation pass, see section 1's "Fix, done" note for
the detail): the modifier family and scripted conversion (was rank 1), Text
Size unification (was rank 2), theming `MenuBarExtra` and the toast overlay
(was rank 7), and the dead theme accessors (was rank 8).

Done 2026-09-06 (language picker, was rank 1): reproduced, and the root
cause runs deeper than "zero `Text` calls pass `bundle: .module`."
`swift build`/`swift run` -- the only way this app is ever built, since it
has no Xcode project -- copy `Localizable.xcstrings` into the resource
bundle VERBATIM. There is no SwiftPM build phase for the String Catalog
format, only Xcode's own "Compile String Catalogs" step has one, so every
one of the 21 languages was dead JSON regardless of what any call site
passed as `bundle:`. Confirmed directly: a `swift build -v` on the
untouched tree shows the raw 600+ KiB file being copied, and no `.lproj`
directory exists anywhere in the build output at any `swift-tools-version`.

The fix runs Xcode's own compiler ahead of the SwiftPM build rather than
reimplementing it. `scripts/compile-strings.sh` calls `xcstringstool
compile` (the same private tool the Xcode build phase calls, found via
`xcrun --find`) against `Localization/Localizable.xcstrings` -- moved out of
`Sources/TurboSparkApp/Resources/`, since it is a build INPUT and not a
runtime resource -- and writes the per-language `.lproj` output directly
into `Resources/`, which `Package.swift`'s `.process("Resources")` rule
already bundles: SwiftPM DOES understand a plain `<language>.lproj` folder,
unlike the `.xcstrings` source format one level up. `make compile-strings`
is a new Makefile target. `swift-app`, `swift-app-build`, `swift-app-release`
and `scripts/make-app-bundle.sh` (the actual release path) all depend on it,
so the DMG a tag push produces is not exempt.
`Tests/TurboSparkAppTests/LocalizationTests.swift` is the regression guard:
it resolves a known key through the compiled French bundle and asserts every
`AppLanguage` case has a compiled `.lproj`, comparing through
`Bundle.preferredLocalizations` rather than a raw path lookup, because
`xcstringstool` lowercases its output folder names (`pt-br.lproj`,
`zh-hans.lproj`) while `AppLanguage`'s raw values keep the mixed-case BCP-47
spelling (`pt-BR`, `zh-Hans`) that the real locale-matching resolves
case-insensitively -- a naive path-equality test would redden on exactly the
region- and script-tagged languages while the app runs them correctly.

The second half survives from the original diagnosis: of 1,120 `Text(...)`
calls, 686 were a literal `LocalizedStringKey` (549 bare, 137 interpolated)
and none passed `bundle:`. All 686 do now (674 converted by script, 12 by
hand where a nested ternary's own quotes made the regex unsafe to trust).
The other 434 are `Text(String)` calls over dynamic content and are
untouched by design -- that overload takes no `bundle:` parameter at all.
One script mistake worth recording because it reproduces easily: a first
pass matched `Text(` as a SUBSTRING, corrupting `onInsertPromptText(...)`
and an enum case named `...noExtractableText(...)` into invalid calls
carrying a `bundle:` argument nothing declares. Caught by the next build
(it does not compile), reverted, and the real fix required requiring a
non-identifier character immediately before `Text(`.

Still open: the catalog covers 196 keys against several hundred distinct
`Text(...)` literals in the app, so most strings display in English in every
language until someone writes the missing translations. That is a content
gap, not a mechanism bug, and out of scope for this pass.

Done 2026-09-06 (settings search, was rank 1): the "keywords already drift"
half is fixed -- cross-referenced every pane's real `Section`/`Toggle`/`Text`
labels against its tab's `keywords` and added what a user would type for a
control that is really there. The concrete example: `permissions.keywords`
had no entry for "Command Classifier Veto," a toggle added the SAME DAY as
this audit item, so the newest addition was already unreachable by search
before anyone typed a second query. `ServerAndAppearanceStoreTests
.testSettingsKeywordsCoverRecentlyAddedControls` pins that case and three
others. The "never pane content" half is UNCHANGED: search still matches
only the tab title and this hand-typed list, never the text actually
rendered in a pane, and building real content-indexing search is a
separate, larger feature this pass does not attempt.

Done 2026-09-06 (subagent sampling, was rank 2): every subagent turn ran at
a hardcoded `temperature: 0.2` regardless of the Engine pane's setting.
`AppModel.samplingOptions()` (`AppModel+Generation.swift`) is now the one
place that reads temperature, top-k, top-p, repetition penalty, seed and
stop sequences from the user's live settings, extracted out of
`executeGenerationTurn` so both callers share it rather than drifting the
way `SubagentRunner`'s system-prompt assembler already had (`swift/CLAUDE.md`
Gotcha 46). `SubagentRunner.run`/`runBody` take a `samplingOptions:
GenerateOptions` parameter now. The two `AppModel`-context call sites
(`AppModel+Subagents.swift`, `AppModel+Agents.swift`) pass
`samplingOptions()` directly, and the two `AppToolRegistry` call sites (the
`agent` tool, the skill-invokes-agent path) read it through a new
`AppToolRegistry.subagentSamplingOptionsProvider`, the same provider-closure
pattern `userSystemPromptProvider` already uses for the identical reason
(`SubagentRunner` is an `enum` with no `AppModel` to ask). `maxNewTokens`
stays a subagent-owned `2048` on purpose: a subagent's own tool-use turn
budget is an architectural choice, not a sampling preference, and is a
different quantity than how long a user wants one chat reply to run.
`SamplingOptionsTests.swift` pins the extraction, the disabled-toggle
behavior (a toggle left off must reach the engine's own default, not the
stored-but-unused number), and that constructing an `AppModel` installs the
provider.

Done 2026-09-06 (`LanguageDetector`, was rank 2): `currentKeyboardLanguage()`,
`detectTextLanguage(_:)` and `isRTL(languageCode:)` had zero production
callers (each reached only from its own test) and are deleted. Each reads
like a scaffold for a feature -- auto-selecting the app language from the
keyboard, flipping text direction per message -- that was never wired to a
call site. `currentKeyboardLayoutName()` stays: `GeneralSettingsPaneView`
reads it for the "Active Keyboard Layout" informational row, which is the
"fourth is display-only" half of this item. `import NaturalLanguage`
dropped with `detectTextLanguage`, its only user in the file.

Done 2026-09-06 (`scanModels` in a view body, was rank 3): fixed in both
named files. `ModelStorageManager.scanModels` walks the whole target
directory with a `FileManager` enumerator, stat-ing every entry -- real I/O,
not a property read -- and it ran inline in `body`, so SwiftUI re-ran the
walk, once per configured folder, on every body evaluation either view
received for ANY reason. Both now cache the count in `@State`, recomputed by
a `.task(id:)` keyed on what actually changes it (the folder list in
`CustomModelFoldersSectionView`, the LM Studio path and detection toggle in
`ModelsSettingsPaneView`) rather than on every render, with the existing
"Rescan Now" button also triggering a manual recompute.

Done 2026-09-06 (HF endpoint, was rank 1, last of the six ranked items):
`ServerAdvancedSettingsView`'s field persisted `hfEndpointInput` on change
and stopped there. `HfAuthTokenCardView`'s own editor did the rest --
calling `TurboSparkCatalog.setHfEndpoint`, which is what every model
install, probe and browse call reads OUTSIDE server context -- so editing
through the first field left the catalog on the stale endpoint until the
next app launch (when `loadSettings()` re-applies it), while the second
field applied it immediately.

Each editor also inlined its own trim-and-compare for "what counts as the
default," a second place for the same drift to happen again.
`HfEndpointResolution.effectiveEndpoint(from:)` is now the one function
both call, and `ServerAdvancedSettingsView`'s field calls `setHfEndpoint`
too. It also drops the `.disabled(isRunning)` this pane's other fields
keep: unlike a port or a memory-guard tier, this setting's catalog effect
has nothing to do with whether a server happens to be running, so disabling
it there would only narrow the window in which the two editors disagree
rather than close it. `HfEndpointResolutionTests.swift` pins the shared
function.

This closes every item this audit ranked. Section 4's "Cleared after
checking" list and this section's method (5) still apply to whatever the
next pass finds.

Done 2026-09-06 (server key generate and copy): the API key field in Server
Advanced gained two buttons beside it. Generate fills the field from
`ServerAPIKeyGenerator.generate()` (`sk-` plus a lowercased UUID) and is
disabled while a server runs, with the field it fills, since the key is
read at start; Copy writes the trimmed effective key
(`AppModel.serverAPIKey(from:)`, the value the server actually checks) to
the clipboard and stays enabled while a server runs, because handing the
live key to a client is its purpose. `ServerAPIKeyGeneratorTests` pins the
generated shape. The key itself was already end to end -- `--api-key` on
`turbospark-server`, `TURBOSPARK_API_KEY`, the Keychain store and the
Engine pane's duplicate field predate this -- so these are affordances on
an existing setting, not a new one.

Cleared after checking, so nobody re-derives them: the Profiles caption about
isolation is accurate (skills and agents resolve through
`UserProfileStore.userScopeSubdirectory`), `/v1/embeddings` is a real route
and is listed in `ServerEndpointCatalog`, and every control in the Safety and
Permissions panes reaches a consumer.

## 5. How to re-run this audit

For a persisted field, grep its `AppModel` property name and discard the
hits in `AppModel+Persistence.swift`, `MacAppSettings.swift` and the pane
that edits it. Anything left is a consumer, and nothing left is a dead
setting. For a control, find the bound property and apply the same rule.
For a button, read its action, and for a caption, check the claim against
the code it describes.

The counts in section 1 come from grepping `theme.ui(`, `theme.code(` and
the `.font(` patterns over `Sources/TurboSparkApp`. They are the numbers to
re-take after the font conversion.
