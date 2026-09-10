# swift/

Two SwiftPM packages over the C ABI in `crates/ffi`. `TurboSpark` is the
binding (async/await, `AsyncThrowingStream`, Codable wire types), and
`TurboSparkApp` is a SwiftUI chat and model-management app built on it.
Both are macOS/Apple Silicon only and link the engine IN PROCESS: no HTTP,
no IPC, no server.

`README.md` beside this file is the user-facing quickstart (prerequisites,
API examples). This file is the map plus the gotchas that apply regardless
of what you touch. `swift/docs/` holds feature-specific architecture and
incident detail; the table at the bottom says which page owns which area
-- read the page for the area you are changing, not the whole directory.
Keep code, comments and docs ASCII: no emojis and no em dashes (project
rule).

Read `crates/ffi/CLAUDE.md` first when the change crosses the boundary.
Its Gotchas 1, 2, 7, 8, 9 and 10 are the Rust half of Gotchas 1, 2, 4, 5
and 6 here, and neither half makes sense alone. `docs/SWIFT_BINDINGS.md`
documents the ABI contract itself.

## Layout

```
swift/
+-- docs/                            # feature architecture, audit & reference docs for TurboSparkApp
+-- TurboSpark/                      # the binding (library)
|   +-- Package.swift                # platforms .macOS(.v13); unsafeFlags -L
|   +-- Sources/CTurboSpark/         # module.modulemap (committed)
|   |                                # turbospark.h + libturbospark_ffi.a
|   |                                # STAGED by make swift-lib, gitignored
|   +-- Sources/TurboSpark/
|   |   +-- TurboSparkSession.swift  # open/generate/cancel/tokenize/fitWindow
|   |   +-- Catalog.swift            # list/probe/cost/install/recommend/delete
|   |   +-- Errors.swift             # TurboSparkError + takeString/decode/check
|   |   +-- Options.swift            # OpenOptions, GenerateOptions
|   |   +-- SessionTypes.swift       # SessionInfo (resolved-at-open facts)
|   |   +-- GenerationTypes.swift    # GenerationEvent/Result, PhaseReport
|   |   +-- ChatTypes.swift          # ChatMessage, WindowFitOutcome
|   |   \-- SystemTypes.swift        # ModelRecommendation, SystemTelemetry
|   \-- Tests/TurboSparkTests/
|       +-- SurfaceTests.swift       # ABI/header agreement, NO model needed
|       \-- RealModelTests.swift     # end to end, gated on two env vars
\-- TurboSparkApp/                   # the app (executable)
    +-- Package.swift                # platforms .macOS(.v14); repeats the -L
    +-- Package.resolved             # pins swift-markdown-ui
    +-- Sources/TurboSparkApp/
    |   +-- App/                     # @main scene, RootView, AppSettingsView
    |   +-- Chrome/                  # NavigationRailView, TopBarView,
    |   |                            # StatusBarView, ModelLoaderControl
    |   +-- Files/                   # AttachmentImporter, FilePreviewView,
    |   |                            # FilesSectionView
    |   +-- State/                   # AppModel (@MainActor) + extensions,
    |   |                            # AppChat/AppProject,
    |   |                            # GlobalMcpFileStore, SystemPermissionsManager
    |   +-- Generation/              # composer, output pane, chat sidebar (+ projects,
    |   |                            # chat rows, footer subviews), tool cards,
    |   |                            # ProjectSettingsSheet (+ permissions, rules subviews),
    |   |                            # ProjectMcpSettingsSheet
    |   +-- Installation/            # model hub (+ filter bar), catalog sheet, probe,
    |   |                            # ModelDetailPaneView (+ hardware fit, tech specs)
    |   +-- Diagnostics/             # inspector options, metric formatting, phase counters
    |   +-- Server/                  # ServerPaneView + header band, loaded models,
    |   |                            # charts, console, connect card, advanced;
    |   |                            # ServerMetricsStore and ServerEndpointCatalog
    |   |                            # are pure and are what the tests reach
    |   +-- Presentation/            # markdown render, docx/xlsx/pdf extract
    |   +-- Components/              # AppearanceSettingsPaneView (+ ThemeConfigCard,
    |   |                            # AppearancePreferencesCard), McpSettingsPaneView,
    |   |                            # ModelsSettingsPaneView (+ CustomModelFoldersSection),
    |   |                            # McpServerEditorSheet (+ McpTransportFieldsView),
    |   |                            # McpImportSheet (+ McpCatalogSourceFormView),
    |   |                            # PermissionsSettingsPaneView,
    |   |                            # PluginSettingsPaneView (+ PluginMarketplaceSheet),
    |   |                            # ToastOverlayView, ErrorBanner
    |   +-- Theme/                   # AppearanceSettings, AppearanceTypes, AppDockIconRenderer,
    |   |                            # TurboSparkTheme, AppChromePresentation,
    |   |                            # PointerCursorModifier, ThemeCodePreviewView
    |   +-- Tools/                   # everything a tool call passes through;
    |   |   |                        # swift/docs/SWIFT_TOOLS.md is the map
    |   |   +-- Registry/            # AppToolRegistry (+Handlers, +Vocabulary),
    |   |   |                        # AppToolTypes, AppToolCatalog: the five
    |   |   |                        # files adding a tool touches
    |   |   +-- Core/                # ToolCallParser, AppToolPermissionEngine,
    |   |   |                        # ToolRiskClassifier, CommandGate (+ features),
    |   |   |                        # AppToolSandbox, ProcessExecutor, OpenAIToolSchema
    |   |   +-- Hooks/               # PreToolUse/PostToolUse engine, store, matcher
    |   |   +-- Guardrails/          # ForgeGuardrailsEngine (tool-call rescue)
    |   |   +-- Custom/              # user JSON tools: definition, parser, executor
    |   |   +-- MCP/                 # McpClientEngine, McpServerSpec, ProjectMcpDetector
    |   |   +-- File/ Web/           # per-domain schemas and executors
    |   |   +-- Terminal/            # TerminalTools (schemas only), plus
    |   |   |                        # ShellCommandRunner, BackgroundShellManager,
    |   |   |                        # ShellCwdTracker, ShellOutputFormatting
    |   |   \-- Tasks/ Planning/ Projects/ Automation/
    |   \-- Resources/               # app-prompts.json, Logos/ (Bundle.module)
    \-- Tests/TurboSparkAppTests/    # Unit tests covering appearance settings,
                                     # MCP client engine, project MCP detection,
                                     # project rules detection, system permissions,
                                     # and tool permissions.
```

`AppModel` is split across `AppModel.swift` plus one extension per domain.
**The base file is STORED PROPERTIES and `init`, and nothing else** -- the
file had drifted to 863 lines by growing accessors instead, which is why
four gating predicates were each missing a term a sibling already carried
(`state#76`, `#73`, `#85`). Add new behaviour as a new extension file
rather than growing the base one; if a computed accessor is going into
`AppModel.swift`, it belongs in an extension.

Rather than list the extensions here, where the list rots: `ls
Sources/TurboSparkApp/State/AppModel+*.swift`. Three groupings are worth
knowing because they are not obvious from the names. A TURN is spread over
`+Submission` (up to the appended user turn), `+Generation` (the stream),
`+AgentLoop` (once a tool call is parsed out of the reply), `+Cancellation`
(when it is stopped) and `+History` / `+SkillState` (the two prompt
shapes). MODELS are split by what is on disk (`+ModelDiscovery`) against
what is resident (`+Models`). And the SERVER is split by starting one
(`+Server`) against which models it is holding (`+ServerAttachment`).

**A model's proposed tool calls are parsed in exactly one place**
(`Tools/Core/ToolCallParser`). `AppModel` and `SubagentRunner` carried
byte-identical copies of that parser until 2026-09-04, and that
duplication is the concrete reason `state#68`, `#74` and `#75` were three
separate discoveries: a fix to how the main loop reads or answers a call
had no way of reaching the isolated one. Each caller keeps its own guard,
which genuinely differs, and neither keeps its own parser.

The plugin system, its marketplace, and the shared `resolveExecutablePath`
lookup are documented in full in `swift/docs/SWIFT_PLUGINS.md` and
`swift/docs/SWIFT_TOOLS.md` -- read those before adding a contribution
surface or re-deriving the enable cascade.

## Build, test, dev commands

Every target below is a root `Makefile` wrapper. Run them from the
repository root.

```sh
# Builds crates/ffi for aarch64-apple-darwin and STAGES the archive plus the
# canonical header into Sources/CTurboSpark. Nothing Swift here compiles
# until this has run once: SwiftPM cannot reach outside its own package
# directory, so both files are copies and both are gitignored.
make swift-lib

# The binding's own suite. Needs no model, ~1 s. This is the ONLY thing in
# the repository that can check the hand-written turbospark.h: the Rust tests
# reach the same function bodies through the rlib and pass against a wrong
# declaration.
make swift-test

# Same suite plus the end-to-end arm (minutes, real Metal forward pass).
# Without MODEL those cases SKIP with a note; a MODEL pointing at a missing
# install FAILS.
make swift-test-real MODEL=~/models/gemma4.gturbo

# BLOCKED is a SECOND install and reaches what MODEL structurally cannot.
# Point it at a MoE or sub-4-bit install; the tests CHECK that it is one.
make swift-test-real MODEL=~/models/qwen38-27b-mtp.gturbo \
                     BLOCKED=~/models/ornith35b.gturbo

# Run the TurboSparkApp test suite. Seconds for the whole thing, so run it
# freely; no count is quoted here because the last one sat at "44+" until the
# suite had grown past 400.
cd swift/TurboSparkApp && swift test

# One suite, for the edit loop: ~0.3 s against ~9 s for the whole thing.
cd swift/TurboSparkApp && swift test --filter AppearanceSettingsTests

# The app.
make swift-app-build        # debug
make swift-app-release      # release
make swift-app              # build and run
make swift-demo             # alias for swift-app; TurboSparkDemo is gone (Gotcha 10)

# Removes both .build trees AND the two staged files, so the next Swift
# build needs `make swift-lib` again.
make clean-swift
```

Iterating on SwiftUI only? Call SwiftPM directly and skip the staging step:

```bash
cd swift/TurboSparkApp && swift run TurboSparkApp
```

That is not a shortcut for convenience alone. `make swift-app` depends on
`swift-lib`, which `touch`es every `.swift` file in both packages
(Gotcha 3), so going through `make` recompiles the whole app every single
time. Use `make` when the Rust side moved and SwiftPM when it did not.

`swift run TurboSparkApp` has no bundle identifier, so LaunchServices-based
screenshot tooling (agentic `list_apps`/`request_access` automation) cannot
target it -- both return empty/not-found. Build `make app-bundle` first if a
session needs to screenshot the app programmatically.

## Adding or changing text, fonts, themes, or localized strings

Read this before adding ANY user-visible text, a font treatment, a theme
color, or a localized string, however small. Each rule exists because
its violation already cost a real cleanup pass -- the two big ones are
the 2026-09-06 font-propagation fix (raw `.font(...)` calls that never
read the theme) and the 2026-09-08 type-scale standardization (437
numeric font sizes across 28 distinct values playing the same roles),
both recorded in `swift/docs/SWIFT_SETTINGS_AUDIT.md` section 1.

**Size text BY ROLE, never by number.** `AppFontStep` is the canonical
scale; its doc table lists every role's rendered size at the 16pt
default base and is the one change point for retuning. Write
`.themedFont(.small)` / `.themedCode(.small)` (they read `\.appTheme`
themselves, so the call site needs no environment declaration) or
`theme.ui(.role)` / `theme.code(.role)` where a `Font` value is wanted.
The explicit-size spellings (`theme.ui(points:)` and friends) are
DELETED, so an off-scale size is a compile error, not a review comment
(Gotcha 63). The one numeric survivor, `.themedFont(fitting:)`, is for
glyphs whose size a FRAME dictates (a logo letterform at
`size * 0.45`), and must never take a constant.

**A size the scale lacks is a new role, not a workaround.** Add an
`AppFontStep` case, document its rendered size in the doc table, and
update the pinned expectation in
`FontPropagationTests.testCanonicalScaleRendersDocumentedSizes`. That
is the whole point of the scale: retuning a role is one reviewed edit
that moves every surface playing it.

**Never give a view its own family or color.** Font families come from
the Appearance settings through the theme; a call site that must LOOK
serif or monospaced at the default setting passes `systemDesign:`,
which applies only while the family is the system face. Colors come
from the `TurboSparkTheme` accessors or the environment theme -- never
`Color(nsColor: .separatorColor)` or friends, which ignore the palette
entirely (Gotcha 62). Read `\.appTheme` in a view body; calling
`AppearanceManager` directly from one is the pre-`ResolvedAppTheme`
bug that made settings only sometimes apply.

**Every `.font(` call reads the theme.** `FontPropagationTests` checks
this per CALL SITE -- one themed call no longer whitewashes a file of
raw ones -- and rejects raw size literals in any spelling. A themed
font hoisted into a local must carry a marker in its name (`uiFont`,
`codeFont`, ...); the marker list and the per-file exempt list (the two
per-mode preview cards, the font-picker rows, the modifier itself) are
explicit in the test with reasons, so extending either is a visible
diff, not a silent widening.

**User-facing strings are localized in the same change.** `Text("...")`
takes `bundle: .module`; the key goes into
`TurboSparkApp/Localization/Localizable.xcstrings` with a translation
for EVERY language in the catalog in the same commit, and new strings
are listed separately in the delivery so they are not
reverse-engineered from the diff. `swift build` never compiles the
catalog itself -- only `make compile-strings` does (`swift-app` runs it
for you). The six parity gates and the rest of the pipeline are
`swift/docs/SWIFT_LOCALIZATION.md`.

**The edit-loop gate for any of the above** is
`swift test --filter FontPropagationTests` (~0.3 s) plus a green
`swift build` -- between them they catch an off-scale size, a raw
literal, a non-themed `.font(`, and a moved scale factor before anything
else has to. The AppKit print/export surfaces
(`NSFont.systemFontSize`) are the documented exception to all of this,
not an invitation; new SwiftUI UI has no business there.

## Gotchas

Cross-cutting: build system, the binding's own contract, and rules that
apply whatever area of the app you are touching. A gotcha specific to one
feature has moved to its `swift/docs/` page; its number stays here as a
pointer so an existing cross-reference (`Gotcha N`, a `state#N` comment)
keeps resolving.

1. **`cancel()` IS WHY `TurboSparkSession` IS A CLASS WITH A SERIAL QUEUE
   RATHER THAN AN `actor`, AND CONVERTING IT WOULD BREAK STOP SILENTLY.** An
   actor method cannot run while another is in flight, so `cancel()` on an
   actor would suspend behind the generation it is cancelling and take effect
   only once the model had finished on its own: the Stop button would appear
   to do nothing on exactly the long turns it exists for. Everything except
   `cancel()` runs on the private serial `queue`, which is also what meets
   the C layer's one-generation-at-a-time contract by construction. The
   `@unchecked Sendable` on the type and on the `Handle` wrapper rests on
   that confinement plus the header's statement that `ts_session_cancel` is
   an atomic store safe from any thread (`crates/ffi/CLAUDE.md` Gotcha 1).
   Do not "modernize" this into an actor.

2. **A SwiftPM `-L` FLAG IS RESOLVED AGAINST THE PACKAGE BEING BUILT, NOT THE
   ONE THAT DECLARED IT, SO EVERY CONSUMER REPEATS IT.** `TurboSpark`'s
   `Package.swift` carried `-LSources/CTurboSpark`, correct when its own
   tests link and wrong for a dependent; `TurboSparkApp` carries
   `-L../TurboSpark/Sources/CTurboSpark`, the same directory seen from its
   own root. Since 2026-09-05 the LIBRARY target declares no `-L` at all --
   the flag moved to `TurboSparkTests`, the only target in that package that
   links, and the library face carries only `.linkedLibrary("turbospark_ffi")`.
   The `unsafeFlags` still block `TurboSpark`'s TEST target from being
   consumed as a versioned dependency by an out-of-repo package, which is
   accepted.

3. **SwiftPM DOES NOT TREAT THE STATICLIB AS A BUILD INPUT, so a rebuilt
   `.a` under unchanged Swift sources triggers NO relink.** The archive
   arrives through an unsafe linker flag, which SwiftPM passes through
   without adding a dependency edge. The test target then links the PREVIOUS
   library and reports on code that is no longer in the tree, which is
   exactly the wrong failure for the one suite that can check the header.
   `scripts/swift-lib.sh` ends with a `find ... -exec touch {} +` over both
   packages' sources for that reason. Two consequences: the touch is
   load-bearing and must not be tidied away, and it is why `make swift-app`
   pays a full Swift rebuild every invocation.

4. The install byte callback fires concurrently from several download
   threads, and the walk cannot resume. Moved to
   `swift/docs/SWIFT_MODEL_HUB.md`.

5. **TWO TYPES IN `Catalog.swift` DECODE snake_case WHERE EVERYTHING ELSE
   DECODES camelCase, DELIBERATELY.** The rest of this binding's JSON is a
   wire shape invented for it, so the Rust side emits camelCase and no
   `CodingKeys` exist anywhere, which is what makes a spelling drift on
   either side a test failure rather than a silent nil. `CatalogEntry` and
   `InstalledModel` are the engine's own on-disk formats (`models.json` and
   `~/.turbospark/installed.json`) passed through unchanged, so a GUI's rows
   and the CLI's rows are provably the same data. `decode` deliberately does
   NOT set `keyDecodingStrategy`. The drift this guards against was caught
   here once already, on `downloadBytes`.

6. `SessionInfo` holds what was resolved, not what was asked for. Moved to
   `swift/docs/SWIFT_SESSION_CAPABILITIES.md`.

7. **`liveTokenCount` COUNTS NON-EMPTY CONTENT EVENTS, NOT TOKENS, SO THE
   HUD's tok/s IS NOT A THROUGHPUT MEASUREMENT.** `crates/ffi`'s
   `on_progress` emits `TS_EVENT_CONTENT` only when the channel splitter
   returns non-empty text, and the Swift callback drops empty payloads on top
   of that. Special tokens decode to the empty string, the streaming
   detokenizer withholds partial UTF-8 and releases it as a tail, and
   REASONING text goes to a different event entirely. So the live counter
   undercounts on any turn with framing or thinking in it. The authoritative
   figures are `GenerationResult.newTokens` and `tokensPerSecond`, which
   `AppDiagnostics` uses. Never quote the live number in a benchmark.

8. **AN EMPTINESS GUARD BELONGS ON A DECODER'S OUTPUT, NEVER ON ITS INPUT**
   (root Gotcha 44). The `streamCallback` in `TurboSparkSession.swift` does
   drop `len == 0`, and that is safe ONLY because the channel state machine
   already ran on the Rust side and this callback feeds no state machine of
   its own. Any future Swift-side parser over the event stream must not
   inherit that line: on Harmony every frame token arrives as empty text, and
   skipping them makes the whole turn read as one run of content.

9. Build the reasoning level picker from `info.reasoningEfforts`, never
   from the family. Moved to `swift/docs/SWIFT_SESSION_CAPABILITIES.md`.

10. **`TurboSparkDemo` IS GONE; `swift-demo` IS NOW AN ALIAS FOR
    `swift-app`.** What ships is `TurboSparkApp`: multi-chat with
    persistence, a project/agent system that executes tools, a model hub
    with catalog install and Hugging Face probing, document attachment, and
    a phase-counter inspector. Any older note describing a "deliberately
    minimal" demo is describing a deleted target.

11. `resolveSecurePath` did not check containment until 2026-08-28. Moved
    to `swift/docs/SWIFT_TOOLS.md` section 16.

12. **RELEASE ARTIFACTS AND THEIR TRIPLE-COPIED VERSION NUMBER.**
    `docs/RELEASE.md` is the home for `scripts/make-app-bundle.sh` and
    `scripts/make-dmg.sh`; read it before cutting a release. Worth carrying
    without opening that page: `swift run` produces a bare executable, not
    a bundle, so identity (bundle id, `Info.plist`, the preferences domain)
    exists only in the bundling script, and a `swift run` build's
    `@AppStorage` settings and the installed app's do not share a
    preferences file. `LSMinimumSystemVersion` (the script), `.macOS(.v14)`
    (`TurboSparkApp/Package.swift`) and the Homebrew casks' `>= :sonoma`
    are three copies of one number; change one and change all three.

13. All app state lives in three JSON files, and a decode failure used to
    be silently fatal to the file's content. Moved to
    `swift/docs/SWIFT_STORAGE.md`.

14. See Gotcha 12.

15. **THE 400-LINE GUIDELINE IS ACTIVELY ENFORCED VIA SUBVIEW MODULARITY.**
    Large views (`ChatSidebarView`, `ModelDetailPaneView`, `ModelHubView`,
    `AppearanceSettingsPaneView`, `ProjectSettingsSheet`,
    `ModelsSettingsPaneView`) and core classes (`AppModel`,
    `AppearanceSettings`) are decomposed into dedicated subviews, domain
    extensions, and type files. Split by subview and functional extension
    when touching or extending them rather than growing a single file.
    Nothing in `TurboSpark/` is near the limit: the binding is thin on
    purpose, and logic that creeps into it is logic no Rust test can reach.

16. **THE APP CONSUMES THE STREAM ON THE MAIN ACTOR, ONE `objectWillChange`
    PER EVENT.** `AppModel` is `@MainActor`, so the `Task` in
    `executeGenerationTurn` inherits it and every `.content` chunk mutates a
    `@Published` property on the main thread. That is what makes the
    transcript live, and it means a view that does expensive work per body
    evaluation (a full markdown re-parse, a re-render of the whole
    transcript) pays it per token. `ResponseMarkdownRenderer` and
    `ChatMessageMarkdownView` are where that cost is; measure there before
    blaming the engine for slow-looking decode. `session.cancel()` is called
    directly (not through the actor) and is why Stop is immediate.

17. **THE WINDOW IS FOUR CHROME BANDS AND EACH ONE OWNS ONE QUESTION.**
    `RootView` is a sidebar (sections, and the chat list when there is one),
    a top bar (what the machine is doing),
    the working panes, and a status strip (context fill, fans, tok/s,
    tokens, thermal, memory pressure). "Which model, is it loaded" is NOT a
    top-bar question since 2026-09-09: `ModelLoaderControl` moved under the
    prompt composer, in a compact one-line density, and is chat-scoped
    rather than window-scoped -- so it is absent from the Files, Installed,
    Discover and Server sections by design.

    **THE LEFT COLUMN IS ONE SURFACE IN TWO PRESENTATIONS, NOT TWO BANDS.**
    A permanent 52pt icon rail used to sit beside a 260pt chat sidebar, which
    read as two vertical bars stacked together. `AppSidebarView` is the single
    column since 2026-09-09: expanded it is `SidebarSectionNavView`'s labeled
    rows over `ChatSidebarView`, collapsed it is `NavigationRailView`
    unchanged. **The toggle no longer HIDES anything**, and that is the whole
    design rather than a detail -- the old split existed so sections stayed
    reachable while the sidebar was hidden, and folding navigation into a
    hideable sidebar would strand it twice, once on Cmd+B and once in the four
    sections that never showed a chat list. Collapsing to the rail satisfies
    the same constraint with one column. Three consequences. The window
    minimum drops by `navigationRailWidth + dividerWidth` when expanded,
    because the sidebar's width REPLACES the rail's instead of stacking on it
    (`minimumWindowWidth` takes `isSidebarExpanded:` and adds one column).
    `RootView.showsChatSidebar` is gone; `activeSection == .chat` is
    `AppSidebarView`'s business and must NOT be re-conjoined into the width
    call, or the window can shrink under its own sidebar in Files and Server.
    And the profile/settings footer moved OUT of `ChatSidebarView` into
    `AppSidebarView`, because it has to be present in the sections that have
    no chat list to hang it off.

    **The top bar is THREE GROUPS, and the middle one is `.fixedSize()`.**
    It used to be one left-aligned run, on the reasoning that the phase
    indicator appears and disappears once per turn and a centred model
    loader would slide every time it does. Since 2026-09-07 the bar is a
    leading group (sidebar toggle, phase indicator, git pill), a fixed-size
    `ChromeTelemetryView` in the centre, and a trailing group (inspector
    toggle). Both outer groups are `frame(maxWidth: .infinity)`, so an
    HStack splits the slack evenly and the centre does not move. **KEEP THE
    TRAILING GROUP'S FLEXIBLE FRAME even though one control is left in it**
    -- it is half of what holds the centre still, and it looks exactly like
    the kind of wrapper a cleanup deletes. The original hazard is still
    real: it is the LAYOUT that defuses it, so keep the `.fixedSize()` and
    both flexible frames if you touch that band.

    **`ChromeTelemetryView` polls memory and CPU on the same 2-second timer
    the status strip used, and that is the whole reason it is allowed up
    there.** The strip's old `memoryReadout` and `cpuReadout` were deleted
    when it landed, along with their `memoryHistory`/`cpuHistory` buffers
    (nothing else read them, so the graph mode lost those two sparklines and
    keeps the throughput one). Per-TOKEN numbers -- tok/s and the token
    count -- stay in the strip, because a number that updates per token
    pulls the eye and that is what the original note was about. Do not move
    those up. `FanReadoutView` carries its own LEADING divider and now takes
    `showsLeadingDivider:`, because with memory and CPU gone it can be first
    in the strip.

    The rail carries five sections since 2026-08-30 (Chat, Files,
    Installed, Discover, Server); Server was APPENDED at the last shortcut
    rather than inserted, so the four that existed keep the numbers anyone
    has already learned. `testEverySectionHasAUniqueTitleAndShortcut`
    guards a missing or duplicated tooltip.

    **THE WINDOW OPENS AT 94% OF THE SCREEN'S VISIBLE FRAME, AND
    `.defaultSize` ALONE COULD NOT DELIVER THAT.** That modifier decides the
    FIRST launch only; macOS autosaves a `Window` scene's frame and restores
    it forever after, so raising the default is invisible to every existing
    install, which is everyone. `ForegroundAppDelegate` therefore carries a
    one-time migration keyed on
    `TurboSpark.didAdoptRoomierDefaultWindowFrame`, which grows the restored
    frame PER AXIS and only upward -- the first cut tested
    `width < target || height < target` and then assigned the target flat,
    which grew the height and made the window NARROWER on a machine already
    wider than the target. It runs async, because at
    `applicationDidFinishLaunching` the SwiftUI scene has not built its
    window yet and `NSApp.windows` is empty.

    The right column is one slot, not two: `previewAttachment != nil` takes
    it from the inspector, which is why `.toggleInspector` closes the
    preview first. And the window is `.hiddenTitleBar`, so the system still
    draws the traffic lights over the content at roughly x = 13 to 66;
    `AppChromeLayout.trafficLightClearance` keeps the top bar's first
    control clear of them.

18. **`TextEditor` HAS NO INTRINSIC HEIGHT, AND THE TWO OBVIOUS WAYS TO GIVE
    IT ONE BOTH SILENTLY PICK THE MAXIMUM.** It is greedy vertically, so an
    empty composer renders exactly as tall as a full one. A ZStack with a
    hidden sizer `Text` behind it does not help (a ZStack sizes to its
    LARGEST child, which is the editor), and neither does
    `.frame(minHeight:maxHeight:)` -- that frame is FLEXIBLE, reporting the
    proposal clamped into range rather than the child's ideal, so with a tall
    proposal it always lands on `maxHeight`. Both look correct in code review
    and are only visible in a screenshot.

    `PromptComposerView.editor` measures instead: a hidden `Text` with the
    same font and insets, `fixedSize(horizontal: false, vertical: true)`,
    reports its ideal height through a `PreferenceKey`, and the editor gets
    `min(max(measured, floor), ceiling)`. Reach for the measurement whenever
    a SwiftUI view has to size to content that a greedy child would otherwise
    swallow.

19. **macOS TCC AND PROTECTED FOLDER ACCESS.** Access to `~/Documents`,
    `~/Downloads`, and `~/Desktop` is governed by macOS Transparency,
    Consent, and Control. `SystemPermissionsManager` probes directory
    readability via `contentsOfDirectory(atPath:)` and triggers interactive
    `NSOpenPanel` approval or deep-links directly to macOS System Settings.
    Users can also authorize custom project directories in the Files &
    Permissions settings tab. `state#62`: the home row could not report
    anything other than granted, and overlapping probes published stale
    readings.

20. MCP servers and streams (stdio and SSE). Deleted -- fully covered by
    `swift/docs/SWIFT_TOOLS.md`'s existing map (sections 10-11) and its
    Gotchas section (16).

21. Dynamic theme and accent injections. Moved to
    `swift/docs/SWIFT_SETTINGS_AUDIT.md`.

22. A badge or a filter that cannot fail carries no information -- the
    Model Hub shipped three of them. Moved to
    `swift/docs/SWIFT_MODEL_HUB.md`.

23. A zero from an absent measurement is not a measurement of zero. Moved
    to `swift/docs/SWIFT_MODEL_HUB.md`.

24. **TWO UNRELATED THINGS IN THIS APP ARE CALLED "GUARDRAILS."**
    `AppGuardrailsMode` (`State/MacAppSettings.swift`) is FORGE TOOL-CALL
    guardrails: whether a model's tool calls get dialect rescue and schema
    validation (`swift/docs/SWIFT_TOOLS.md`). `AppLoadGuardOption`
    (`State/AppRuntimeOptions.swift`) is MEMORY guardrails: how much of the
    machine a model may commit when it loads (`docs/LOAD_GUARD.md`). They
    share no code, no settings key and no UI surface, and the second is
    deliberately NOT named `AppGuardrailsMode` -- the collision was caught
    before the type existed rather than after. `MacAppSettings` carries
    BOTH `guardrailsMode` and `loadGuard` as separate persisted keys.

25. `AppModel.activeLoadGuard` exists so the hub and the loader cannot
    resolve different memory tiers. Moved to `swift/docs/SWIFT_MODEL_HUB.md`.

26. `AppModel.server` outlives `AppModel.session` unless something stops it
    first. Moved to `swift/docs/SWIFT_MODEL_HUB.md`.

27. **SWIFT COMPILER WARNINGS: NON-THROWING `URL` PATHS AND `onChange` ARITY.**
    `URL(fileURLWithPath:).standardizedFileURL.path` is non-throwing;
    wrapping it in `try?` warns ("no calls to throwing functions occur
    within 'try' expression"). And `.onChange(of:perform:)` with a
    single-parameter closure is deprecated on macOS 14.0+; use the
    0-parameter or 2-parameter form instead.

28. **A DEFAULT SPELLED AT FOUR SITES DISAGREED AT THE ONE USERS REACH, AND
    THE PERMISSIVE SPELLING WAS THE SHIPPED ONE.** `AppProject.init`, its
    decode fallback and `AppModel.createProject` all defaulted to
    `AppProjectPermissions.standard` (`terminal: .ask`).
    `ProjectSettingsSheet`'s new-project branch seeded `.auto`
    (`terminal: .allow`, `fileWrite: .allow`, `mcp: .allow`). A project can
    only be created through that sheet, so every project any user ever made
    ran model-proposed shell commands with no prompt, while the
    conservative default the other three sites agreed on was decoration.
    All four now read `AppProjectPermissions.newProjectDefault`; a fifth
    site spelling its own is what that constant exists to make visible.
    `swift/docs/SWIFT_TOOLS.md` section 16 has the containment half of this
    story.

29. A denylist over a string bound for `/bin/zsh -c` is the wrong shape,
    not an incomplete list. Moved to `swift/docs/SWIFT_TOOLS.md` section 16.

30. A projectless chat has no workspace, and there is no defensible
    default. Moved to `swift/docs/SWIFT_TOOLS.md` section 16.

31. **A CLASS INITIALIZER THAT THROWS PART-WAY DOES NOT RUN `deinit`, SO A C
    HANDLE ACQUIRED BEFORE THE THROW LEAKS IN FULL.** `TurboSparkSession.init`
    used to assign `handle` and then read `ts_session_info_json` into `info`;
    a failure there left an instance that was never fully initialized, Swift
    skipped `deinit`, and `ts_session_close` never ran -- stranding mapped
    weights, KV cache and compiled Metal pipelines for the life of the
    process, silently, with the caller seeing exactly the error it expected.
    The info read now happens before any stored property is assigned and
    closes the handle by hand on the failure path.
    `SurfaceTests.testAThrowingInitDoesNotRunDeinitSoCLeanupMustBeManual`
    pins the language rule rather than the initializer, because making the
    real one throw at that point needs a fault injection the C ABI does not
    offer.

    **AND `generate` USED TO RETAIN NOTHING FOR THE TURN.** Its worker was
    `queue.async { [handle] in ... }` -- the capture list named `handle`
    alone, `onTermination` is `[weak self]` by design, and the returned
    stream held no reference back, so no strong reference to the session
    existed anywhere during a generation. A caller releasing its session
    mid-turn ran `ts_session_close` beside `ts_generate`, which
    `turbospark.h` forbids. The capture is `[self]` now and that is
    load-bearing; the app never hit it only because `executeGenerationTurn`
    binds `session` strongly.

32. An unimplemented transport must throw, not return a success string --
    the MCP SSE stub used to fabricate success. Moved to
    `swift/docs/SWIFT_TOOLS.md` section 16.

33. Images travel as ordered content parts, matched to the reference
    rather than chosen. Moved to `swift/docs/SWIFT_VISION.md`.

34. **A SWIFT BUILD IN A WORKTREE FAILS ON A MISSING HEADER, BECAUSE
    `CTurboSpark` IS GITIGNORED.** Gitignored files are not carried into a
    worktree (root Gotcha 13). The staged `libturbospark_ffi.a` and
    `turbospark.h` are exactly that. Either run `make swift-lib` in the
    worktree, or copy the directory across:
    `cp -R swift/TurboSpark/Sources/CTurboSpark <worktree>/swift/TurboSpark/Sources/`.

    **AND `Bundle.module` INSIDE A TEST TARGET IS NOT THE APP'S BUNDLE.**
    `TurboSparkAppTests` declares no resources of its own. `Bundle.module`
    there resolves to the test bundle and finds nothing. Reach an app
    resource through an accessor in the APP target instead.
    `CommandGate.oracleFixtureURL` is the pattern.

35. **THE AGENT LOOP HAS TWO PROMPT SHAPES, AND THE PATCH MUST MERGE BEFORE
    TOOL CALLS ARE PARSED.** `executeGenerationTurn` branches once, at the
    context assembly: the append-only walk over every message and every
    tool result, or `buildSkillStateHistory`, which sends
    [system + protocol][the task][current state][newest observation] and is
    O(1) in step count. The branch is `selectedProject?.skillStateEnabled`,
    default FALSE. On a 4,096-context install the append-only arm stops
    between step 30 and 35 of 50 with an HTTP 400, where the bounded arm
    finishes at 3 to 5x fewer tokens (`docs/SKILL_STATE.md`).

    A state patch is JSON in the SAME reply that may carry a tool call, so
    `applySkillStatePatch` runs first and strips the `<state_patch>` block
    before `extractToolCalls` ever sees it; parsing calls first makes the
    tool parser take the patch for a bogus call. An invalid patch is
    DROPPED, not partially applied: a bad merge would silently corrupt every
    step after it, while a dropped one costs one step's bookkeeping and
    lands on `skillStateLastError`. `AppSkillStateSchema` is the single
    source for the prompt AND the validator, deliberately generic rather
    than per domain.

36. `NSApp.effectiveAppearance` is not moved by `.preferredColorScheme`,
    and two packaging traps make a registered font fail silently. Moved to
    `swift/docs/SWIFT_SETTINGS_AUDIT.md`.

37. The test suite overwrote a user's chat archive, and the only symptom
    was a chat that would not stay deleted. Moved to
    `swift/docs/SWIFT_STORAGE.md`.

38. **AN INCREMENTAL `swift build` THAT DID NOTHING PRINTS THE SAME
    `Build complete!` AS ONE THAT SUCCEEDED.** A run that compiles emits
    `[N/M] Compiling ...`; a no-op emits `[0/3] Write swift-version...` and
    `Build complete! (0.2s)`. Read the output of the run that COMPILED, or
    `touch` the files first.

39. `AppModel().someProperty == someDefault` tests `MacAppSettings`'s
    default, not the property declaration. Moved to
    `swift/docs/SWIFT_STORAGE.md`.

40. `AppStorageRoot` covers the app's stores and not the engine's, so a
    test reaching `TurboSparkCatalog` reads real user data. Moved to
    `swift/docs/SWIFT_STORAGE.md`.

41. **TWO THINGS THAT LOOK LIKE FAILURES AND ARE NOT, WHEN MUTATION-CHECKING
    SWIFT.** A Swift TRAP (`Int(1e300)`, an out-of-range slice) aborts the
    xctest process with `exited with unexpected signal code 5` and prints
    `Fatal error:` -- there is no `Test Case ... failed (` line at all, so a
    harness keying on that string reports the mutation as SURVIVING, or as a
    build error, when it actually reproduced the bug exactly. Read the raw
    output for a trap before believing either verdict.

    And `swift build` run from a SUBDIRECTORY of the package fails with
    `linker command failed` and no error line above it:
    `-L../TurboSpark/Sources/CTurboSpark` is resolved against the CWD
    (Gotcha 2), not against the package root. Build and test from
    `swift/TurboSparkApp` itself.

42. **A RED PRE-EXISTING TEST AFTER A STATE FIX IS AS LIKELY TO BE THE DEFECT
    REPRODUCING AS A REGRESSION.** `testSystemPermissionsManagerStatusProbe`
    asserted the home folder reads `.granted`, which was only ever true
    because that row could not report anything else (`state#94`); a second
    case asserted a level was keyed on the wrong field, which is what the
    code did and what the fix corrected (`state#96`). Both look exactly
    like a refactor that broke something. The tell is one question about
    the ASSERTION, not the diff: does it state a requirement, or restate
    the implementation. Update the assertion WITH the reason; do not revert
    the fix and do not delete the case.

43. Which directories a test may write to is a three-way answer, and only
    two of them are written down. Moved to `swift/docs/SWIFT_STORAGE.md`.

44. **THREE MECHANICS OF THE APP SUITE, EACH ONE BUILD CYCLE'S WORTH.**
    `swift test --filter SuiteName/testCaseName` runs ONE case in ~0.3 s,
    against ~15 s for the whole suite -- what makes a mutate/run/restore
    loop cheap enough to do per assertion. `swift test` runs TWO harnesses
    and prints two summaries: swift-testing's `Test run with 0 tests in 0
    suites passed` is not the result, and a `| tail` lands on it; read
    `Executed N tests, with M failures`. And `AgentManager.shared.builtInAgents[0]`
    is `explore`, which disallows every write -- resolve a subagent by
    NAME (`findAgent(name: "general-purpose")`), never by index.

45. **ONE JOB IN THIS REPOSITORY COMPILES SWIFT, IT IS NOT THE ONE PRs RUN,
    AND ITS SDK IS NOT YOURS.** `verify` runs `macos-latest` and builds
    Rust; `package-macos` (and `release.yml`'s `build-macos`) is push-only
    and is the only thing that compiles the app. It pinned `macos-14` --
    Xcode 15, the macOS 14 SDK -- which nothing in the local verification
    policy reaches, so the Swift half can sit broken on `main` while every
    local build is green. The runner is `macos-15` now, and the SOURCE was
    made to compile on 14 anyway: bumping the runner does not lower that
    floor (`.macOS(.v14)`, `Info.plist` `14.0`), and `release.yml`'s
    `check-version` job asserts the two agree.

    Three failure modes an older SDK rejects and a newer one accepts, and
    only the first is what anyone expects. A guarded call to a newer API
    still has to COMPILE: `if #available(macOS 15.0, *)` around
    `Color.mix(with:by:)` is correct about runtime availability and
    irrelevant to the SDK, which has no such member at all -- `#available`
    gates execution, not symbol resolution; reach for the AppKit equivalent
    instead. Only `body` is main-actor-isolated on the older SDK: a newer
    SwiftUI infers `@MainActor` for the whole conforming type, so an
    explicit `@MainActor` on the TYPE (not the member) is needed for
    helpers that call each other. And the type checker's budget is
    smaller: a `Form` holding five inline `Section`s exceeded the solver
    there while compiling fine locally; one property per section.

    The standing consequence survives the runner bump, because the gap
    only narrowed: `swift build` passing locally still says nothing about
    the job that ships the DMG, and the feedback arrives on a push to
    `main` rather than on the PR.

46. The system prompt has three sources and two assemblers that share no
    code. Moved to `swift/docs/SWIFT_CONTEXT_RING.md`.

47. This app had three answers to "does this model support tool calls"
    and they disagreed. Moved to `swift/docs/SWIFT_SESSION_CAPABILITIES.md`.

48. Steering resolves once, at model open, and the app used to say
    nothing about that. Moved to `swift/docs/SWIFT_SESSION_CAPABILITIES.md`.

49. The app's guardrails setting did not reach the served path. Moved to
    `swift/docs/SWIFT_TOOLS.md` section 16.

50. Shell execution moved out of the tool registry into four
    single-purpose files. Moved to `swift/docs/SWIFT_TOOLS.md`.

51. `continue: false` is not a block, and conflating them costs a Stop
    re-entry. Moved to `swift/docs/SWIFT_TOOLS.md`.

52. Two settings surfaces moved off their old stores (Keychain,
    `appearance.json`), and one has no test coverage on purpose. Moved to
    `swift/docs/SWIFT_SETTINGS_AUDIT.md`.

53. Ghost mode (temporary chats) has two layers, and only the first is a
    guarantee. Moved to `swift/docs/SWIFT_GHOST_MODE.md`.

54. Compaction makes the prompt disagree with the transcript by exactly
    the boundary. Moved to `swift/docs/SWIFT_COMPACTION.md`.

55. A ThermalForge fan hold outlives this app, and the quit-restore is the
    feature's safety net. Moved to `swift/docs/SWIFT_FAN_CONTROL.md`.

56. A setting is dead until its accessor has a caller outside the pane
    that edits it. Moved to `swift/docs/SWIFT_SETTINGS_AUDIT.md`.

57. Auto-memory has four load-bearing joints, and each fails silently when
    rebuilt wrong. Moved to `swift/docs/SWIFT_MEMORY.md`.

58. Chat search excludes ghosts by the flag, and its dialog keys are
    window-level shortcuts, not a key monitor. Moved to
    `swift/docs/SWIFT_CHAT_SEARCH.md`.

59. The menus show one chord per action, so an alternate chord is a
    hidden button, not a second menu item. Moved to
    `swift/docs/KEYBOARD_SHORTCUTS.md`.

60. A command-menu item written the natural way is English under every
    language, and nothing in the build stops it. Moved to
    `swift/docs/SWIFT_LOCALIZATION.md`.

61. **AGENT MODE'S CLASSIFIER IS THE ASK-BAND, AND `hardGated` IS WHAT
    KEEPS THE HARD STUFF OUT OF IT.** `.agentAuto` routes the engine's
    `.ask` verdicts to a local-model classifier, and the whole safety
    argument rests on `ToolRiskAssessment.hardGated` being set by every
    deterministic guard (terminal denylist, the structural shell check,
    the `CommandGate` veto, sensitive paths, sandbox-denied web targets,
    the repo-import MCP gate) and by NOTHING else. A new `.high` source
    that forgets to set it becomes classifier-judged; a new guard that
    marks an unrecognized-command ask hard kills the feature's whole
    population. The engine's step 7b must stay conditional on `.auto` OR
    `.agentAuto` -- with only `.auto`, a cloned `.mcp.json` is judged by a
    model instead of by the user. Routing is
    `AgentModeRouting.preClassifierDecision` and both call sites (main
    loop and `SubagentRunner+Gate`) go through it; a third inline copy is
    how the two drift. See `swift/docs/SWIFT_AGENT_MODE.md`.

62. **A THEMED ACCESSOR THAT RETURNS A SYSTEM COLOR IS NOT THEMED.**
    `TurboSparkTheme.surfaceColor`/`barBackgroundColor`/`sidebarBackgroundColor`/
    `railBackgroundColor`/`hairlineColor` used to return
    `.controlBackgroundColor`/`.windowBackgroundColor`/`.separatorColor`
    (system materials) instead of deriving from the active theme's own
    background hex, and `RootView.swift` painted the window root with the
    same raw system color directly -- so a custom dark theme still drew
    system-neutral-gray boxes with no relation to the palette. Fixed
    2026-09-08: every accessor now blends the theme's own background
    lighter/darker by a fixed fraction (`TurboSparkTheme.elevate`). Reach
    for one of these accessors on any new surface, never a raw
    `Color(nsColor: ...)`. Separately: a `Section("...")` inside
    `Form(.grouped)` draws AppKit's own grouped-list chrome, which no
    SwiftUI `.background()` can override -- confirmed across the whole
    Model Settings panel.

63. **A FONT SIZE AT A CALL SITE IS A ROLE OR IT IS DRIFT.** The 09-06
    font-propagation pass made every site READ the theme but left every
    site free to pick its SIZE, and two conventions grew beside each
    other: named steps and explicit `theme.ui(points: N)` literals --
    437 of the latter across 28 distinct values, the same role
    (secondary rows, empty-state symbols) written at five neighbouring
    sizes in different files. Standardized 2026-09-08: `AppFontStep` is
    the canonical scale (its doc table lists every role's rendered size
    at the 16pt default base), the `points:` spellings are DELETED so an
    off-scale size is a compile error, and the one numeric survivor,
    `themedFont(fitting:)`, is for layout-derived glyph sizes only (a
    letterform at `size * 0.45`), never a constant.
    `FontPropagationTests` checks `.font(` per CALL SITE rather than per
    file, rejects any raw size literal, and pins the scale table case by
    case. Reach for a role first; a size the scale lacks is a new
    `AppFontStep` case, which is one reviewed edit instead of a
    scattered literal. The AppKit print surfaces
    (`NSFont.systemFontSize`) are the documented exception
    (`swift/docs/SWIFT_SETTINGS_AUDIT.md` section 1), not an invitation.

## The `state#N` ledger

`AppModel` and its extensions carry `(state#N)` markers on the comments
that explain a fixed state-layer defect. The full index, 117 entries, is
`swift/docs/SWIFT_STATE_LEDGER.md` -- add the next number there when you
add a marker in source.

## Where feature-area detail lives

Read the page for the area you are touching; you do not need the others.
Each page carries its own "read this before" list at the top.

| Page | Covers |
|---|---|
| `swift/docs/SWIFT_TOOLS.md` | tool execution, containment, permissions, hooks, MCP, subagents, adding or removing a tool |
| `swift/docs/SWIFT_AGENT_MODE.md` | the `.agentAuto` permission mode: the classifier contract, routing rules, hard gates, fallback counters, hints |
| `swift/docs/SWIFT_PLUGINS.md` | the plugin system: manifest, contributions, enable cascade, marketplace |
| `swift/docs/SWIFT_SKILLS.md` | skills: architecture, scopes, file layout, marketplace |
| `swift/docs/SWIFT_MEMORY.md` | auto-memory: the per-project directory, the index, the `memory` tool, the `#` quick-save |
| `swift/docs/SWIFT_COMPACTION.md` | context compaction: trigger, boundary, summarizer, the ghost rule |
| `swift/docs/SWIFT_TURN_PIPELINE.md` | the message queue, steer delivery at step boundaries, system reminders |
| `swift/docs/SWIFT_GOALS.md` | the `/goal` loop: the stop-seam evaluator, deferral + idle check-ins, stall pause, restore rules |
| `swift/docs/SWIFT_MESSAGE_EDITING.md` | message retry, edit and branch |
| `swift/docs/SWIFT_GHOST_MODE.md` | temporary (ghost) chats: the two layers, the three rules that must not break |
| `swift/docs/SWIFT_STORAGE.md` | the three JSON stores, `AppStorageRoot`, which directories a test may write to |
| `swift/docs/SWIFT_CONTEXT_RING.md` | the composer's context-usage indicator; the system prompt as one builder, three consumers |
| `swift/docs/SWIFT_CHAT_SEARCH.md` | the Cmd+K search dialog: what is searched, ghost exclusion, matching semantics |
| `swift/docs/SWIFT_MODEL_HUB.md` | install, the model hub's badges and filters, `activeLoadGuard`, selecting/opening/unloading, server multi-model attach |
| `swift/docs/SWIFT_SESSION_CAPABILITIES.md` | reading capabilities off `session.info`: reasoning, tool calling, steering |
| `swift/docs/SWIFT_VISION.md` | image attachments: the wire shape, prompt ordering, the vision gate |
| `swift/docs/SWIFT_FAN_CONTROL.md` | the ThermalForge status-bar control and its quit-restore |
| `swift/docs/SWIFT_SETTINGS_AUDIT.md` | the settings audit: font propagation, per-field wiring, still-open items, the re-audit method |
| `swift/docs/SWIFT_LOCALIZATION.md` | the string catalog, its compile step, and the parity gates |
| `swift/docs/SWIFT_QWEN_PARITY2.md` | the second web-shell parity pass: git commands, bang commands, split panes, sidebar organization, chart fences, interactive AskUserQuestion, the QR descoping |
| `swift/docs/KEYBOARD_SHORTCUTS.md` | shortcuts, menu commands, VoiceOver, the unsloth-compatible alternate chords |
| `swift/docs/SWIFT_STATE_LEDGER.md` | the full `state#N` index |
