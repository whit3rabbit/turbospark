# swift/

Two SwiftPM packages over the C ABI in `crates/ffi`. `TurboSpark` is the
binding (async/await, `AsyncThrowingStream`, Codable wire types);
`TurboSparkApp` is a SwiftUI chat and model-management app built on it. Both
are macOS/Apple Silicon only and link the engine IN PROCESS: no HTTP, no IPC,
no server.

`README.md` beside this file is the user-facing quickstart (prerequisites,
API examples). This file is the working notes: what is where, what the build
actually does, and what has already gone wrong. Keep code, comments and docs
ASCII: no emojis and no em dashes (project rule).

Read `crates/ffi/CLAUDE.md` first when the change crosses the boundary. Its
Gotchas 1, 2, 7, 8, 9 and 10 are the Rust half of Gotchas 1, 2, 4, 5 and 6
here, and neither half makes sense alone. `docs/SWIFT_BINDINGS.md` documents
the ABI contract itself.

## Layout

```
swift/
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
    |   |                            # AppChat/AppProject/AppTool,
    |   |                            # GlobalMcpFileStore, SystemPermissionsManager
    |   +-- Generation/              # composer, output pane, chat sidebar, tool cards,
    |   |                            # ProjectSettingsSheet, ProjectMcpSettingsSheet
    |   +-- Installation/            # model hub, catalog sheet, probe, ETA cards
    |   +-- Diagnostics/             # inspector options, metric formatting, phase counters
    |   +-- Presentation/            # markdown render, docx/xlsx/pdf extract
    |   +-- Components/              # AppearanceSettingsPaneView, McpSettingsPaneView,
    |   |                            # McpServerEditorSheet, PermissionsSettingsPaneView,
    |   |                            # ToastOverlayView, ErrorBanner
    |   +-- Theme/                   # AppearanceSettings, TurboSparkTheme,
    |   |                            # AppChromePresentation, PointerCursorModifier,
    |   |                            # ThemeCodePreviewView
    |   +-- Tools/                   # AppToolCatalog, AppToolRegistry,
    |   |   +-- Core/                # AppToolPermissionEngine, OpenAIToolSchema
    |   |   +-- MCP/                 # McpClientEngine, McpServerSpec, ProjectMcpDetector
    |   |   +-- File/                # FileReadWriteTools
    |   |   \-- Web/                 # WebTools
    |   \-- Resources/               # app-prompts.json, Logos/ (Bundle.module)
    \-- Tests/TurboSparkAppTests/    # 44+ unit tests: AppearanceSettingsTests,
                                     # McpClientEngineTests, ProjectMcpDetectionTests,
                                     # ProjectRulesDetectionTests, SystemPermissionsTests,
                                     # ToolPermissionsTests
```

`AppModel` is split across `AppModel.swift` plus `AppModel+{Chat, Files,
Generation, Installation, Models, Persistence, Projects, Tools}.swift`.
Published state and derived properties live in the base file; every behaviour
is an extension. Add new behaviour as a new extension file rather than growing
the base one.

## Build, test, dev commands

Every target below is a root `Makefile` wrapper. Run them from the repository
root.

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

# Run the TurboSparkApp test suite (44+ unit tests covering appearance settings,
# MCP client engine, project MCP detection, rules detection, system
# permissions, and tool permissions).
cd swift/TurboSparkApp && swift test

# The app.
make swift-app-build        # debug
make swift-app-release      # release
make swift-app              # build and run
make swift-demo             # alias for swift-app (see Gotcha 10)

# Removes both .build trees AND the two staged files, so the next Swift
# build needs `make swift-lib` again.
make clean-swift
```

Iterating on SwiftUI only? Call SwiftPM directly and skip the staging step:

```bash
cd swift/TurboSparkApp && swift run TurboSparkApp
```

That is not a shortcut for convenience alone. `make swift-app` depends on
`swift-lib`, which `touch`es every `.swift` file in both packages (Gotcha 3),
so going through `make` recompiles the whole app every single time. Use
`make` when the Rust side moved and SwiftPM when it did not.

## Gotchas

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
   `Package.swift` carries `-LSources/CTurboSpark`, which is correct when its
   own tests link and wrong for anyone depending on it; `TurboSparkApp`
   carries `-L../TurboSpark/Sources/CTurboSpark`, the same directory seen
   from its own root. A third consumer needs its own spelling of that path.
   The observable is a `ld: warning: search path 'Sources/CTurboSpark' not
   found` on every app build, which is the library's own flag being resolved
   against the app's root and finding nothing; the link then succeeds on the
   app's own copy of it. Expected, not a regression. The `unsafeFlags` also
   block `TurboSpark` from being consumed as a versioned dependency by an
   out-of-repo package, which is accepted: the library it points at is a
   build artifact, not a checked-in file.

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

4. **THE INSTALL BYTE CALLBACK FIRES CONCURRENTLY FROM SEVERAL DOWNLOAD
   THREADS, SO PROGRESS CAN GO BACKWARDS.** `TurboSparkCatalog.install`
   documents taking the maximum rather than the last value;
   `AppModel+Installation.swift` does exactly that with a local `maxBytes`.
   A progress bar driven from the last event alone jitters backwards on a
   real install. The install also runs on a DEDICATED `Thread`, not a global
   queue slot: it blocks for tens of minutes, and parking a shared
   concurrent-queue worker that long starves the rest of the process. And the
   walk CANNOT RESUME, so a cancelled or failed install restarts from zero;
   the UI should say so before starting.

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

6. **`SessionInfo` HOLDS WHAT WAS RESOLVED, NOT WHAT WAS ASKED FOR, AND
   UNDER AUTOMATIC SIZING NOTHING WAS ASKED FOR.** Read `maxContext` and
   `expertCacheSlots` off `session.info` and never off the `OpenOptions` that
   produced it; no footprint or throughput figure is readable without the
   slot count (root Gotchas 36 and 58). Speculation follows the same rule:
   `info.speculation.block != nil` IS the "is it on" test, there is no second
   flag that could disagree, and `drafter` is non-nil exactly when `block`
   is. Non-nil is a statement about the SESSION and not about the next turn:
   acceptance is exact only at temperature 0, so a sampled turn decodes
   sequentially whatever it says.

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

9. **DO NOT APPEND `reasoning` TO CONVERSATION HISTORY.** Harmony drops
   prior-turn analysis and Qwen's template drops prior-turn `<think>` blocks,
   so feeding it back sends the model something it was never trained to read.
   `GenerationResult.content` is the assistant turn; `reasoning` is for
   display. `AppModel` stores it on `AppChatMessage.reasoning` for the
   collapsible view and rebuilds `[ChatMessage]` from `content` alone.
   Reasoning LEVELS are the checkpoint's set and not this library's: Qwen 3.8
   refuses `.high` and tops out at `.xhigh` while Harmony and Muse Glimmer
   accept `.high`, and a refused level throws from the template. Gate the UI
   on `info.reasoningSupport`, whose `.toggleOnly` case means the levels
   should be greyed rather than the toggle hidden.

10. **`TurboSparkDemo` IS GONE; `swift-demo` IS NOW AN ALIAS FOR
    `swift-app`.** What ships is `TurboSparkApp`: multi-chat with
    persistence, a project/agent system that executes tools, a model hub with
    catalog install and Hugging Face probing, document attachment, and a
    phase-counter inspector. The root `AGENTS.md` described a "deliberately
    minimal" demo that exists to verify the binding, which was true of the
    package this one replaced; corrected 2026-08-28, and worth knowing that
    any older note reading that way is describing a deleted target.

11. **THE APP IS UNSANDBOXED AND EXECUTES MODEL-PROPOSED SHELL COMMANDS.**
    `AppToolRegistry.execute` implements `run_command` alongside file read
    and write. The barrier is `AppModel.permission(for:)`, which resolves the
    project's `AppToolPermission` to `.ask`, `.allow` or `.deny` per
    category; `.allow` on the terminal category runs the command with no
    prompt, and a project's `maxAutonomousSteps` (default 5) bounds the agent
    loop. Read `State/AppTool.swift` and `State/AppProject.swift` together
    before changing anything on that path, and do not widen a default
    permission without saying so.

    **`resolveSecurePath` DID NOT CHECK CONTAINMENT UNTIL 2026-08-28, and its
    name said it did.** It standardized the caller's path and appended it to
    the root, which resolves `..` correctly and then follows it out: measured
    against a root of `/Users/me/proj`, `../../../../etc/passwd` came out as
    `/etc/passwd`. It now refuses absolute and `~` paths up front and
    compares the resolved target against the resolved root, symlinks included
    on both sides, so a link inside the project pointing outside it is
    refused too. Two things worth keeping. The check constrains the FILE
    tools only, because `run_command` takes no path and a shell leaves the
    root by its own means; the permission gate is the whole story there. And
    the root has to be resolved as well as the target, or a project under a
    symlinked path (`/tmp` is one on macOS) fails its own containment test.

12. **`swift run` PRODUCES A BARE EXECUTABLE, NOT AN `.app` BUNDLE, AND THE
    SHIPPED APP IS ASSEMBLED BY A SCRIPT.** `ForegroundAppDelegate` calls
    `NSApp.setActivationPolicy(.regular)` at launch precisely so a
    command-line-launched binary gets a Dock icon and a menu bar. Resources
    still work under `swift run` (`Bundle.module` is SwiftPM's generated
    bundle, which is how `Logos/` and `app-prompts.json` are found).

    **`scripts/make-app-bundle.sh` is the only place a bundle identity
    exists**: it writes the tree's one `Info.plist`, sets
    `CFBundleIdentifier` to `com.whit3rabbit.turbospark`, copies every
    `*.bundle` from `.build/release` into `Contents/Resources`, drops the
    three CLI binaries into `Contents/MacOS`, and ad-hoc signs the result.
    `scripts/make-dmg.sh` wraps that for release. `docs/RELEASE.md` is the
    home for both.

    **THE IDENTIFIER MOVES THE PREFERENCES DOMAIN, and that is invisible
    until someone compares two installs.** A `swift run` build writes
    `~/Library/Preferences/TurboSparkApp.plist`; the bundle writes
    `com.whit3rabbit.turbospark.plist` (measured by launching it, not
    inferred from the docs). So every `@AppStorage` key -- appearance, text
    size, language -- reads as unset the first time a developer opens the
    installed app, while their `swift run` settings sit intact in the other
    file. Gotcha 13's three JSON stores are NOT affected: those paths are
    hardcoded `"TurboSpark"` under Application Support and have no bundle
    input at all. Anything needing a REAL identity beyond a name -- Developer
    ID signing, notarization, Keychain, sandbox entitlements -- still does
    not exist; the signing seam is `CODESIGN_IDENTITY`, which defaults to
    ad-hoc.

    **`LSMinimumSystemVersion` IS THE THIRD COPY OF A NUMBER.** The script
    writes `14.0`, `TurboSparkApp/Package.swift` declares
    `platforms: [.macOS(.v14)]`, and both Homebrew casks say
    `depends_on macos: ">= :sonoma"`. Change one and change all three;
    Gotcha 14 is the same hazard on the deployment-target axis.

13. **ALL APP STATE LIVES IN THREE JSON FILES UNDER
    `~/Library/Application Support/TurboSpark/`** (`settings.json`,
    `chats_archive.json`, `projects_archive.json`), each written whole with
    `.atomic` on every mutation. Two consequences. Every store's `load()`
    swallows a decode failure and returns the empty default, so a schema
    change that is not backwards-compatible silently discards the user's
    chats rather than erroring: add fields with `decodeIfPresent` and a
    default, the way `MacAppSettings`'s hand-written `init(from:)` already
    does.

    **THAT HAZARD WAS NOT HYPOTHETICAL AND HAD ALREADY FIRED** (found
    2026-08-29 on the archive on this machine). `AppChatMessage` gained
    `toolCalls` and `toolResults` as non-optional arrays on the synthesized
    decoder, so ONE message written before those fields existed threw
    `keyNotFound`, `load()` swallowed it, and the app came up with an empty
    chat list while a four-message conversation sat intact on disk -- which
    the next `persistChats()` would have overwritten for real. Note the shape
    of the symptom, because it is what makes this class expensive: nothing
    errors, nothing is logged, and the user reads it as "the app lost my
    chats" rather than as a decode failure. `AppChatMessage` and `AppChat`
    now carry hand-written tolerant `init(from:)`s and `load()` REPORTS the
    error to stderr before falling back. Every field added to either struct
    from now on gets `decodeIfPresent` and a default.

    And `persistChats()` re-encodes the entire archive on every
    keystroke of the draft, since `promptText`'s setter calls it; that is
    fine at current sizes and is the first thing to look at if typing ever
    feels heavy.

14. **THE TWO PACKAGES DECLARE DIFFERENT DEPLOYMENT TARGETS AND
    `scripts/swift-lib.sh` PINS A THIRD NUMBER TO ONE OF THEM.** The script
    exports `MACOSX_DEPLOYMENT_TARGET=13.0` and its comment says that must
    match "both Package.swift files"; `TurboSpark` is `.macOS(.v13)` and
    `TurboSparkApp` is `.macOS(.v14)`. The failure the script guards against
    is objects built for a NEWER SDK than the link target, so an archive at
    13.0 linked into a v14 app is the safe direction and the comment is what
    is stale. It stops being safe the moment the archive's number is raised
    past a consumer's, so change the script and both manifests together.

15. **THE 400-LINE GUIDELINE IS ALREADY BROKEN ON THE VIEW SIDE.**
    `ChatSidebarView` (476, down from 522 once the rail took its section
    list), `ModelDetailPaneView` (432), `ModelHubView` (407) and
    `InspectorOptionsSection` (406) are over it. Split by
    subview when touching one of them rather than adding to it. Nothing in
    `TurboSpark/` is anywhere near the limit and should stay that way: the
    binding is thin on purpose, and logic that creeps into it is logic no
    Rust test can reach.

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
    `RootView` is a rail (sections), a top bar (which model, and is it
    loaded), the working panes, and a status strip (memory, context fill,
    tok/s). Two rules fall out of that split and both were arrived at by
    looking at the running app. Sections live in the RAIL so the chat
    sidebar can be hidden without stranding navigation, which is why
    `showsChatSidebar` is gated on `activeSection == .chat`. And the top bar
    is LEFT-ALIGNED rather than centred: the phase indicator appears and
    disappears once per turn, and a centred model loader slides sideways
    every time it does.

    The right column is one slot, not two: `previewAttachment != nil` takes
    it from the inspector. That is why `.toggleInspector` closes the preview
    first -- a shortcut that toggles a pane hidden behind another one reads
    as a broken shortcut.

    Traffic lights are the layout constraint nobody remembers. The window is
    `.hiddenTitleBar`, so the system still draws them over the content at
    roughly x = 13 to 66; `AppChromeLayout.trafficLightClearance` is what
    keeps the top bar's first control clear of them, and the rail starts
    below `topBarHeight` for the same reason.

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
    `~/Downloads`, and `~/Desktop` is governed by macOS Transparency, Consent,
    and Control (TCC). `SystemPermissionsManager` probes directory readability
    via `contentsOfDirectory(atPath:)` and triggers interactive `NSOpenPanel`
    approval or deep-links directly to macOS System Settings
    (`Privacy_FilesAndFolders`, `Privacy_AllFiles`). Users can also authorize
    custom project directories in the Files & Permissions settings tab.

20. **MCP SERVERS AND STREAMS (STDIO AND SSE).** `McpClientEngine` handles
    Model Context Protocol JSON-RPC 2.0 initialization and tool calls for
    both stdio subprocesses and HTTP/SSE endpoints. Environment variables in
    config (`${HOME}`, `${workspaceFolder}`) expand before spawning, and
    server-level `autoApprove` flags bypass interactive approval prompts unless
    an operation is classified as high-risk.

21. **DYNAMIC THEME AND ACCENT INJECTIONS.** `AppearanceManager` controls
    dark/light mode, custom theme presets, and accent colors. Custom accent
    colors inject dynamically into controls via `activeAccentColor(isDark:)`
    and `TurboSparkTheme`.

22. **A BADGE OR A FILTER THAT CANNOT FAIL CARRIES NO INFORMATION, AND THE
    MODEL HUB SHIPPED THREE OF THEM.** All three read as working features and
    all three were found by writing the first unit test over the code rather
    than by looking at it (`ModelHubFilterTests`).

    `ModelCardView` drew an unconditional `checkmark.seal.fill` captioned
    "Verified model in TurboSpark catalog" on EVERY row, while `models.json`
    marks rows `verified`, `runs` or `caveat` and the view never read the
    field. `ModelHubView`'s capability filter matched alias SUBSTRINGS, so
    `.conversational` fell through to `break` and returned the whole catalog,
    `museGlimmer` was listed under both "Reasoning" and "Dense", and the fit
    filter offered "Too large" on machines where no row is refused -- a
    choice that can only ever return an empty list.

    **THE FORMAT LABEL WAS THE EXPENSIVE ONE, because it is the field a user
    picks a row by.** `ModelFamilyVisuals.resolve` stated `formatLabel` by
    hand in each of its 16 family branches, and a family and a quantization
    are INDEPENDENT: `mistral7b` and `tinyllama` are GGUF rows that both
    rendered "MLX INT4", `ornith9b` is GGUF Q8_0 rendered as MLX because its
    alias has no "gguf" in it, `gemma4-gguf` is Q8_0 rendered as "Q4_K_M",
    and `bonsai27b` is MLX affine 1-bit rendered as INT4. Wrong on a majority
    of the shipped rows, and wrong in the direction that matters.

    The fix is one derivation instead of sixteen assertions:
    `ModelFamilyVisuals.formatLabel(alias:name:)` reads the parenthesised
    suffix of the catalog row's own `name` up to the first comma, which IS
    where the catalog states the format. Nothing to keep in sync, and adding
    a row needs no code change. `ModelHubFilter` then builds its dropdown
    options from that same call, so the badge and the filter cannot disagree.

    Two rules out of it. **Derive a display value from the data or read it
    off the row; never restate it per branch** -- the restatement is correct
    on the day it is written and silently wrong at the next checkpoint. And
    **build a filter's options from what is present**, not from the full enum,
    or the UI offers choices that match nothing.

23. **A ZERO FROM AN ABSENT MEASUREMENT IS NOT A MEASUREMENT OF ZERO, AND THE
    MODEL DETAIL PANE PRESENTED THREE OF THEM AS FACTS.** A `.unknown` fit
    verdict means the sizing could not be determined, and the numeric fields
    that arrive with it are zeros. `ModelDetailPaneView` rendered them through
    the same grid as a real reading, so an unsized row advertised "Unified
    Memory: Zero KB" and "Max Context: 0 tokens" beside a summary that said
    "unknown (probe it)".

    **The slot count is the dangerous one and it does not even look wrong.**
    With no architecture read there is no expert stride, `Auto` divides by
    nothing and returns `DEFAULT_CACHE_SLOTS`, so the pane showed "Expert
    Slots: 16 slots" -- an answer arrived at BY IGNORANCE that is
    indistinguishable from a measured 16 (root `CLAUDE.md` Gotcha 58). The
    card now branches on `fitIsKnown`, every cell is guarded on being
    non-zero, and the unsized state shows only what the CATALOG states plus
    the command that would produce the rest.

    There is deliberately no Probe button on that card. `ts_recommend_json` is
    offline-only and this binding exposes no probing call, so the button would
    have to lie about what it does; the pane names
    `turbospark-model recommend --probe` instead. **Check that the work exists
    before adding the control that claims to do it.**

    A smaller sibling in the same pane: `TextField`'s title on macOS is a
    VISIBLE LABEL, not a placeholder. `InspectorOptionsSection`'s rate-cap
    field passed "Uncapped" as that title without `.labelsHidden()`, so it was
    drawn beside the field and clipped to "Un-" by the inspector's width,
    while every neighbouring Picker in the same section already hid its label.
    A stray truncated word next to a control is worth reading as a missing
    `.labelsHidden()` before it is read as a layout problem.
