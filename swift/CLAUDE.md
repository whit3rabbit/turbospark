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
    |   |                            # McpServerEditorSheet, PermissionsSettingsPaneView,
    |   |                            # ToastOverlayView, ErrorBanner
    |   +-- Theme/                   # AppearanceSettings, AppearanceTypes, AppDockIconRenderer,
    |   |                            # TurboSparkTheme, AppChromePresentation,
    |   |                            # PointerCursorModifier, ThemeCodePreviewView
    |   +-- Tools/                   # AppToolCatalog, AppToolRegistry,
    |   |   +-- Core/                # AppToolPermissionEngine, OpenAIToolSchema
    |   |   +-- MCP/                 # McpClientEngine, McpServerSpec, ProjectMcpDetector
    |   |   +-- File/                # FileReadWriteTools
    |   |   \-- Web/                 # WebTools
    |   \-- Resources/               # app-prompts.json, Logos/ (Bundle.module)
    \-- Tests/TurboSparkAppTests/    # Unit tests covering appearance settings,
                                     # MCP client engine, project MCP detection,
                                     # project rules detection, system permissions,
                                     # and tool permissions.
```

`AppModel` is split across `AppModel.swift` plus `AppModel+{Chat, Files,
Generation, Hooks, Installation, Mcp, Models, Persistence, Projects, Server,
Tools, Transcript}.swift`. Published state and core lifecycle live in the base file;
every functional domain is an extension. Add new behaviour as a new extension
file rather than growing the base one.

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
   **BUILD THE LEVEL PICKER FROM `info.reasoningEfforts`, NOT FROM
   `Reasoning.allCases` AND NOT FROM THE FAMILY.** The accepted set belongs to
   the checkpoint's own chat template: Qwen 3.8 refuses `.high` and tops out
   at `.xhigh` while gpt-oss and Muse Glimmer accept `.high` and have no
   `.xhigh`, and a refused level throws from the template mid-turn. The engine
   probes that set at open (five renders of a two-line conversation) and
   reports it; `reasoningSupport` says only what KIND of control is
   meaningful. `AppModel` offered `allCases` for every `.level` checkpoint
   until 2026-08-31, so the menu carried an entry that failed the turn.

   **THE THREE THINGS THAT COST SOMETHING TO REDISCOVER.** Levels rendering
   the same prompt are collapsed by the engine, so a `.toggleOnly` checkpoint
   reports exactly two and its on-level is `.low` BY POSITION -- print "On",
   never the spelling (`ReasoningLevelPolicy.label(for:support:)`). There is
   no answer at all before a session exists, and the guess that used to fill
   the gap was a hardcoded family set, i.e. exactly the table root Gotcha 56
   refuses; the controls gate on `AppModel.reasoningPickerEnabled` and the
   inspector disables rather than hides, since a missing row reads as a
   missing feature. And a preference restored from another model is clamped
   to the nearest expressible rung WITH A TOAST, because dropping it to
   `.off` silently turns thinking off for someone who turned it on -- the
   silent no-op the whole feature exists to avoid.

   The decision lives in `ReasoningLevelPolicy`, a pure type, for Gotcha 26's
   reason: `AppModel.info` is computed from `session`, so assertions written
   against `AppModel` need a real install and therefore never ran.

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
    loop. **That description was true of the MECHANISM and false of the
    CONFIGURATION every real project ran under until 2026-08-30: the sheet
    that creates them seeded `terminal: .allow`. Read Gotchas 28 and 29 with
    this one.** Read `State/AppTool.swift` and `State/AppProject.swift` together
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

    **AND A `swift run` BINARY CANNOT BE DRIVEN BY UI AUTOMATION.** It has no
    bundle identifier, so macOS app-allowlist APIs (computer-use
    `request_access`, and anything else keyed on bundle ID) cannot match it
    even while it is running and frontmost as `TurboSparkApp`. Visual
    verification needs `scripts/make-app-bundle.sh` first, which is also the
    only build where the nested resource bundle is exercised.

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

15. **THE 400-LINE GUIDELINE IS ACTIVELY ENFORCED VIA SUBVIEW MODULARITY.**
    Large views (`ChatSidebarView`, `ModelDetailPaneView`, `ModelHubView`,
    `AppearanceSettingsPaneView`, `ProjectSettingsSheet`,
    `ModelsSettingsPaneView`) and core classes (`AppModel`, `AppearanceSettings`)
    are decomposed into dedicated subviews, domain extensions, and type files.
    Split by subview and functional extension when touching or extending them
    rather than growing a single view or model file. Nothing in `TurboSpark/`
    is near the limit: the binding is thin on purpose, and logic that creeps
    into it is logic no Rust test can reach.

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

    **THE RAIL CARRIES FIVE SECTIONS SINCE 2026-08-30** (Chat, Files,
    Installed, Discover, Server). Server was APPENDED at the last shortcut
    rather than inserted, so the four that existed keep the numbers anyone
    has already learned. A new case needs no view change --
    `railButton` builds its hover tooltip from `title` and `shortcutKey` for
    everything in `allCases` -- which is exactly why
    `testEverySectionHasAUniqueTitleAndShortcut` exists: a missing or
    duplicated one is a blank or wrong tooltip that only a screenshot would
    catch. The Server pane is the one section that keeps working while it is
    off screen, because its poll timer follows the SERVER rather than the
    view: something else (unloading a model from Chat) can stop one.

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

24. **TWO UNRELATED THINGS IN THIS APP ARE CALLED "GUARDRAILS".**
    `AppGuardrailsMode` in `State/MacAppSettings.swift` is FORGE TOOL-CALL
    guardrails: whether a model's tool calls get dialect rescue and schema
    validation. `AppLoadGuardOption` in `State/AppRuntimeOptions.swift` is
    MEMORY guardrails: how much of the machine a model may commit when it
    loads (`docs/LOAD_GUARD.md`). They share no code, no settings key and no
    UI surface, and the second is deliberately NOT named `AppGuardrailsMode`
    -- the collision was caught before the type existed rather than after.

    Two consequences. `MacAppSettings` carries BOTH `guardrailsMode` and
    `loadGuard` as separate persisted keys, so a reader grepping for "the
    guardrails setting" finds the wrong one half the time. And the memory
    pane's section is titled "Model loading guardrails" while the tool one is
    "Forge Guardrails", which is the only thing separating them for a user.

25. **`AppModel.activeLoadGuard` EXISTS SO THE HUB AND THE LOADER CANNOT
    RESOLVE DIFFERENT TIERS.** `TurboSparkCatalog.recommend` and
    `ts_session_open` share one memory budget by construction, which is what
    makes a hub verdict worth showing; a hub ranking under `.relaxed` while
    sessions open under `.strict` promises a fit the loader then refuses, in
    the one place a user cannot see the two disagree. All three `recommend`
    call sites (`ModelHubView`, `CatalogSheet`, `ModelInstallView`) and
    `buildOpenOptions` read that one accessor. A fourth caller building its
    own from `runtimeOptions` would compile and be wrong only when the user
    moves the setting off the default.

26. **`AppModel.server` OUTLIVES `AppModel.session` UNLESS SOMETHING STOPS IT
    FIRST, AND THAT SOMETHING IS EVERY CALLER THAT CLEARS `session`.** Added
    2026-08-30 (`TurboSparkServer`, `AppModel+Server.swift`). A
    `TurboSparkServer` holds its own reference to the engine on the Rust
    side (`crates/ffi/CLAUDE.md` Gotcha 13's whole design), so `session = nil`
    alone does not stop a server started against it -- the model stays
    resident and the server keeps answering requests for a model the UI no
    longer shows as loaded. `open(_:)`, `unloadModel()` and
    `setModelURL(_:)` all call `stopServer()` before clearing `session` for
    exactly this reason; a fourth call site that clears `session` directly
    (rather than through one of those three) would compile and leak the
    old model for as long as the server keeps running. `TurboSparkServer`
    itself guards the OTHER direction: `ts_server_stop` frees its C handle,
    so `stop()` and `deinit` both route through one `NSLock`-guarded
    idempotent path rather than each calling the C function directly, which
    would double-free if a caller stopped it explicitly and then let it go
    out of scope.

    **NOTHING IN THE UI STATES AN ADDRESS, A PORT OR AN AUTH STATE OF ITS
    OWN; ALL THREE ARE READ BACK OFF `info()`.** Audited 2026-08-30, and all
    three arms of that sentence were false before it. The toast and the
    Address row spelled `127.0.0.1` by hand beside a read-back port -- true,
    because `crates/ffi`'s `Server::start` binds that literal, and an
    assertion rather than a reading in exactly Gotcha 22's shape (a hub badge
    restating what the row already stated). `ServerInfo` carries `host` now,
    with a `baseURL` helper, and `AppModel.serverInfo` is the ONE accessor
    the view reads: `info()` takes a lock, crosses the ABI and decodes JSON,
    and a SwiftUI body runs far more often than a server changes.

    **BOTH ROWS ARE A VALUE (`ServerStatusRows`) RATHER THAN INLINE VIEW
    CODE, WHICH IS THE ONLY REASON THEY CAN BE TESTED AT ALL.**
    `ServerStatusRowsTests` needs no model, no session and no bound socket,
    and every `ServerInfo` in it is DECODED from the JSON
    `ts_server_info_json` emits, so the cases pin the wire spelling too.
    **The case that carries the file is `testAddressFollowsAHostThatIsNotLoopback`**,
    and its value was measured rather than assumed: hardcoding the address
    back to `"http://127.0.0.1:\(port)"` reddens it and the IPv6 case while
    leaving `testAddressIsBuiltFromTheReportedHostAndPort` GREEN. A test
    written at the address this engine happens to bind cannot tell a reading
    from a literal, which is the whole defect -- so a file of only-loopback
    cases would have proved nothing. `AppModel.serverAPIKey(from:)` is split
    out for the same reason and is `nonisolated`, being a pure function of
    its argument.

    **THE AUTH STATE IS THE HALF THAT WAS INVISIBLE RATHER THAN MERELY
    UNVERIFIED.** `startServer` trims the key field and maps empty to `nil`,
    so a key of nothing but spaces starts an UNAUTHENTICATED server -- and
    the panel rendered identically either way, with `info.authEnabled`
    decoded and shown nowhere. There is an Auth row now, and the toast names
    it.

    **AND `ServerOptions`'s DOC WAS WRONG ABOUT THE SECURITY PROPERTY, NOT
    JUST UNVERIFIED.** It called an unauthenticated default "appropriate for
    a server bound to loopback and reachable only from inside this process".
    A loopback TCP socket is reachable by every process on the machine; that
    is not a property a socket can have. It is the sentence a reader would
    have used to decide against setting a key, which is what makes it the
    most expensive of the three. The settings caption says the same thing
    plainly now.

    **SINCE 2026-08-30 A SERVER SERVES SEVERAL MODELS, AND `stopServer()` IS
    THE WRONG TOOL FOR UNLOADING ONE.** The rule above -- every site that
    clears `session` must stop the server first -- was right for one model
    and is now too big a hammer: stopping the server to swap the chat model
    would take every OTHER attached model down with it. `open(_:)`,
    `unloadModel()` and `setModelURL(_:)` call
    `detachChatSessionFromServer()` instead, which removes whatever entry
    that session was attached under and leaves the rest serving. A fourth
    site that clears `session` without coming through there keeps that model
    resident, served, and invisible in the Chat pane -- the same failure,
    now per model.

    **TWO REFERENCES HOLD AN ATTACHED MODEL AND BOTH HAVE TO GO.** The
    server holds one on the Rust side and `AppModel.serverAttachedSessions`
    holds the Swift one; dropping either alone keeps the weights, the KV
    cache and the compiled pipelines resident. `detachModelFromServer(id:)`
    does both. The CHAT session is the deliberate exception and needs no
    branch: `AppModel.session` is a third reference the Chat pane still
    holds, so ejecting it from the Server pane stops it being SERVED and
    leaves it loaded.

    **NO VIEW BODY CALLS INTO THE BINDING.** `AppModel.serverInfo` is a
    published SNAPSHOT rather than the computed accessor it started as:
    `info()` takes a lock, crosses the ABI and decodes JSON, and the Server
    pane re-evaluates far more often than a server changes.
    `refreshServerInfo()` is the only writer, called on start, attach,
    detach and once per poll tick for the uptime. `poll` is likewise driven
    by the timer alone, at 2 Hz -- a rate chosen so the console feels live,
    not a limit: the engine's ring holds ~2,000 events, which is hundreds of
    requests, so nothing is lost at any rate a person would pick. When
    something IS lost the engine says so and
    `ServerMetricsStore.droppedEvents` sums it, because a console missing
    rows reads exactly like a server that was idle.

    **THE PANE'S ARITHMETIC IS IN TWO PURE TYPES SO IT CAN BE TESTED.**
    `ServerMetricsStore` folds the four events that describe a request back
    into one record and derives the series; `ServerEndpointCatalog` holds
    the route list and the connect snippets. Neither touches SwiftUI or a
    socket. That is what caught a real bug: `trim()` ran only on the batch
    path, so the public single-event `ingest` grew the window without limit
    -- and the same gap had made
    `testAnInFlightRequestSurvivesTrimmingAndItsLateEventsStillLandOnIt`
    VACUOUS, since no trimming ever happened for it to survive.

    Three things the pane refuses to state, each already paid for elsewhere
    in this file. A model with no session here shows "attached" rather than
    a zero context and a plausible-looking 16 slots arrived at by ignorance
    (Gotcha 23). The endpoint list carries no `/v1/embeddings` row, because
    this engine has no embedding path and a listed route that 404s is a
    capability claim a reader could build against -- pinned by
    `testNoEndpointIsAdvertisedThatTheEngineCannotServe`. And there is no
    time-to-first-token chart: nothing inside a generation can measure one
    (`crates/server/CLAUDE.md` Gotcha 29), so the pane shows prefill, decode
    and the queue, which is the subtraction that answers the question
    anyway.

27. **SWIFT COMPILER WARNINGS: NON-THROWING `URL` PATHS AND `onChange` ARITY.**
    Two recurring Swift warning patterns to avoid:
    - `URL(fileURLWithPath:).standardizedFileURL.path` is non-throwing. Wrapping
      it in `try?` produces a compiler warning ("no calls to throwing functions
      occur within 'try' expression"). Standardize file paths directly without
      `try?`.
    - `.onChange(of:perform:)` with a single-parameter closure `(V) -> Void` is
      deprecated on macOS 14.0+. Use the modern 0-parameter form
      `.onChange(of: value) { ... }` or the 2-parameter form
      `.onChange(of: value) { oldValue, newValue in ... }` (or `{ _, newValue in }`).

28. **A DEFAULT SPELLED AT FOUR SITES DISAGREED AT THE ONE USERS REACH, AND
    THE PERMISSIVE SPELLING WAS THE SHIPPED ONE.** `AppProject.init`, its
    decode fallback and `AppModel.createProject` all defaulted to
    `AppProjectPermissions.standard` (`terminal: .ask`). `ProjectSettingsSheet`'s
    new-project branch seeded `.auto`, which is `terminal: .allow`,
    `fileWrite: .allow`, `mcp: .allow`. A project can only be created through
    that sheet, so every project any user ever made ran model-proposed shell
    commands with no prompt, while the conservative default three other sites
    agreed on was decoration. All four now read
    `AppProjectPermissions.newProjectDefault`; a fifth site spelling its own
    is what that constant exists to make visible. Gotcha 11 says the barrier
    is `AppModel.permission(for:)`, and that was true of the mechanism and
    false of the configuration it ran under.

29. **A DENYLIST OVER A STRING BOUND FOR `/bin/zsh -c` IS THE WRONG SHAPE,
    NOT AN INCOMPLETE LIST.** `ToolRiskClassifier` matched ~20 regexes against
    raw command text, and under `.auto` `AppToolPermissionEngine.evaluate`
    returns `.allow` for anything not `.high` -- so `.safe` and `.low` are one
    decision there and the classifier's real output is binary: does this run
    unwatched. `rm -rf ~/Documents` matched. `r""m -rf ~/Documents` did not,
    and zsh runs them identically. Nor did `eval $(printf ...)`,
    `$'\x72m' -rf ~`, `CMD=rm; $CMD -rf ~`, or `` `echo rm` -rf ~ ``. **18 of
    23 corpus strings in `TerminalRiskGateTests` scored `.low` or `.safe`
    against the old code**, each one an edit away from a pattern that WAS
    caught.

    The gate is now positive: `TerminalCommandClassifier.isAutoApprovable`
    runs a command only when it is a single simple invocation (no
    `| ; & $ \` < > ( ) { }` anywhere, no quote/backslash/`=` in the head
    word) AND its program is on a read/build allowlist. The denylist stays,
    because a matched pattern names a specific reason for the approval sheet
    and the allowlist's generic one is worse to show a user.

    Two traps found by tuning it. Rejecting quotes ANYWHERE fails
    `git commit -m 'msg'`, `grep -rn 'struct' src/` and `find . -name '*.swift'` --
    quotes hide a head word and mean nothing in an argument, so the check is
    per position. And `python3` is on the allowlist because `.auto` promises
    to run `python3 -m pytest`, which makes the inline-code-flag rejection
    (`-c`, `-e`, `--eval`, `--command`, scoped to interpreters so `grep -e`
    still works) the ONLY thing keeping that entry safe.

    `TerminalCommandClassifier.isCollapsible` reads the FIRST WORD only and is
    presentation, never a gate: it scored `cat README && python3 -c '...'` as
    `.safe`. It and `isAutoApprovable` are kept apart so a display tweak
    cannot widen the gate again.

30. **A PROJECTLESS CHAT HAS NO WORKSPACE, AND THERE IS NO DEFENSIBLE
    DEFAULT.** `AppToolRegistry.execute` rooted a chat with no project at
    `FileManager.default.homeDirectoryForCurrentUser`, which is narrower than
    the `currentDirectoryPath` of `/` it replaced in the way that counts
    least: `resolveSecurePath`'s containment check passes for
    `~/Library/Application Support`, browser profiles, shell history and every
    token on disk, and `isSensitivePath` knows about a dozen filenames out of
    all of that. Path-taking and process-spawning tools are refused by name
    now (`workspaceRootedToolNames`); `skill`, the task tools and
    `askuserquestion` need no root and still work, which is the case the
    fallback was really reaching for.

31. **A CLASS INITIALIZER THAT THROWS PART-WAY DOES NOT RUN `deinit`, SO A C
    HANDLE ACQUIRED BEFORE THE THROW LEAKS IN FULL.** `TurboSparkSession.init`
    assigned `handle` and then read `ts_session_info_json` into `info`; a
    failure there left an instance that was never fully initialized, Swift
    skipped `deinit`, and `ts_session_close` never ran -- stranding mapped
    weights, KV cache and compiled Metal pipelines for the life of the
    process, silently, with the caller seeing exactly the error it expected.
    The info read now happens before any stored property is assigned and
    closes the handle by hand on the failure path.
    `SurfaceTests.testAThrowingInitDoesNotRunDeinitSoCLeanupMustBeManual`
    pins the language rule rather than the initializer, because making the
    real one throw at that point needs a fault injection the C ABI does not
    offer.

    **AND `generate` RETAINED NOTHING FOR THE TURN.** Its worker was
    `queue.async { [handle] in ... }` -- the capture list named `handle`
    alone, `onTermination` is `[weak self]` by design, and the returned stream
    holds no reference back, so no strong reference to the session existed
    anywhere during a generation while `deinit`'s own comment asserted the
    stream's task provided one. A caller releasing its session mid-turn ran
    `ts_session_close` beside `ts_generate`, which `turbospark.h` forbids. The
    capture is `[self]` now and that is load-bearing; the app never hit it
    only because `executeGenerationTurn` binds `session` strongly.

32. **AN UNIMPLEMENTED TRANSPORT MUST THROW, NOT RETURN A SUCCESS STRING.**
    `McpClientEngine.callToolViaSSE` returned the literal
    `"SSE remote tool execution completed."` for every call without issuing a
    request, and `discoverToolsViaSSE` built a `URLRequest`, never sent it,
    and returned `[]` -- which reads as "this server publishes no tools". So
    an SSE server config reported every tool call as having succeeded: the
    model was told an external action happened and the transcript showed a
    green result. Same class as the fabricated "Executed successfully" that
    `AppToolRegistry.execute`'s default case was fixed for (T5), one layer
    over. Both arms throw and name the transport now.

    Two smaller things in the same file. MCP stdio children were seeded from
    `ProcessInfo.processInfo.environment`, handing a third-party server binary
    every credential the app was launched with; they get `PATH`, `HOME`,
    `LANG`, `TMPDIR` plus the config's own `env` now. And
    `resolveExecutablePath` returned `/usr/bin/env` for anything it could not
    find, moving the lookup to spawn time where nothing could observe or
    report it -- it searches `PATH` itself and returns nil, so an unresolvable
    command is an error naming itself.

33. **IMAGES TRAVEL AS ORDERED CONTENT PARTS, AND THE ONE THING THAT KEEPS
    THAT FROM BEING AN ABI BREAK IS THAT AN EMPTY `images` ENCODES AS A BARE
    STRING.** Added 2026-08-30. `ChatMessage` kept `content: String` and
    gained `images: [ChatImage]` beside it rather than turning `content` into
    an enum: that field is read in dozens of places with nothing to do with
    vision, and a message carrying no picture still encodes
    `"content": "hello"` byte for byte, which is what every caller predating
    this sends. Only a message with an image switches to the parts array.
    `testAMessageWithoutImagesEncodesContentAsABareString` is the guard.

    **IMAGES ARE PREPENDED TO THE TEXT, MATCHED TO THE REFERENCE RATHER THAN
    CHOSEN.** `apply_chat_template(processor, config, question,
    num_images=1)` builds `[image, text]`. Appending moves every mRoPE
    position past the image and produces a different prompt for the same
    request -- fluently, with no error.

    **GATE THE ATTACH CONTROL ON `info.vision.active`, WHICH IS NOT "DOES
    THIS FAMILY HAVE A TOWER".** An install can carry one and refuse every
    image: the pixel budget comes from the checkpoint's own
    `preprocessor_config.json` and has no default worth falling back to
    (`crates/vision-io` Gotcha 6), so an install streamed without that
    sidecar reports `active == false` with the reason. `AppModel`'s
    `visionIsActive` / `attachmentContentTypes` are ONE accessor pair for
    `activeLoadGuard`'s reason (Gotcha 25): two views assembling their own
    picker list would drift, and the one that drifted would accept a file the
    turn then refuses.

    **THREE PLACES A PICTURE IS LOST RATHER THAN ERRORED, ALL FOUND BY
    WRITING THE TESTS.** Each reads as the model ignoring the image.
    - `AppChatMessage.imagePaths` had to be on the MESSAGE, not just the
      draft: the prompt is rebuilt from the transcript on every agent step,
      so a picture held only in the composer is sent on step one and
      silently dropped on step two. It decodes with `decodeIfPresent` and a
      default, for Gotcha 13's reason.
    - `executeGenerationTurn`'s `guard !msg.content.isEmpty` predates images
      and drops an image-only turn whole. An image-only turn has no text and
      is still a turn.
    - `AttachmentImporter` sent every file through `DocumentTextExtractor`,
      which throws `unsupportedFormat` on every image type. Correct for its
      own job, and why images could not be attached at all: the pixels go to
      the tower and the prompt needs only the path.

    **AND ONE PLACE A ZERO WAS PRESENTED AS A FACT.** An image extracts no
    text, so the chip's "0 chars" was Gotcha 23's shape and read as a failed
    import. `AppPromptAttachment.detailText` is a VALUE rather than inline
    view code so the branch can be tested at all, which is Gotcha 26's
    `ServerStatusRows` lesson applied a second time.

    The live limit worth knowing: `updateTokenEstimate` is a FLOOR on a turn
    carrying images. `countTokens` renders the template, which emits one
    marker per image whatever its size, and the expansion to that page's
    merged-token count happens later in the engine's splice.

34. **A SWIFT BUILD IN A WORKTREE FAILS ON A MISSING HEADER, BECAUSE
    `CTurboSpark` IS GITIGNORED.** Gitignored files are not carried into a
    worktree (root Gotcha 13). The staged `libturbospark_ffi.a` and
    `turbospark.h` are exactly that. Either run `make swift-lib` in the
    worktree, or copy the directory across:
    `cp -R swift/TurboSpark/Sources/CTurboSpark <worktree>/swift/TurboSpark/Sources/`.

    **AND `Bundle.module` INSIDE A TEST TARGET IS NOT THE APP'S BUNDLE.**
    `TurboSparkAppTests` declares no resources of its own. `Bundle.module`
    there resolves to the test bundle and finds nothing. Reach an app resource
    through an accessor in the APP target instead. `CommandGate.oracleFixtureURL`
    is the pattern, and it exists for this reason.

35. **THE AGENT LOOP HAS TWO PROMPT SHAPES NOW, AND THE PATCH MUST MERGE
    BEFORE TOOL CALLS ARE PARSED.** `executeGenerationTurn` branches once, at
    the context assembly: the append-only walk over every message and every
    tool result, or `buildSkillStateHistory`, which sends
    [system + protocol][the task][current state][newest observation] and is
    O(1) in step count. The branch is `selectedProject?.skillStateEnabled`,
    default FALSE, and the append-only arm is untouched when it is off. Why it
    exists at all is measured rather than argued: on a 4,096-context install
    the append-only arm stops between step 30 and 35 of 50 with an HTTP 400,
    where the bounded arm finishes at 3 to 5x fewer tokens
    (`docs/SKILL_STATE.md`).

    **The ordering is load-bearing.** A state patch is JSON in the SAME reply
    that may carry a tool call, so `applySkillStatePatch` runs first and
    returns the text with the `<state_patch>` block removed;
    `extractToolCalls` then never sees it. Parse calls first and the tool
    parser takes the patch for a call, which fails as a bogus tool invocation
    rather than as anything that names the state.

    **An invalid patch is DROPPED, not partially applied**, and the reason is
    the asymmetry: a bad merge silently corrupts every step after it, while a
    dropped one costs one step's bookkeeping and lands on
    `skillStateLastError`. Same instinct as Gotcha 13's tolerant decode.

    **`AppSkillStateSchema` is the single source for the prompt AND the
    validator.** Two spellings would let the app ask for a field it then
    rejects, which reads to a user as the model being stupid. The schema is
    GENERIC rather than per domain, which departs from the paper knowingly
    (it names "no fixed schema known in advance" as its first limitation, and
    a general coding assistant is arguably that case).

    `AppSkillStateTests` is offline and mutation-checked five ways;
    `AppSkillStateRealModelTests` drives a real server with the shipped
    protocol text and the shipped validator, and skips without
    `TURBOSPARK_SKILL_STATE_SERVER`. Run the second one after touching the
    schema or the protocol wording: growing the schema is exactly what
    `docs/SKILL_STATE.md` predicts would bring back the paper's
    schema-comprehension failures, so it is the change that needs a real model
    rather than a fixture.

36. **A SETTINGS CONTROL THAT CHANGES NOTHING IS GOTCHA 22'S BADGE, AND THE
    APPEARANCE PANE SHIPPED SIX OF THEM.** Audited 2026-08-31. `AppearanceManager`
    exposed `uiFont(...)` and `codeFont(...)`; `uiFont` had ZERO call sites in
    the app and `codeFont` had exactly ONE, inside `ThemeCodePreviewView` --
    the pane's own preview of itself. So UI/code font family, weight and size
    moved the preview card and nothing else, against ~864 hardcoded `.font(...)`
    calls. `activeForegroundColor`, `metadataForeground` and
    `borderStrokeOpacity` had no callers at all, and `diffMarkers` was read
    only by that same preview. The tell is cheap and worth running on any
    settings surface: grep for the accessor, not for the setting.

    **THE RESOLUTION BUG UNDERNEATH IT: `NSApp.effectiveAppearance` IS NOT
    MOVED BY `.preferredColorScheme`.** All seven `TurboSparkTheme` accessors
    branched on it. It is APPLICATION-level, while `.preferredColorScheme` sets
    the window's, so forcing the app to Light on a dark system drew light
    chrome out of `darkConfig` -- dark accent, dark background, dark contrast.
    `AppearanceSettingsPaneView` had always derived this correctly from
    `manager.appearance` plus `\.colorScheme`, so two spellings of one question
    disagreed and only the settings pane was right. `ThemeCodePreviewView` had
    the same bug with `isDark` ALREADY IN SCOPE, which made both its Light and
    Dark cards show whichever mode happened to be active. The fix is
    `ResolvedAppTheme`, an `Equatable` environment value injected once by
    `RootView`; being in the environment is also what makes it reactive, since
    a static computed var re-reads its inputs but cannot tell SwiftUI anything
    changed.

    **TWO PACKAGING TRAPS, AND BOTH FAIL SILENTLY.** `Package.swift` declares
    `.process("Resources")`, and that rule FLATTENS subdirectories: bundled
    fonts land at the resource bundle's root, not under `Fonts/`, exactly as
    `Resources/Logos/*.svg` do. Measured in both the `.build` bundle and the
    shipped `.app`, so the two layouts agree. That also rules out an
    `Info.plist` `ATSApplicationFontsPath`, which only looks directly under
    `Contents/Resources` and never inside the nested `.bundle` -- and which
    `swift run` has no plist for anyway (Gotcha 12). Register through
    `CTFontManagerRegisterFontsForURL` at `.process` scope instead.
    **And `??` is the wrong operator for the subdirectory fallback**:
    `Bundle.urls(forResourcesWithExtension:subdirectory:)` returns an EMPTY
    ARRAY rather than nil for a missing subdirectory, so a nil-coalescing chain
    never falls through, zero faces register, and nothing logs an error.

    **THE TEST FOR THAT SKIPPED TWICE BEFORE IT COULD FAIL.** Its `XCTSkipIf`
    was keyed first on `registerBundledFonts()`'s return value and then on
    `bundledFontURLs` -- both computed by the code under test, so a broken
    lookup and an empty directory were the same zero and the case went GREEN
    (skipped) under mutation both times. It keys on the checked-in files via
    `#filePath` now, and the mutation reddens. Generalise: **a skip condition
    derived from the thing under test cannot tell "nothing to do" from "it is
    broken"**, which is Gotcha 22's "a badge that cannot fail" applied to a
    gate rather than to a view.

37. **THE TEST SUITE OVERWROTE A USER'S CHAT ARCHIVE, AND THE ONLY SYMPTOM WAS
    A CHAT THAT WOULD NOT STAY DELETED.** Found 2026-08-31 from a bug report
    of exactly that shape: a chat titled "T10 chat" reappeared on every launch
    after being deleted in the UI. It is `HookDecisionRoutingTests`'s fixture
    (`AppChat(title: "T10 chat")`), and it was sitting ALONE in the real
    `~/Library/Application Support/TurboSpark/chats_archive.json`.

    **SEVEN STORES EACH SPELLED THEIR OWN PATH** -- `AppChatFileStore`,
    `AppProjectFileStore`, `MacAppSettings`, `GlobalMcpFileStore`,
    `AppHookStore`, `CustomToolManager` and `AppHookStdinPayload` all computed
    `applicationSupportDirectory + "TurboSpark"` inline, with no test seam
    anywhere. So an `AppModel()` built in a test read and WROTE real user data,
    and since Gotcha 13's archive is written WHOLE with `.atomic`, a one-chat
    fixture REPLACED the user's entire history. 20 `AppModel()` constructions
    across 10 test files could each do it.

    Three things make this class expensive and worth recognising. **Nothing
    fails**: the suite is green, the app starts, and the only tell is a chat
    the user did not create. **Deleting it in the UI does not help**, because
    the next `swift test` writes it back, so it reads as a broken Delete rather
    than as test pollution. And **it is unrecoverable** -- there is no backup,
    the write is atomic, and by the time anyone notices, the original is
    several runs gone.

    `AppStorageRoot` is the one root now, and two properties are load-bearing.
    The redirect is AUTOMATIC (keyed on the XCTest host, not on a flag a test
    sets), because a new test cannot be trusted to opt into a protection whose
    failure mode is silent; and it keys on the PROCESS rather than per test,
    because these stores are `.shared` singletons and static enums whose first
    touch can precede any test body. `TURBOSPARK_STATE_DIR` overrides it.

    `StorageIsolationTests` guards it, and the case that carries the file is
    `testSavingAChatDoesNotTouchTheRealArchive` -- it hashes the real archive
    around a save. Disabling the redirect reddens 3 of its 4 cases AND changes
    that hash, which is the original data loss reproduced on demand.

38. **AN INCREMENTAL `swift build` THAT DID NOTHING PRINTS THE SAME
    `Build complete!` AS ONE THAT SUCCEEDED.** A run that compiles emits
    `[N/M] Compiling ...`; a no-op emits `[0/3] Write swift-version...` and
    `Build complete! (0.2s)`. So a `swift build 2>&1 | grep error:` does the
    real compile, and the tidy `swift build` you run afterwards to "confirm"
    reports success having built nothing -- which is indistinguishable from a
    green build of your change. Read the output of the run that COMPILED, or
    `touch` the files first.

39. **A PERSISTED `@Published` PROPERTY'S DECLARED DEFAULT IS UNREACHABLE
    THROUGH THE PUBLIC INITIALIZER, AND A NAIVE TEST FOR IT PASSES OR FAILS
    ON LEFTOVER STATE INSTEAD.** `AppModel.init()` calls `loadSettings()` as
    its first statement, so a line like `@Published var interactionMode =
    .chat` is overwritten before any caller can observe it -- the value a
    fresh `AppModel()` actually reports comes from `MacAppSettings`'s own
    default. `AppModel().interactionMode == .chat` therefore does not test
    the property declaration; it tests whichever `settings.json` happens to
    sit in the shared, per-PROCESS `AppStorageRoot.directory` (Gotcha 37),
    which carries state across every test file that ran earlier in the same
    `swift test` invocation. Mutating the property's own default left such a
    test green; mutating `MacAppSettings`'s default (the level that is
    actually reachable) reddened it correctly. Test the `MacAppSettings`
    default directly, and for an `AppModel`-level round trip, delete the
    relevant file under `AppStorageRoot` first (`try?
    FileManager.default.removeItem(at: AppStorageRoot.file("settings.json"))`)
    so the assertion does not depend on execution order.

40. **`AppStorageRoot` COVERS THE APP'S STORES AND NOT THE ENGINE'S, SO A TEST
    THAT REACHES `TurboSparkCatalog` READS REAL USER DATA.** Gotcha 37's "one
    root now" is true of the seven stores that spelled their own Application
    Support path; `~/.turbospark/installed.json` is the CATALOG's, reached
    through the FFI, and nothing redirects it. So any assertion driven through
    `refreshModels`, `deleteModel` or `installed()` answers differently on a
    machine with a given alias installed -- which is not a flaky test, it is a
    test measuring the developer's disk.

    Found 2026-09-03 by a SURVIVING mutation: `deleteModel`'s alias-collision
    guard could not be reddened, because whether a colliding row existed was a
    property of `~/.turbospark` rather than of the fixture. The fix is the
    usual one -- extract the decision as a pure static
    (`AppModel.isCatalogTracked(model:in:)`) and feed it a fixture. Prefer that
    to seeding the real store, which is Gotcha 37's own failure mode wearing a
    different hat.

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

## The `state#N` ledger

`AppModel` and its extensions carry `(state#N)` markers on the comments that
explain a fixed state-layer defect. The numbers are inline references with no
central file, so before this index a reader who found `state#9` could only
learn what it meant by grepping for other mentions of it. What each covers:

| N | What it was |
|---|---|
| 1-5, 8, 13 | Predate this index and are no longer cited by any surviving comment. Recoverable only from git history. |
| 6 | Approving a pending call resumed at step 1, resetting `maxAutonomousSteps` on every approval. |
| 7 | `selectedChat`'s getter repaired `selectedChatID` on READ, publishing from inside a SwiftUI view update. |
| 9 | `generating` drops when a call is proposed, so a chat switch is legal in between; every append routes by captured `chatID`. |
| 10 | A finished turn's tail clobbered the state a reentrant continuation had just set up; `generationEpoch` guards it. |
| 11 | `setModelURL`'s `defer { opening = false }` fired before the awaited open began, so the loading indicator never rendered. |
| 12 | A skill's disabled flag lived only on the in-memory copy, and the `skill` tool ran disabled skills anyway. |
| 14 | `selected` was keyed on `alias`, which is not unique once a scanned row exists; keyed on `path` now. |
| 15 | A cancelled install's delayed tail reset the NEW install's state; `installEpoch` guards it. |
| 16 | `handleExtractedToolCall` spawned a detached `Task`, so `generating` went false while a tool ran (Send live, Stop dead). |
| 17 | `executeGenerationTurn` / `continueAgentLoop` resolved `selectedChatIndex` instead of taking the turn's chat. |
| 18 | `SubagentRunner` executed model-proposed tools with no permission gate at all. |
| 19 | An approved call ran under whichever project was selected at approval time, not the one its `.allow` was computed against. |
| 20 | Approve/deny appended a SECOND assistant turn and left the proposal stuck at `.pendingApproval`. |
| 21 | The expert-slot picker offered values the engine panics on, and omitted the legal 24. |
| 22 | A project agent taking a built-in's name inherited no restrictions, so a clone could widen `/explore`. |
| 23 | A rules symlink resolved outside the project and its contents reached the system prompt. |
| 24 | The JSON stores swallowed write failures and let the next write overwrite an unreadable file. |
| 25 | The quit flush lived in a view modifier, skipped the server, and bailed while generating. |
| 26 | `deleteProject` left the worktree, the hook store and the snapshot store bound to the deleted project. |
| 27 | `cancelInstall` claimed a cancellation the engine cannot perform and reopened the install guard. |
| 28 | A stop pressed during a bind was dropped, leaving a server listening that the UI showed as stopped. |
| 29 | Approving lowered `generating` on top of the continuation turn it had just started (state#16 reopened on the approval path); denying never raised it at all. |
| 30 | The turn AFTER an approved call took its project from the selection, not from the chat -- system prompt, workspace root, agent type, step cap, guardrails and the next call's permission evaluation all followed a switch state#19 had pinned for the call itself. |
| 31 | A guardrail-RESCUED call has its prose sanitized to "", and the history assembler's emptiness guard skipped the whole message before its `toolResults` loop ran, so the model never saw its own result and reissued the call to the step cap. |
| 32 | Tool results went back to the engine as mid-history `system` messages, which three of five fallback renderers refuse and `fit_window` prices at `u64::MAX` -- so the run lost its own history rather than reporting anything. Sent as `.tool` now. |
| 33 | `run()`'s submission task was stored nowhere, so a wedged `UserPromptSubmit` hook left `submitting` true with Send and Stop both dead and nothing for `cancel()` to reach. |
| 34 | Stop was refused while a call awaited approval (`generating` is false there by design), so `cancel()`'s own `clearPendingToolCall()` was unreachable -- and when it did run it left the persisted proposal at `.pendingApproval` across relaunches. |
| 35 | `UInt32(maxNewTokens)` TRAPPED the process on a hand-edited `settings.json`; clamped on load and at the use site. |
| 36 | `fitWindow` was called at the full `maxContext` and its outcome discarded: a truncated history was sent silently and a no-room verdict still called `generate`. |
| 37 | `extractToolCalls` returns every parsed call, the loop runs one, and the rest were dropped with nothing written anywhere. Recorded as refused with the rule now. |
| 38 | A `Stop` hook that blocked after a cancel appended an orphan user turn `continueAgentLoop` then refused to act on. |
| 39 | state#23's symlink containment was added to `ProjectRuleDetector` alone. `SkillParser` and `AgentParser` read the same class of file out of the same untrusted clone with no check: a skill body is returned by the `skill` tool and rated always-safe, an agent file becomes a subagent SYSTEM PROMPT. One `PathContainment` helper now, so a fourth reader can see the rule exists. |
| 40 | Hooks failed OPEN on timeout, spawn failure and transport failure: all three built `outcome == nil`, the aggregator skipped nil, the engine defaulted to `.allow`. A deny hook whose interpreter is missing permitted every call it was written to block. `.unavailable` resolves `.ask` on a permission event and feedback elsewhere. |
| 41 | The hook trust hash omitted the SOURCE, so trusting an entry in `~/.claude/settings.json` trusted the byte-identical entry in a cloned repository's. |
| 42 | `AppJSONStore.load` used `try? Data(contentsOf:)`, so a permissions or I/O failure read as "first run" and the next atomic write overwrote an intact file. And `lastWriteError` had no production reader at all -- the mechanism for making a failed write visible was itself invisible. |
| 43 | `deleteModel` had no `!generating` guard (`unloadModel()` returns silently there, so the directory was removed under a live mmap), never checked `serverAttachedSessions`, and matched on `alias ||` past state#14. |
| 44 | `installRepo` recorded no `installingAlias` and never checked `abandonedInstallAliases`, so state#27's two-writer protection covered catalog installs only. |
| 45 | `decodeIfPresent` tolerates an ABSENT key and nothing else. Three inner decoders threw on an unknown enum raw value or one bad array element and took the whole archive with them, past the hand-written tolerance of the outer ones. Element-level tolerance only: a wrong-TYPED array key still throws, or an intact file gets read as empty and overwritten. |

Add the next number here when you add the marker, or the index rots the way
the numbering did.
