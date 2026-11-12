# TurboSparkApp

Native macOS SwiftUI desktop chat and autonomous agent application for TurboSpark. Text and Image generation use in-process Metal pipelines. The API workspace also manages a separate loopback OpenKind service for TypeSafe requests. The app includes multi-turn conversation persistence, native tool execution, Model Context Protocol (MCP) server support, and Syntext code search.

## Platform Requirements

- **Operating System**: macOS 14.0 (Sonoma) or newer on Apple Silicon (M1/M2/M3/M4).
- **Toolchain**: Xcode 15+ / Xcode Command Line Tools, Swift 5.9+ or Swift 6, and Rust stable toolchain with `aarch64-apple-darwin`.

---

## Directory & Subsystem Structure

```
swift/TurboSparkApp/
+-- Package.swift
+-- Localization/                     # Localizable.xcstrings and generated runtime .lproj
+-- Sources/TurboSparkApp/
|   +-- App/                          # App lifecycle: TurboSparkApp.swift, RootView.swift
|   +-- Audio/                        # Audio Studio: record, transcribe, speech, voice cloning, music
|   +-- Browser/                      # In-app browser pane, tabs, settings, dialog approval
|   +-- BrowserAutomation/            # Agent browser control sessions and command port
|   +-- Chrome/                       # Navigation rail, window toolbar, status bar
|   +-- Components/                   # Settings panes (Appearance, MCP, Plugins, Models, Soul)
|   +-- Diagnostics/                  # Performance inspector, telemetry, HUD, fan control
|   +-- Files/                        # Attachment importer, document previews, drag-and-drop
|   +-- Generation/                   # Chat sidebar, message timeline, prompt composer, context ring
|   +-- Humanize/                     # Humanize workspace: draft and rewrite cards, diff
|   +-- Images/                       # Image generation composer, gallery, settings
|   +-- Import/                       # Chrome bookmark import
|   +-- Installation/                 # Model catalog browser, download manager, probe sheets
|   +-- Presentation/                 # Markdown rendering, syntax highlighting, document parsing
|   +-- Resources/                    # Bundled prompts, logos, risk classifier weights
|   +-- Server/                       # Text/Image API and TypeSafe playground UI
|   +-- State/                        # AppModel state coordinator, goals, compaction, storage
|   +-- Theme/                        # Shared controls, surfaces, progress bar, motion, type, colors
|   +-- Tools/                        # Agent tool execution, MCP client, Syntext search, sandbox
|   \-- Workflows/                    # Workflow interpreter, run engine, journal
\-- Tests/TurboSparkAppTests/         # 100+ unit, integration, and UI state test suites
```

---

## Building and Running

Because `TurboSparkApp` links against the Rust inference engine and requires compiled runtime string catalogs, run prerequisite builds before compiling with SwiftPM:

### 1. Build Prerequisites (from repository root)

```sh
# 1. Build and stage the C ABI static library
make swift-lib

# 2. Compile string catalogs into runtime .lproj resources
make compile-strings
```

### 2. Run the App

From repository root:
```sh
make swift-app
```

Or from this directory via SwiftPM:
```sh
swift run TurboSparkApp
```

An unbundled build is attributed to the app that launched it, so microphone
and screen recording prompts and System Settings rows name Terminal or Xcode,
not TurboSpark. Test recording permissions in the bundle from
`make app-bundle`.

### 3. Build Release Bundle

```sh
# From repository root:
make app-bundle   # Produces dist/TurboSpark.app
make dmg          # Produces signed, mountable installer DMG
```

---

## Testing

Run the full Swift application test suite directly using `swift test`:

```sh
# Run all unit tests for TurboSparkApp
swift test

# Run a specific test suite
swift test --filter McpClientEngineTests
swift test --filter SyntextCodeSearchToolTests
swift test --filter GoalLoopTests
```

The test suite contains over 100 test files covering:
- **Agent Tools & Execution**: `ProcessExecutorTests.swift`, `McpClientEngineTests.swift`, `SyntextCodeSearchToolTests.swift`, `TerminalRiskGateTests.swift`, `ToolPermissionsTests.swift`.
- **Conversation State & Compaction**: `MessageQueueTests.swift`, `MessageEditBranchTests.swift`, `AppChatCompactionTests.swift`, `MemoryFeatureTests.swift`.
- **Goals & Background Work**: `GoalLoopTests.swift`, `PlanningAndInteractiveToolTests.swift`, `SubagentBatchRoutingTests.swift`.
- **Model Management & Hub**: `ModelManagerTests.swift`, `ModelFitAndInstallGateTests.swift`, `ModelProbeSheetTests.swift`.
- **Localization & Accessibility**: `LocalizationParityTests.swift`, `LocalizationAndAccessibilityTests.swift`.
- **Audio Studio**: `AudioVoiceCloneTests.swift`, `AudioVoiceCloneWorkspaceTests.swift`, `AudioVoiceReferencePresentationTests.swift`, `AudioRecordingSetupTests.swift`, `AudioLibraryTests.swift`.
- **Security & Sandboxing**: `TrustBoundaryAndDurabilityTests.swift`, `WorkspaceAndNetworkGateTests.swift`.

---

## Subsystem Architecture

### 1. Agent Runtime & Tools (`Tools/`)
Provides native autonomous tools for the model:
- `ProcessExecutor.swift`: Bounded shell execution with environment isolation.
- `TerminalRiskGate.swift` & `CommandGate.swift`: Real-time command risk assessment and approval sheet gating.
- `SyntextCodeSearchTool.swift`: Opt-in indexed workspace code search (`SYNTEXT.md`).
- `FileTools.swift` & `MultiEdit.swift`: Precise file inspection, patch application, and non-contiguous replacements.
- `GitWorktreeTool.swift`: Git branch, status, diff, and commit operations.

### 2. Model Context Protocol Client (`Tools/Mcp/`)
Embeds an extensible MCP client engine:
- Connects to stdio and SSE MCP servers.
- Manages permissions, configuration schemas, and dynamic tool search progressive disclosure.
- Built-in MCP marketplace integration for installing approved tool servers.

### 3. State Coordination & Storage (`State/`)
The `AppModel` coordinator acts as the single source of truth:
- Manages multi-chat conversations, branching edit trees, and active session generation.
- Executes automatic background context compaction (`SWIFT_COMPACTION.md`) when token counts approach model limits.
- Persists user preferences, projects, soul prompts (`SOUL.md`), and custom instructions across sessions.

### 4. Interactive Composer & Context Ring (`Generation/`)
- Real-time token accounting via a dynamic context ring displaying token utilization.
- Popover breakdown separating system prompts, active project memory, chat turns, tool schemas, and file attachments.
- Message alternate branching and regeneration controls (`SWIFT_MESSAGE_EDITING.md`).

### 5. API Workspace (`Server/`)
- Text hosts the existing chat and completion endpoints. Image adds a
  prompt-to-PNG endpoint on the same TurboSpark listener and API key.
- TypeSafe starts a separate OpenKind daemon with its own loopback URL and
  Keychain key. Its native playground builds Noul, Choice, and Score requests
  or accepts raw JSON, then shows typed answers and local timing results.
- The workspace retains metrics, request queues, and endpoint monitoring.
  See [Swift API workspace](../docs/SWIFT_API_WORKSPACE.md).

### 6. Audio Studio (`Audio/`)
- `AudioWorkspaceController` and its extensions own recording, import,
  playback, model selection, and generation jobs. Views read it and call its
  seams; they hold no audio state of their own.
- Voice Cloning (`VoiceCloneWorkspaceView`) is a two-column page: voice setup
  (reference card, engine cards, model) beside a steps strip, the script
  composer, a live progress card, and the takes list. It falls back to one
  column for a narrow window or Extra Large text.
- The reference card takes a dropped file, an import, an inline microphone
  recording (stops at 30 seconds), or a pick from the library sheet, which
  also lists references recently used by past takes. Consent covers one exact
  reference region; reusing a reference or recording a new one never carries
  it. See [Audio workspace](../../docs/AUDIO_WORKSPACE.md).
- Recording asks for macOS access only when the user chooses to.
  `RecordingPermissions` reads microphone and screen recording status without
  prompting. Start recording and the reference card's Record button open the
  access sheet (`RecordingAccessSheet`) when a source is missing, and macOS
  prompts only after the user presses Allow there. The sheet can record with
  just the allowed source, and Not now leaves import, transcription, and
  editing usable. `AudioCapture` never prompts. Without access it throws an
  error that points to the sheet.

---

## Design Guidelines & Accessibility

Read [DESIGN.md](../../DESIGN.md) before any UI change. It holds the shared
control vocabulary (`Theme/AppControls.swift`, `Theme/AppSurfaces.swift`,
`Theme/AppProgressBar.swift`, `Theme/TurboSparkMotion.swift`) and the layout,
theming, accessibility, and localization rules.

- **ASCII Only**: All documentation, comments, and strings follow project ASCII rules.
- **Motion**: Animations use the `TSMotion` curves and honor Reduce Motion (the app setting first, then the system's). Glows become solid rings under Increase Contrast.
- **macOS Reduce Transparency**: Automatically detects and respects macOS accessibility settings, rendering opaque backgrounds when Reduce Transparency is enabled (`Theme/`).
- **VoiceOver & Keyboard Navigation**: Every interactive element includes explicit accessibility labels and keyboard shortcuts (`KEYBOARD_SHORTCUTS.md`).
