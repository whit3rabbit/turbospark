# TurboSparkApp

Native macOS SwiftUI desktop chat and autonomous agent application for TurboSpark. Runs in-process on Apple Silicon with direct access to Metal inference pipelines, featuring multi-turn conversation persistence, native tool execution, Model Context Protocol (MCP) server support, Syntext code search, and an embedded OpenAI/Anthropic-compatible HTTP API server.

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
|   +-- Chrome/                       # Navigation rail, window toolbar, status bar
|   +-- Components/                   # Settings panes (Appearance, MCP, Plugins, Models, Soul)
|   +-- Diagnostics/                  # Performance inspector, telemetry, HUD, fan control
|   +-- Files/                        # Attachment importer, document previews, drag-and-drop
|   +-- Generation/                   # Chat sidebar, message timeline, prompt composer, context ring
|   +-- Installation/                 # Model catalog browser, download manager, probe sheets
|   +-- Presentation/                 # Markdown rendering, syntax highlighting, document parsing
|   +-- Resources/                    # Bundled prompts, logos, risk classifier weights
|   +-- Server/                       # Embedded API server UI, metrics store, endpoint catalog
|   +-- State/                        # AppModel state coordinator, goals, compaction, storage
|   +-- Theme/                        # Typography, colors, accessibility, Reduce Transparency
|   \-- Tools/                        # Agent tool execution, MCP client, Syntext search, sandbox
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
- **Security & Sandboxing**: `TrustBoundaryAndDurabilityTests.swift`, `WorkspaceAndNetworkGateTests.swift`.

---

## Subsystem Architecture

### 1. Agent Runtime & Tools (`Tools/`)
Provides native autonomous tools for the model:
- `ProcessExecutor.swift`: Bounded shell execution with environment isolation.
- `TerminalRiskGate.swift` & `CommandGate.swift`: Real-time command risk assessment and approval sheet gating.
- `SyntextCodeSearchTool.swift`: Indexed workspace code search with live buffer awareness (`SYNTEXT.md`).
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

### 5. Embedded API Server (`Server/`)
- Host an OpenAI/Anthropic compatible endpoint directly from the GUI app.
- Shares the GPU-resident model session without duplicating memory.
- Provides real-time metrics dashboards, request queues, and endpoint monitoring.

---

## Design Guidelines & Accessibility

- **ASCII Only**: All documentation, comments, and strings follow project ASCII rules.
- **macOS Reduce Transparency**: Automatically detects and respects macOS accessibility settings, rendering opaque backgrounds when Reduce Transparency is enabled (`Theme/`).
- **VoiceOver & Keyboard Navigation**: Every interactive element includes explicit accessibility labels and keyboard shortcuts (`KEYBOARD_SHORTCUTS.md`).
