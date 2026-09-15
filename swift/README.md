# Swift Packages & macOS Application

This directory contains the Swift packages and native desktop application for TurboSpark:
1. **`TurboSpark`**: The core SwiftPM library package providing idiomatic Swift async/await bindings over the in-process C ABI (`crates/ffi`).
2. **`TurboSparkApp`**: The native macOS SwiftUI desktop chat and model management application, featuring a multi-turn agent runtime, native tool execution, Model Context Protocol (MCP) client, Syntext code search, and an embedded API server.

`TurboSpark` runs on macOS 13.0+ (Ventura) and `TurboSparkApp` runs on macOS 14.0+ (Sonoma), both on Apple Silicon. Both link against the native Metal inference engine in-process with zero HTTP or IPC overhead.

---

## Directory Structure

```
swift/
+-- docs/                            # 34 architectural, audit, and reference guides
+-- TurboSpark/                      # SwiftPM library package (macOS 13.0+)
|   +-- Package.swift
|   +-- Sources/
|   |   +-- CTurboSpark/             # Staged C header and staticlib (gitignored)
|   |   \-- TurboSpark/              # Swift async API (Session, Catalog, Server, Types, Errors)
|   \-- Tests/
|       \-- TurboSparkTests/         # ABI surface and real-install integration tests
\-- TurboSparkApp/                   # macOS SwiftUI Application (macOS 14.0+)
    +-- Package.swift
    +-- Localization/                # String catalogs (Localizable.xcstrings) and compiled .lproj
    +-- Sources/TurboSparkApp/
    |   +-- App/                     # Entry point (TurboSparkApp.swift, RootView.swift)
    |   +-- Chrome/                  # Window chrome, top bar, navigation rail, status bar
    |   +-- Components/              # Settings panes (Appearance, MCP, Plugins, Models, Soul Prompt)
    |   +-- Diagnostics/             # Inspector, telemetry, runner metrics, status HUD, fan control
    |   +-- Files/                   # AttachmentImporter, FilePreviewView, FilesSectionView
    |   +-- Generation/              # Chat sidebar, message pane, prompt composer, context ring
    |   +-- Installation/            # Model catalog downloader, ModelDetailPaneView, probe
    |   +-- Presentation/            # Markdown rendering, document text extraction (PDF/Word/Excel)
    |   +-- Resources/               # App prompts, bundled logos, command gate weights, soul prompts
    |   +-- Server/                  # Embedded API server pane, metrics store, endpoint catalog
    |   +-- State/                   # AppModel state coordinator, multi-profile storage, compaction
    |   +-- Theme/                  # Typography, colors, appearance tokens, dock icon, accessibility
    |   \-- Tools/                   # Agent tools, MCP client, Syntext search, sandbox, command gate
    \-- Tests/TurboSparkAppTests/    # 100+ app unit and integration tests
```

---

## Prerequisites

- **macOS**: 13.0 (Ventura) or newer for `TurboSpark` library; 14.0 (Sonoma) or newer for `TurboSparkApp` on Apple Silicon (M1/M2/M3/M4).
- **Rust toolchain**: Stable with target `aarch64-apple-darwin` (`rustup target add aarch64-apple-darwin`).
- **Xcode Command Line Tools**: `xcode-select --install` or full Xcode with Metal toolchain (`xcrun -sdk macosx metal`).
- **Swift**: Swift 5.9+ or Swift 6.

---

## Building and Running

All targets are driven from the repository root `Makefile`.

### 1. Build the FFI Static Library

Before building or testing any Swift package, build the Rust `turbospark-ffi` static library and stage the canonical header:

```sh
make swift-lib
```

This compiles `crates/ffi` for `aarch64-apple-darwin` with `MACOSX_DEPLOYMENT_TARGET=13.0` and stages `libturbospark_ffi.a` and `turbospark.h` into `swift/TurboSpark/Sources/CTurboSpark/`.

### 2. Compile String Catalogs

Before building `TurboSparkApp`, string catalogs must be compiled into runtime `.lproj` resources:

```sh
make compile-strings
# or directly:
# ./scripts/compile-strings.sh
```

### 3. Build the macOS App

Debug build:
```sh
make swift-app-build
```

Release build:
```sh
make swift-app-release
```

Package self-contained `.app` bundle or `.dmg` installer:
```sh
# Produces dist/TurboSpark.app with bundled CLI binaries
make app-bundle

# Produces verified, mountable dist/TurboSpark-<version>.dmg
make dmg
```

### 4. Run the macOS App

```sh
make swift-app
# or alias:
make swift-demo
```

### 5. Run Tests

- **C ABI & Surface Tests** (no downloaded model required, ~1 second):
  ```sh
  make swift-test
  ```

- **App Unit Tests** (covers state, tools, MCP, permissions, localization, git worktrees):
  ```sh
  cd swift/TurboSparkApp && swift test
  ```

- **End-to-End Real Model Tests** (requires a `.gturbo` model directory):
  ```sh
  make swift-test-real MODEL=~/models/gemma4.gturbo
  ```

- **Speculation Refusal & Vision Tests** (optional second and third installs):
  ```sh
  make swift-test-real MODEL=~/models/qwen38-27b-mtp.gturbo \
                       BLOCKED=~/models/ornith35b.gturbo \
                       IMAGE=~/models/vision-test.gturbo
  ```

---

## Swift API Overview (`TurboSpark`)

### Session Generation

```swift
import TurboSpark

let options = OpenOptions()
let session = try await TurboSparkSession(modelPath: "~/models/gemma4.gturbo", options: options)

var genOptions = GenerateOptions()
genOptions.temperature = 0.2
genOptions.reasoning = .medium

let messages = [ChatMessage(role: .user, content: "Explain how flash attention works.")]
for try await event in session.generate(messages, options: genOptions) {
    switch event {
    case .prefill(let done, let total):
        print("Prefill: \(done)/\(total)")
    case .content(let text):
        print(text, terminator: "")
    case .reasoning(let thinking):
        print("Thinking: \(thinking)")
    case .toolCall(let call):
        print("Tool call: \(call.name)(\(call.argumentsJSON))")
    case .stopped(let stopReason, _, _):
        print("Stopped: \(stopReason)")
    case .finished(let result):
        print("\nCompleted: \(result.newTokens) tokens, stop reason: \(result.stopReason)")
    }
}

// Non-blocking cancellation from any thread or Task
session.cancel()
```

### Catalog Management

```swift
import TurboSpark

// List installed and available models
let installed = try TurboSparkCatalog.installed()
let available = try TurboSparkCatalog.available()

// Stream download and repack directly from Hugging Face
for try await event in TurboSparkCatalog.install("gemma4") {
    switch event {
    case .stage(let name):
        print("Stage: \(name)")
    case .bytes(let done, let total):
        print("Progress: \(done)/\(total) bytes")
    case .finished(let model):
        print("Installed \(model.alias) at \(model.path)")
    }
}
```

### In-Process Local API Server

```swift
import TurboSpark

// Start an OpenAI/Anthropic-compatible server on loopback
let server = try TurboSparkServer.start(options: ServerOptions(port: 8080))
try server.attach(session: session)
if let url = server.baseURL {
    print("API server listening at \(url)/v1/chat/completions")
}
```

---

## macOS Application Highlights (`TurboSparkApp`)

1. **Multi-Turn Agent Runtime**:
   Autonomous execution loops supporting native tools (terminal shell commands, file viewing, targeted multi-line editing, Syntext code search, git management, and web browsing).
2. **Model Context Protocol (MCP) Client & Marketplace**:
   Supports both standard stdio and SSE MCP servers with interactive approval policies, schema validation, and tool search progressive disclosure.
3. **Syntext Indexed Code Search**:
   High-performance workspace indexing (`SYNTEXT.md`) enabling fast regex, exact, and AST code search with live unsaved file buffer updates.
4. **Context Ring & Compaction**:
   Visual context usage indicator displaying exact token consumption with breakdown popovers (system instructions, memory, turns, tools, attachments) and automatic conversation compaction (`SWIFT_COMPACTION.md`).
5. **Soul Prompts & Project Customization**:
   Customizable personality and system prompt directives with per-project `SOUL.md` instruction loading.
6. **Goal-Driven Execution Loops**:
   Long-running `/goal` commands that iterate until completion with stall detection, check-in schedules, and stop evaluation (`SWIFT_GOALS.md`).
7. **Accessibility & Theming**:
   Full VoiceOver integration, dynamic typography, dark/light themes, and strict macOS Reduce Transparency support.

---

## Permission Gate & Containment

`TurboSparkApp` executes model-proposed shell commands with security boundaries. In `.agentAuto` mode, safe read-only operations run without prompts while mutating commands trigger approval sheets. The safety barrier combines an allowlist in `TerminalCommandClassifier` with a bundled 192 KiB linear classifier (`CommandGate`) that scores commands for hazard and obfuscation in approximately 30 microseconds.

See [`docs/PERMISSION_GATE.md`](../docs/PERMISSION_GATE.md) and [`swift/docs/SWIFT_TOOLS.md`](docs/SWIFT_TOOLS.md) for full design and security details.

---

## Feature Architecture Guides

Comprehensive documentation lives in `swift/docs/`:

| Page | Covers |
|---|---|
| [`swift/docs/SWIFT_TOOLS.md`](docs/SWIFT_TOOLS.md) | Native tool execution, containment, sandboxing, hooks, and MCP client |
| [`swift/docs/SWIFT_TOOL_CATALOG.md`](docs/SWIFT_TOOL_CATALOG.md) | Reference catalog for all built-in agent tools and parameters |
| [`swift/docs/SYNTEXT.md`](docs/SYNTEXT.md) | Syntext indexed code search, project indexing, and live buffer sync |
| [`swift/docs/SWIFT_AGENT_MODE.md`](docs/SWIFT_AGENT_MODE.md) | The `.agentAuto` classifier permission mode and safety gates |
| [`swift/docs/SWIFT_PLUGINS.md`](docs/SWIFT_PLUGINS.md) | Plugin system manifest, contributions, lifecycle, and marketplace |
| [`swift/docs/SWIFT_SKILLS.md`](docs/SWIFT_SKILLS.md) | Agent skills architecture, scopes, file layout, and marketplace |
| [`swift/docs/SWIFT_MEMORY.md`](docs/SWIFT_MEMORY.md) | Auto-memory per-project directory, indexing, and memory tools |
| [`swift/docs/SWIFT_COMPACTION.md`](docs/SWIFT_COMPACTION.md) | Context compaction thresholds, summarization, and boundary rules |
| [`swift/docs/SWIFT_GOALS.md`](docs/SWIFT_GOALS.md) | Autonomous `/goal` loop, stall checks, and stop evaluators |
| [`swift/docs/SWIFT_CONTEXT_RING.md`](docs/SWIFT_CONTEXT_RING.md) | Composer context ring, tint tiers, and breakdown popovers |
| [`swift/docs/SWIFT_TURN_PIPELINE.md`](docs/SWIFT_TURN_PIPELINE.md) | Message queue, mid-turn steering at step boundaries, and system reminders |
| [`swift/docs/SWIFT_MESSAGE_EDITING.md`](docs/SWIFT_MESSAGE_EDITING.md) | Message edit, retry, and branching with version switching |
| [`swift/docs/SWIFT_MODEL_HUB.md`](docs/SWIFT_MODEL_HUB.md) | Catalog browsing, probing, fit evaluation, and server model attachment |
| [`swift/docs/SWIFT_PROFILES.md`](docs/SWIFT_PROFILES.md) | User profiles, settings isolation, and profile switching |
| [`swift/docs/SWIFT_LOCALIZATION.md`](docs/SWIFT_LOCALIZATION.md) | String catalog compilation, lproj structure, and parity gates |
| [`swift/docs/KEYBOARD_SHORTCUTS.md`](docs/KEYBOARD_SHORTCUTS.md) | Menu commands, keyboard shortcuts, and VoiceOver accessibility |
| [`swift/docs/SWIFT_SETTINGS_AUDIT.md`](docs/SWIFT_SETTINGS_AUDIT.md) | Settings persistence, UI control wiring, and audit checklist |
| [`swift/docs/SWIFT_STATE_LEDGER.md`](docs/SWIFT_STATE_LEDGER.md) | State defect ledger (`state#N`) documenting state mitigations |

For low-level C ABI and Swift binding contracts, see [`docs/SWIFT_BINDINGS.md`](../docs/SWIFT_BINDINGS.md).
