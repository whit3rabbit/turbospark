---
description: Comprehensive user guide for the TurboSpark native macOS desktop application.
icon: window
---

# Desktop App Guide

`TurboSpark.app` is a native SwiftUI desktop application designed specifically for Apple Silicon macOS. It integrates local conversational chat, agent tools, multimodal image generation, text rewriting (Humanize), an Audio Studio, and decision models into a single native interface.

## Interface Workspaces

The application features dedicated workspaces accessible from the sidebar:

* **Chat**: Multi-session conversational LLM interface with system instructions, parameter controls, and agent tooling.
* **Humanize**: Purpose-built text rewriting workspace powered by `humanizer-12b` with custom style rules and number slip audits.
* **Images**: Prompt-to-image generation powered by Swift MLX (Z-Image Turbo and Qwen-Image 2.1 Turbo).
* **Audio Studio**: Audio recording, Whisper transcription, speech synthesis, MegaTTS3 zero-shot voice cloning, and MiniMax music generation.
* **API & Decision Models**: Server configuration and the TypeSafe decision engine backed by the bundled OpenKind daemon.
* **Model Hub**: Catalog exploration, hardware-fit recommendations, download progress, and disk management.

---

## Chat Workspace

### Starting and Managing Conversations

* Click **+ New Chat** in the toolbar to create an independent conversation thread.
* Conversations are automatically saved to your encrypted local profile.
* You can rename or delete sessions from the sidebar context menu.

### Generation Controls & Sampling

Click the **Sliders** icon in the chat header to adjust generation parameters:

* **Temperature**: Controls randomness (default 0.7).
* **Top-P (Nucleus)**: Cutoff probability for token selection (default 0.95).
* **Min-P**: Minimum probability floor relative to top token.
* **Repetition Penalty**: Penalizes repeated tokens in generation.
* **Reasoning Effort**: For models with reasoning tokens (e.g. DeepSeek-R1 distillations or QwQ), toggles visibility and verbosity of reasoning chains.
* **System Prompt**: Custom instructions applied to the start of the chat context.

---

## Model Selection & Discovery

In the chat header, the **Model** dropdown displays your currently installed models:

* Select **Discover new models** to open the Model Hub.
* The Hub automatically scores and ranks models based on your Mac's physical unified memory, GPU cores, and desired context length.
* Models indicate their verification status:
  * **Verified**: Gated with frozen quality tests and memory oracles.
  * **Runs**: Verified to install and generate coherent text.
  * **Caveat**: Supported with specific hardware or capacity considerations.

---

## Agent Tools & Permission Gates

TurboSpark includes a native agent runtime allowing models to execute tasks locally on your Mac:

* **Bash / Shell Tool**: Runs commands in a sandboxed terminal.
* **File System Tool**: Reads, writes, and searches files in permitted directories.
* **Web Fetch Tool**: Fetches web documentation and APIs over HTTP.
* **Git Tool**: Interacts with local Git repositories.
* **MCP (Model Context Protocol)**: Connects to external MCP servers for custom capabilities.

### Security and Permission Prompts

To protect your system:
* The first time a tool attempts an external action (running a shell command, writing a file outside scratch, accessing the microphone), TurboSpark presents an interactive **Permission Gate**.
* You can inspect the exact tool name, arguments, and target paths.
* Choose **Allow Once**, **Allow for this Session**, or **Deny**.
* Unpermitted tools return an explicit denial to the model, preventing unauthorized execution.

---

## Humanize (Text Rewriter)

The Humanize workspace runs `humanizer-12b` (a dense Gemma 4 12B model) to rewrite draft text:

* **Protected Terms**: Specify words or phrases that must remain untouched in the output.
* **Forbidden Terms**: Highlight words to avoid in the rewritten version.
* **Replacements**: Define `old -> new` replacement mappings.
* **Reproducibility**: Set a fixed numeric seed to produce identical rewrites for identical drafts.
* **Slip Audit**: The built-in audit checks whether the rewrite dropped numbers, invented quantities, or switched script languages.

---

## Audio Studio

The Audio workspace provides comprehensive sound processing:

* **Record**: Capture microphone, system loopback audio, or both simultaneously.
* **Live Transcription**: Realtime speech-to-text preview using Whisper.
* **Voice Cloning**: Clone a reference voice using MegaTTS3 (requires explicit consent confirmation).
* **Music**: Generate tracks with MiniMax Music 3 using the built-in prompt builder and lyrics guide.

---

## Privacy, Keychain, and Encrypted Profiles

* **Local Inference**: All model computations run on device with zero telemetry or cloud dependencies.
* **Encrypted Profiles**: User chats, notes, audio takes, and settings are saved in an encrypted local database.
* **Keychain Storage**: API keys and external credentials are saved only in the macOS Keychain upon explicit user request.
