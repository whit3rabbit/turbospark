---
description: Comprehensive user guide for the TurboSpark native macOS desktop application.
icon: window
---

# Desktop App Guide

`TurboSpark.app` is a native SwiftUI desktop application designed specifically for Apple Silicon macOS. It integrates local conversational chat, agent tools, multimodal image generation, text rewriting (Humanize), an Audio suite with meeting notes, and decision models into a single native interface.

## Interface Workspaces

The application features dedicated workspaces accessible from the icon rail:

* **Chat**: Multi-session conversational LLM interface with system instructions, parameter controls, tool calls, and conversation recall.
* **Humanize**: Purpose-built text rewriting workspace powered by `humanizer-12b` with custom style rules and number slip audits.
* **Images**: Prompt-to-image generation powered by Swift MLX (Z-Image Turbo and Qwen-Image 2.1 Turbo).
* **Audio**: Recording and transcription in three modes with speaker naming and meeting notes, Voiceover, Voice Cloning, Voice Design, speech synthesis with Drama 3, OmniVoice, MegaTTS3 and Kokoro, and MiniMax music generation.
* **API & Decision Models**: Server configuration with an Embeddings tab, and the TypeSafe decision engine backed by the bundled OpenKind daemon.
* **Model Hub**: Catalog exploration, hardware-fit recommendations, download progress, and disk management.

### Window Layout

An icon rail always stays on screen, and beside it the selected section's panel: the chat list in Chat, every Audio page in Audio. The profile menu sits at the foot of the rail, with the temporary chat button beside the Chat/Projects tabs. **View > Toggle Sidebar** hides the panel, never the rail, and is disabled where there is no panel. The window minimum with a panel shown is 953pt.

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

### Tools in Chat

Chat turns can call the rootless tools behind a **Tools** toggle, on by default:

* A chip below the composer, an item in the plus menu, and **Enable tools in Chat by default** in General Settings control it.
* The Chat tool set is web search and fetch, browser automation, questions, checklists, planning, memory, and automation.
* File edit, search, shell, and MCP or custom tools stay **Projects-only**, so Chat cannot touch the filesystem or run commands.

### Conversation Recall

Each chat's user messages and prose answers are journaled to a per-profile store (mode 0700, excluded from backups). Once turns leave the prompt through compaction or the window fit, the ones relevant to the newest message come back as cited `<evidence>` blocks on that message, within at most 1,024 tokens.

* On by default: **Settings > General > Context Compaction**.
* Off for Ghost chats and passphrase-protected profiles, because the store is plaintext.
* Deleting or clearing a chat forgets its turns.

See [Agent Tools and Guardrails](../features/agents-and-tools.md) for the Projects tool runtime and permission gates.

---

## Model Selection & Discovery

In the chat header, the **Model** dropdown displays your currently installed models:

* Select **Discover new models** to open the Model Hub.
* The Hub automatically scores and ranks models based on your Mac's physical unified memory, GPU cores, and desired context length, and audio installs surface in model discovery with recommendations.
* Models indicate their verification status:
  * **Verified**: Gated with frozen quality tests and memory oracles.
  * **Runs**: Verified to install and generate coherent text.
  * **Caveat**: Supported with specific hardware or capacity considerations.

---

## Agent Tools & Permission Gates

TurboSpark includes a native agent runtime allowing models to execute tasks locally on your Mac in Projects:

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

The indexed and unindexed file search never returns hidden files, `.git/` contents (`.git/config` can hold credentials in remote URLs), `node_modules`, `target`, `.build`, or credential files (`.env*`, `*.pem`, `*.key`, `credentials`, `.npmrc`, `.netrc`). A search whose `path` names a sensitive file is classified high risk and prompts.

---

## Humanize (Text Rewriter)

The Humanize workspace runs `humanizer-12b` (a dense Gemma 4 12B model) to rewrite draft text:

* **Protected Terms**: Specify words or phrases that must remain untouched in the output.
* **Forbidden Terms**: Highlight words to avoid in the rewritten version.
* **Replacements**: Define `old -> new` replacement mappings.
* **Reproducibility**: Set a fixed numeric seed to produce identical rewrites for identical drafts.
* **Slip Audit**: The built-in audit checks whether the rewrite dropped numbers, invented quantities, or switched script languages.

The same flow is available from the terminal as `turbospark humanize`.

---

## Audio Suite

The Audio workspace provides recording, transcription, speech, and music. Each page's model picker rescans installed models when it opens, shows **Load** and **Unload** actions with a Loaded badge, and hides the download button for an installed row.

* **Record & Transcribe**: Three modes picked from a pill switcher (Voice Notes, Meeting, Any Audio), each with its own takes and per-job models. Live text streams while recording, and finished takes support speaker naming, revision, and export to Text, SRT, WebVTT and JSON. Meeting takes gain a notes summary, timestamp jump chips, custom vocabulary, and cross-meeting speaker recognition. See [Speech to Text](../audio/speech-to-text.md).
* **Voiceover**: Speaks a script without a reference; Drama 3 and OmniVoice pick the voice. **Voice Cloning**: Drama 3, OmniVoice, MegaTTS3 and Qwen3-TTS Base behind the consent gate, with Drama 3 asking for the reference's transcript. **Voice Design**: describe a voice in words. See [Text to Speech and Voice Cloning](../audio/text-to-speech.md).
* **Music**: Generate tracks with MiniMax Music 3 using the built-in prompt builder and lyrics guide, including the experimental Turbo 8-step checkpoint. See [Music Generation](../audio/music-generation.md).

Recording access is asked for only after the user allows it: starting a recording opens an access sheet that explains how audio is used and shows each permission's state, the macOS prompt appears only after **Allow**, and declining leaves import, transcription and editing usable.

---

## API Workspace

The API & Decision Models workspace configures the local server:

* **Server pane**: model, STT, TTS and music model pickers, the port and binding, and the API key. The key follows the same rule as the TypeSafe key. It is not saved to the Keychain, or read back at launch, unless **Remember this key in the Keychain** is on. A new install keeps a typed key for the session only. A key an earlier version already stored is kept, with the option turned on, so nobody's server quietly starts without authentication. Turning the option off deletes the item.
* **Embeddings tab**: picks the server's embedding model, saved and attached again after every start. It attaches and detaches live, without blocking the UI on a first Core ML load. Served embedding models are listed apart from chat models, and a button sends a test `POST /v1/embeddings`. The same picker replaces the Memory embedding field. Pinned profiles show **Verified** only when their gates ran on that pin. Other paths are probed and shown as **Not verified**, or refused with the reason when the family does not run here.
* **Engine Settings**: the ANE+Steel prefill split for dense Qwen is an Auto / On / Off picker. Auto activates on Apple Silicon M4+ or A18+ and falls back to Steel on older chips or missing programs.

---

## Privacy, Keychain, and Encrypted Profiles

* **Local Inference**: All model computations run on device with zero telemetry or cloud dependencies.
* **Encrypted Profiles**: User chats, notes, audio takes, and settings are saved in an encrypted local database.
* **Keychain Storage**: API keys and external credentials are saved only in the macOS Keychain upon explicit user request, per credential.
* **Appearance**: A Small text size joins the appearance settings. The factory defaults are the 14pt UI size with Hack as the code font. Bundled open-source fonts for code and UI include Iosevka, Geist, Geist Mono, Cascadia Code, Hack, and Source Code Pro.
