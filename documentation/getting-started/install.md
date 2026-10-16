---
description: Install TurboSpark on macOS Apple Silicon.
icon: download
---

# Installation

TurboSpark is built specifically for Apple Silicon (M1, M2, M3, M4 and their Pro, Max, and Ultra variants). The desktop application and Homebrew casks require macOS 14 Sonoma or newer.

## Recommended: Homebrew Cask

Install the complete desktop application and command-line tools:

```sh
brew install --cask whit3rabbit/tap/turbospark
```

This installs `TurboSpark.app` in `/Applications` and links the core CLI tools onto your `PATH`:

* `turbospark`: Unified command-line interface for running, humanizing, and serving models.
* `turbospark-check`: Interactive text generation and chat.
* `turbospark-model`: Model catalog exploration, hardware recommendations, and downloads.
* `turbospark-server`: Local OpenAI- and Anthropic-compatible API server.

### First Launch Quarantine Clearance

The release application is ad-hoc signed. macOS gatekeeper blocks the first launch with an "Apple could not verify..." alert. Clear the quarantine flag once in Terminal:

```sh
xattr -dr com.apple.quarantine /Applications/TurboSpark.app
```

Alternatively, open **System Settings > Privacy & Security**, scroll down to the TurboSpark notification, and click **Open Anyway**.

---

## Command-Line Tools Only

If you only need the CLI binaries on a headless machine or server:

```sh
brew install --cask whit3rabbit/tap/turbospark-cli
```

*Note: The app cask and CLI-only cask conflict. Choose one.*

---

## Manual DMG Installation

Every [GitHub Release](https://github.com/whit3rabbit/turbospark/releases) includes an Apple Silicon disk image named `TurboSpark-<version>-arm64.dmg`:

1. Download the latest DMG from GitHub Releases.
2. Double-click to open and drag `TurboSpark.app` into your `/Applications` folder.
3. Clear the quarantine flag with `xattr -dr com.apple.quarantine /Applications/TurboSpark.app`.

---

## Installing via Cargo (crates.io)

You can build and install the published CLI and server binaries directly with Rust:

```sh
cargo install turbospark-cli turbospark-server
```

Installing `turbospark-cli` compiles all CLI tools, including the media utilities:
* `turbospark-image`: Native Metal image generation with Z-Image-Turbo.
* `turbospark-clone`: Zero-shot voice cloning with MegaTTS3.
* `turbospark-music`: Music generation with MiniMax Music 3.

---

## Building from Source

To compile the entire workspace from source:

1. Install Xcode Command Line Tools: `xcode-select --install`
2. Clone the repository and build with Rust:

```sh
git clone https://github.com/whit3rabbit/turbospark.git
cd turbospark
cargo build --workspace --release
```

Compiled binaries will be located under `target/release/`. See [Building from Source](../developer/building-from-source.md) for full developer prerequisites.
