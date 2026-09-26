---
description: "Install TurboSpark on Apple Silicon."
icon: rocket-launch
---

# Install TurboSpark

TurboSpark runs on Apple Silicon. The desktop app and Homebrew casks require macOS 14 Sonoma or later.

## Desktop app and command-line tools

The app cask installs `TurboSpark.app` and the `turbospark-check`, `turbospark-model`, and `turbospark-server` commands:

```sh
brew install --cask whit3rabbit/tap/turbospark
```

Open TurboSpark from Applications. Continue with the [first chat](quickstart.md).

## Manual download

Download the Apple Silicon DMG from [GitHub Releases](https://github.com/whit3rabbit/turbospark/releases), open it, and drag `TurboSpark.app` to Applications. macOS may ask you to confirm the first launch because the release app is not notarized.

## Command-line tools only

For command-line tools without the app, install the CLI cask:

```sh
brew install --cask whit3rabbit/tap/turbospark-cli
```

The app and CLI-only casks conflict. Choose one.

## Install from crates.io

With Rust installed, you can install the CLI and server crates:

```sh
cargo install turbospark-cli turbospark-server
```

Continue with the [quickstart](quickstart.md) or [desktop guide](desktop.md).
