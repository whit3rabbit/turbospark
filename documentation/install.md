---
description: "Install TurboSpark on Apple Silicon."
icon: rocket-launch
---

# Install TurboSpark

TurboSpark targets Apple Silicon. The macOS app and the Homebrew casks require macOS 14 Sonoma or later.

## App and command-line tools

Install the app cask:

```sh
brew install --cask whit3rabbit/tap/turbospark
```

This installs `TurboSpark.app` and the `turbospark-check`, `turbospark-model`, and `turbospark-server` commands.

## Command-line tools only

If you do not want the app, install the CLI cask:

```sh
brew install --cask whit3rabbit/tap/turbospark-cli
```

The app and CLI-only casks conflict. Install one of them.

## Install from crates.io

With Rust installed, you can install the CLI and server crates:

```sh
cargo install turbospark-cli turbospark-server
```

Continue with the [quickstart](quickstart.md).
