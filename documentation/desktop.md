---
description: "A short guide to chatting with local models in TurboSpark."
icon: laptop
---

# Desktop guide

TurboSpark is a hobby project for running local models on an Apple Silicon Mac. Install the app, choose a model, and chat. See [installation](install.md) if you have not installed it yet.

## Choose a model

Open the model selector and choose **Discover new models** to browse the catalog. Install a model, select it, then choose **Load**. The catalog shows whether a model has frozen verification evidence, has only been run, or has a caveat.

The model file can be much larger than the memory needed to run it. Read [memory and capacity](memory-and-capacity.md) before downloading a large model, and check [supported models](models.md) for current aliases and evidence.

## Chat

Start a chat and send a prompt. The app keeps separate conversations so you can return to earlier chats. If generation stalls, use Stop, then try a shorter prompt or a smaller context.

Image input needs a model install with compatible vision support. Image generation uses a separate image model. Check the model notes before installing either.

## Optional tools

The app includes permission-gated agent tools for tasks such as shell commands, file access, web requests, Git, and MCP. Review the requested permission before allowing a tool to act on your Mac.

## Connect another app

The local server exposes OpenAI- and Anthropic-compatible endpoints. Start it with `turbospark-server --model <alias>`, then follow the [local API examples](call-the-local-api.md). The default address is loopback; use an API key if you configure a wider network bind.
