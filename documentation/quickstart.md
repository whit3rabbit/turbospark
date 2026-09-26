---
description: "Install a model and start your first local chat."
icon: message-circle
---

# First chat

Install and open TurboSpark, then follow these steps:

1. Open the model selector in the composer and choose **Discover new models**.
2. Pick a model and install it. Check the [model guide](models.md) if you are unsure which one to choose.
3. Select the installed model and choose **Load**.
4. Start a chat, enter a prompt, and send it.

The first load can take a little while as TurboSpark prepares the model. The app shows when the model is ready.

## Prefer the terminal?

Install and run a catalog model with the command-line tools:

```sh
turbospark-model recommend
turbospark-model pull gemma4
turbospark-check --model gemma4 --chat
```

Aliases can change as the catalog is updated. Use `turbospark-model list` to see current choices. To connect a client, see [CLI and local API](cli-and-api.md) and [local API examples](call-the-local-api.md).
