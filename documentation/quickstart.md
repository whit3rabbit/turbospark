---
description: "Find, download, and run a model with TurboSpark."
icon: bolt
---

# Quickstart

Install TurboSpark first, then use the model catalog to choose a checkpoint.

```sh
turbospark-model list
turbospark-model recommend
turbospark-model pull gemma4
turbospark-check --model gemma4 --chat
```

The example uses the `gemma4` alias. Availability depends on the current catalog and your machine. See [supported models](models.md) for how to check model status.

To start the local API server with the same model:

```sh
turbospark-server --model gemma4
```

The server listens at `127.0.0.1:8080` by default. See [CLI and local API](cli-and-api.md) for endpoints and more commands.
