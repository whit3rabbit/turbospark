---
description: "Check catalog aliases and evidence status before choosing a model."
icon: compass
---

# Supported models

The model catalog is the source of truth for available aliases and their current support status. A model family being implemented does not mean every checkpoint in that family is ready to use.

List catalog entries and ask for a recommendation:

```sh
turbospark-model list
turbospark-model recommend --context 8192
```

Use the context you expect to run. The recommendation considers the active machine, the catalog evidence, and the requested context. Inspect an alias before installing it:

```sh
turbospark-model info <alias>
```

The evidence status has a specific meaning:

- `verified`: the row has a frozen quality gate and/or memory oracle. Check its details to see which evidence applies.
- `runs`: the model installed and generated coherent text, but does not have a frozen evidence row.
- `caveat`: the model runs with a documented limitation. Read the caveat before choosing it.

For an unlisted Hugging Face checkpoint, `probe` reads its metadata and checkpoint headers without downloading the model weights:

```sh
turbospark-model probe owner/repository@revision --file model.Q4_K_M.gguf
```

A `RUNNABLE` result means the checkpoint passes the probe's current format and runtime checks. It does not establish output quality, performance, or a fit for every machine. The exact repository, revision, format, and tokenizer sidecars matter.

Read [memory and capacity](memory-and-capacity.md) before choosing a large checkpoint, and [model install troubleshooting](model-install-troubleshooting.md) if a pull or probe fails.
