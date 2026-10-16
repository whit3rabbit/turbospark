---
description: "Resolve common model catalog, probe, access, and disk-space issues."
---

# Model install troubleshooting

## A model alias is missing

List the current catalog and inspect the spelling of an alias:

```sh
turbospark-model list
turbospark-model info <alias>
```

A checkpoint is not supported just because its architecture name appears in the catalog. Check the exact repository and format on the [supported models](models.md) page.

## A checkpoint is not in the catalog

Probe its repository before transferring model weights:

```sh
turbospark-model probe owner/repository@revision
```

For GGUF, name the file to inspect:

```sh
turbospark-model probe owner/repository@revision --file model.Q4_K_M.gguf
```

The probe checks repository metadata, checkpoint headers, block types, shape, and tokenizer sidecars. It does not download the model weights or establish quality and performance.

If the probe accepts the checkpoint, install it with an alias:

```sh
turbospark-model pull --repo owner/repository@revision --alias local-model
```

For GGUF, provide the compatible Hugging Face tokenizer repository when needed:

```sh
turbospark-model pull --repo owner/gguf-model@revision --file model.Q4_K_M.gguf --alias local-model --sidecar-repo owner/original-model
```

## Hugging Face reports an access error

For a gated repository, authenticate with `turbospark-model auth --set` or set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the environment. Do not put the token in a command-line argument.

## A pull runs out of disk space

By default, completed download ranges stay in a cache until the install verifies. The cache and final install can occupy disk at the same time. On a disk-constrained machine, `TURBOSPARK_DISABLE_DOWNLOAD_CACHE=1` streams without retaining that extra copy. If the install is interrupted with caching disabled, the missing ranges must be downloaded again.

## A recommendation says the model will not fit

Run `turbospark-model recommend --context <tokens>` with the context you intend to use. Check [memory and capacity](memory-and-capacity.md) to understand the runtime allocation and disk-size estimates.
