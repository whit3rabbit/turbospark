---
description: "Plan model storage and runtime memory for your Mac."
---

# Memory and capacity

Model file size and runtime memory answer different questions. Before installing a model, compare the recommendation for your Mac and the context you plan to use:

```sh
turbospark-model recommend --context 8192
```

The output reports both estimated runtime allocations and install size on disk. Runtime allocations include engine state such as the expert cache and KV cache. The disk figure includes the full model install.

## Mixture-of-Experts models

TurboSpark streams routed expert weights from SSD through a bounded cache. A model install can be larger than available memory because all expert weights do not need to be resident at once. The active expert cache and KV cache still need to fit, and the model still needs enough free disk space to install.

The number and size of experts, the cache slot count, and the context window affect memory use. File size alone cannot predict whether a checkpoint will fit.

## Dense models

Dense models do not use routed-expert streaming. Plan for the mapped model weights plus KV cache and scratch space, which grow with context. A process-footprint measurement may not include all mapped weights.

## Read measurements in context

Published footprint and speed measurements apply to their recorded chip, checkpoint, context, and cache-slot settings. Use the recommendation for the machine and context you plan to run; do not treat one measured row as a universal minimum-memory guarantee.

See [supported models](models.md) for evidence status and [CLI and local API](cli-and-api.md) for recommendation commands.
