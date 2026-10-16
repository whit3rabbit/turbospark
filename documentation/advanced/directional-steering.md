---
description: Runtime directional residual steering and abliterated models for refusal suppression.
icon: compass-drafting
---

# Directional Residual Steering

Directional residual steering is an advanced inference technique that modifies model activations at runtime without retraining or fine-tuning underlying model weights.

In TurboSpark, directional steering is primarily used for **refusal obliteration**: removing refusal behaviors from safety-aligned models while retaining all core knowledge and coding capability.

## How Directional Steering Works

Research into language model representations demonstrates that refusal mechanisms are mediated by specific low-dimensional linear directions in intermediate residual activation layers:

1. **Refusal Vector Identification**: By contrasting model activations on harmless prompts against refused prompts, a mean direction vector $v$ is isolated.
2. **Orthogonal Projection**: During the forward pass, before feeding the residual stream into attention or FFN blocks, TurboSpark computes the projection of the activation vector along $v$ and subtracts it:
   $$x' = x - (x \cdot \hat{v}) \hat{v}$$
3. **Weight Preservation**: The model files on disk remain byte-identical to original upstream weights. Steering can be enabled, disabled, or scaled dynamically per request.

---

## Abliterated Models

TurboSpark provides guidance on using pre-abliterated models - checkpoints where orthogonal projection has been baked into the weight matrices or where steering vectors are packaged in the `.gturbo` bundle:

* Avoids repetitive "I cannot fulfill this request" disclaimers when generating cybersecurity proofs-of-concept, fictional creative writing, or system administration scripts.
* Retains coding fluency, mathematical reasoning, and factual knowledge without catastrophic forgetting.

To apply runtime directional steering:

```sh
# Enable steering vector with scale factor 1.0
export TURBOSPARK_STEER_VECTOR=~/.turbospark/vectors/refusal_direction.bin
export TURBOSPARK_STEER_SCALE=1.0

turbospark-check --model qwen36 --chat
```
