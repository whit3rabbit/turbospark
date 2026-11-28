# gemma4_unified image input: Phase 0

The `gemma4_unified` (12B) image path. It is the Gemma 4 trunk with an
encoder-free embedder instead of the 27-block tower of
[GEMMA4_VISION_PHASE0.md](GEMMA4_VISION_PHASE0.md). Wired end to end on
synthetic fixtures. **No real weights have run.**

## Sources

| Source | Pin |
| --- | --- |
| Reference | mlx-vlm `models/gemma4_unified/{gemma4_unified,config,processing_gemma4_unified}.py` and `models/gemma4/gemma4.py` (`MultimodalEmbedder`), local copy, read not executed |
| `mlx-community/gemma-4-12B-it-4bit` | `73bcf09092aa277861d5a191b989b666f7f32e8f` |
| `mlx-community/gemma-4-12B-it-8bit` | `200bb6db075e137a4deb08838865ac4ddb86292e` |
| `lmstudio-community/gemma-4-12B-it-MLX-4bit` | `f45bda5639cf187cd5f127a5484813f5f1e3a6ad` |
| `google/gemma-4-12B-it` (BF16) | `43c9d014b231474fa7212dbb48a762566c2b255d` |

Facts below come from config, processor config, chat template, safetensors
index and shard header range reads. No weights were downloaded.

## What intake did before

A published `gemma4_unified` checkpoint did not install at all: the walk had no
bucket for `vision_embedder.*` or `embed_audio.*`, and an unclassified name
refuses. `vision_config` itself was ignored.

## Inventory

17 non-trunk tensors on the MLX conversions:

| Tensor | Shape | MLX dtype |
| --- | --- | --- |
| `vision_embedder.patch_ln1.{weight,bias}` | [6912] | BF16 |
| `vision_embedder.patch_dense.weight` | [3840, 6912] | affine U32 + `.scales`, `.biases` |
| `vision_embedder.patch_dense.bias` | [3840] | BF16 |
| `vision_embedder.patch_ln2.{weight,bias}` | [3840] | BF16 |
| `vision_embedder.pos_embedding` | [1120, 2, 3840] | BF16 |
| `vision_embedder.pos_norm.{weight,bias}` | [3840] | BF16 |
| `embed_vision.embedding_projection.weight` | [3840, 3840] | affine U32 + `.scales`, `.biases` |
| `embed_audio.embedding_projection.{weight,scales,biases}` | [3840, 640] | affine |

`.bias` is a linear layer's bias; `.biases` is a quantization plane.

## The forward pass

```text
patch = one 48 x 48 x 3 block, [rows][cols][RGB], pixels / 255, NOT centered
h = LayerNorm(patch)
h = h @ patch_dense^T + bias
h = LayerNorm(h)
h = LayerNorm(h + pos[x][0] + pos[y][1])        # (x, y) = (column, row)
t = RMSNorm_noscale(h, 1e-6) @ projection^T     # no sqrt(hidden) embed scale
```

One patch is one soft token. There is no attention, no pooling and no rope.

The three LayerNorms use eps 1e-5, `nn.LayerNorm`'s default: the reference
passes none and the config's `rms_norm_eps` (1e-6) reaches only the
projector. This was not checked against a running reference.

## Preprocessing

The processor subclasses Gemma 4's: same aspect-preserving resize to
multiples of 48 px under `280 * 9` patches of 16 px, same bicubic filter and
`1/255` rescale, no normalize. Each 48 px model patch is one 3 x 3 pooling
block of the Gemma 4 patches, so the front ends call `gemma4_preprocess`
unchanged and the runtime regroups (`Gemma4Patches::model_patches`). The
reference pads each image to 280 rows and drops the padding after the
embedder; nothing mixes rows, so only real rows run.

## Trunk side

Reused unchanged: `<|image|>` expands to `boi`, N placeholders, `eoi`; rows
replace the placeholder embeddings; sliding layers attend bidirectionally
inside a span (`use_bidirectional_attention: "vision"`); the KV ring holds
`window + 280`. The reference disagreement about full layers recorded in the
Gemma 4 page applies here too. The 12B chat template renders `<|image|>` for
an image part (the three MLX repos ship one template; google's differs).

## Wiring

| Stage | State |
| --- | --- |
| Config | `parse_gemma4_unified_vision_config`; refuses other patch, pool, budget, eps, mask or width values |
| Manifest | `visionTower: "gemma4_unified"`, `visionDepth: 1` |
| Intake | `pull --repo` enables the tower when `vision_embedder.` tensors are present; a plan without vision intent drops them by name |
| Install layout | the embedder and the FP16-dequantized projector are one blob in `packed_vision/` (about 100 MiB by shape) |
| Runtime | `vision/gemma4_unified.rs`, reached through `encode_image_gemma4` |
| CLI, FFI (including vision info), server | the existing Gemma 4 paths, keyed on `gemma4_image_ids` |
| Audio | not ported: `embed_audio.embedding_projection.*` is removed by name at pull, with a log line, only under `model_type: gemma4_unified` |
| `pull-vision` sidecar | refused by name |
| Soft-token budgets other than 280 | not built |
| Video | not built |
| BF16 `google/*` repos | still refused (quantize-on-install does not take the 12B); that spelling (`model.vision_embedder.`) was not wired |

## Gates

Synthetic only:

- `runtime/tests/gemma4_unified_vision_synthetic.rs`: GPU embedder against an
  independent CPU forward pass, chunked span prefill, ring capacity.
- `repack/tests/gemma4_unified_vision.rs`: blob contents, refusals, family
  gating.
- `catalog` `stream.rs` tests: the whole pull over a loopback repo.

Open real-model gate (nothing below has run):

1. `turbospark-model pull --repo mlx-community/gemma-4-12B-it-8bit@200bb6db075e137a4deb08838865ac4ddb86292e --alias gemma4-12b`
   (12.75 GB download). The dense Gemma 4 decode has run on INT8 only
   (`humanizer-12b`), so gate this width first. The 4-bit repo (6.77 GB,
   `73bcf090`) additionally exercises a dense INT4 trunk that no real install
   has run.
2. Tower parity against mlx-vlm on one image, then an image prompt answered
   through the CLI, FFI and server, then the memory oracle.
