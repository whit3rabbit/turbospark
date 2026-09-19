# Qwen3-VL-4B Phase 0 findings (`qwen3_vl` / `qwen3_vl_text`)

**UPDATE 2026-09-18: the family LANDED, text-first.** `ModelFamily::Qwen3Vl`
is registered, runs the shared Llama flow, and the pinned 4B checkpoint
streams, smokes, and carries frozen quality and memory gates (catalog row
`qwen3vl-4b`, `verified`). The open items in section 5 are now ANSWERED:
RoPE scope is FULL (the reference constructs the rotary at the whole head
dim; text positions collapse the mRoPE sections), deepstack's fusion is a
RAW ADD after trunk layers 0/1/2 (read off mlx-vlm's
`_deepstack_process`), and the pinned revision's naming was re-verified (its
own `model.safetensors.index.json` turned out to be stale, which cost the
first pull a 404 and is now defended in the stream path). The vision half --
tower plus deepstack injection -- remains open work, recorded in
`DEVIATIONS.md`'s `qwen3_vl` section.

**Original scope: write, do not build.** This is Part D of the vision memory sidecar
feature (`docs/VISION.md`) -- a fact-finding pass that scopes whether a small
Qwen3-VL bring-up is worth doing, before any code is written. Nothing in
`crates/` changed as a result of this page. Read `docs/NEW_MODEL.md` before
turning this into an actual bring-up.

**The headline finding: most of this is not new work.** The vision tower is
the SAME architecture this port already ships two checkpoints against
(`qwen3_5`'s tower, `docs/VISION_PHASE0.md` / `docs/VISION.md`), just at a
smaller hidden width, and the quantization scheme (INT4 affine, group 64) is
the one this port's kernels already dispatch. What is genuinely new is the
TRUNK (a dense GQA architecture this port has the pieces for but has not
assembled this way) and ONE new injection seam (`deepstack`, described
below). Everything else is a config-row-and-name-table exercise, per
`docs/NEW_MODEL.md`'s own framing of what counts as "new" versus "another
checkpoint of a family this port already runs".

## Reproducing this (nothing here is vendored)

Every number below was read off two published files, fetched by plain
`curl` (a repo's `config.json` is a few KB; a safetensors header is a
ranged GET of the first `8 + header_length` bytes, `crates/repack`
Gotcha 8's own convention) -- no weights downloaded, no model run:

```sh
curl -sL https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/main/config.json
curl -sL https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit/resolve/main/config.json

# The header-length prefix, then the header itself, off the QUANTIZED
# checkpoint's actual weight file -- see "The stale index" below for why
# this has to be `model.safetensors` and not what `model.safetensors.index.json`
# names.
curl -sL -r 0-7 https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit/resolve/main/model.safetensors
curl -sL -r 8-<8+header_length-1> https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit/resolve/main/model.safetensors
```

## 0. The stale index -- a real finding, not a hypothesis

`mlx-community/Qwen3-VL-4B-Instruct-4bit`'s own `model.safetensors.index.json`
claims a two-shard layout (`model-00001-of-00002.safetensors`,
`model-00002-of-00002.safetensors`) under `model.language_model.*` /
`model.visual.*` naming. **Neither is true of the repo as it stands.** The
HF API's file listing shows exactly one weight file, `model.safetensors`
(no shards), and its own header uses DIFFERENT names entirely:
`language_model.model.*` and `vision_tower.*` -- the second of which is the
EXACT prefix this port's `VISION_PREFIX` constant already expects
(`crates/repack/src/gemma4_checkpoint/classify.rs`), with no
`canonicalize_vision_header` rename needed for this exact artifact. The
index file was almost certainly written for an earlier upload (a 2-shard
BF16-to-INT4 conversion pass, going by the shard count) and never
regenerated when the repo was consolidated to one file.

**The lesson, stated once so nobody re-derives it under time pressure**: a
repo's own index/manifest is a claim, not a fact, and this is the same
class of trap `AGENTS.md` Gotcha 46 records for Xet transfer metadata and
Gotcha 58 records for a stale `models.json` row -- read the ACTUAL header
bytes before trusting a sidecar file that describes them. Any real
bring-up must re-verify this against whatever revision it pins (this page
read `sha 2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b`, unpinned/`main` at
read time), since a future re-upload could resolve the staleness by
re-sharding rather than by consolidating further.

## 1. The tower: `VisionShape` already parametrizes this

`vision_config` (`Qwen/Qwen3-VL-4B-Instruct`'s `config.json`), cross-checked
against the quantized checkpoint's own tensor shapes:

| field | qwen3-vl-4b | qwen3_5 (existing, for scale) |
|---|---|---|
| `depth` | 24 | 27 |
| `hidden_size` | 1024 | 1152 |
| `intermediate_size` | 4096 | 4304 |
| `num_heads` | 16 (head_dim 64) | 16 (head_dim 72) |
| `patch_size` / `temporal_patch_size` | 16 / 2 | 16 / 2 |
| `spatial_merge_size` | 2 | 2 |
| `num_position_embeddings` | 2304 (48x48) | 2304 (48x48) |
| `out_hidden_size` | 2560 (matches the trunk's `hidden_size`) | 5120 |
| `hidden_act` | `gelu_pytorch_tanh` (blocks) | same |
| `deepstack_visual_indexes` | **`[5, 11, 17]`** | `[]` (disabled) |

Every field but the last is a plain resize of numbers `VisionShape` already
reads generically (`crates/runtime/src/vision/shape.rs` takes `depth`,
`hidden`, `intermediate`, `heads`, `merge`, etc. as fields with no
architecture-specific literal in the struct itself) -- a new baseline's
`vision_config` block would populate the same struct. `patch_embed.proj.weight`
reads `[1024, 2, 16, 16, 3]` on the real header, the identical
`[out, T, P, P, C]` rank-5 layout `docs/VISION_PHASE0.md` item 5 already
established for the 27B tower, so the same "flatten to `(hidden, patch_dim)`,
no permutation" repack rule applies unchanged.

**`deepstack_visual_indexes` is the one real addition**, confirmed non-empty
here where every `qwen3_5` checkpoint declares it empty
(`docs/VISION_PHASE0.md` item 1). It names three block INDICES (5, 11, 17
of 24) whose outputs feed three extra mergers:

```text
vision_tower.deepstack_merger_list.{0,1,2}.norm.{weight,bias}         [4096]
vision_tower.deepstack_merger_list.{0,1,2}.linear_fc1.{weight,bias}   [4096,4096]
vision_tower.deepstack_merger_list.{0,1,2}.linear_fc2.{weight,bias}   [2560,4096]
```

18 tensors total (3 mergers x 6 tensors each), confirmed by counting the
real header rather than by arithmetic alone. Each deepstack merger has the
IDENTICAL shape to the main `vision_tower.merger.*` (same norm width, same
fc1/fc2 dims) -- it is the same merger structure run three more times, on
three intermediate block outputs instead of the final one.

**What this means for the forward pass**: a second, PARALLEL injection seam
alongside the M-V5 blit this port already has. After blocks 5, 11 and 17
(0-indexed, so their OUTPUTS, before block 6/12/18 runs), each one's
`[seq, hidden]` residual runs through its own deepstack merger to
`[merged, out_hidden]` and is ADDED into the TRUNK's residual stream at the
image's token positions, after trunk layers 0, 1 and 2 respectively (one
extra injection per deepstack index, at an increasing trunk depth) --
this is the mechanism's own name ("deep stack": push intermediate vision
features into progressively deeper trunk layers, not just the final
merger's output at the embedding layer). This is NOT the same seam as
`docs/VISION.md`'s existing embedding-position blit (M-V5's single
`encode_embed_any` replacement at prefill time): that one happens ONCE, at
layer 0's input; deepstack additionally touches trunk layers 0-2's
OUTPUTS. A real bring-up needs to read the reference's exact fusion point
(`mlx-vlm`'s `qwen3_vl` trunk forward, not inferred) before assuming "added
to the residual" is the whole story -- whether it is a raw add, a gated
add, or scaled, is exactly the kind of one-mutation-from-fluent-wrong-model
question `docs/VISION.md`'s "Six things" section catalogs for the existing
tower, and this page does not resolve it.

**`VisionRead`'s writer** (`crates/repack/src/gemma4_checkpoint/vision.rs`)
gains 18 resident tensors for this -- following `crates/repack` Gotcha 12's
existing artifact-versus-config discipline ("a component's config says
what the architecture has; only the tensors say what the artifact ships"),
a real bring-up should confirm every published Qwen3-VL checkpoint that
declares non-empty `deepstack_visual_indexes` actually SHIPS the 18
tensors, the way that gotcha found one `qwen3_5` checkpoint declaring a
tower and shipping none of it.

## 2. The trunk: `qwen3_vl_text`, dense GQA with qk-norm

`text_config`, and cross-checked tensor shapes (packed U32 for a quantized
weight; the logical dims are read off the surrounding config, per this
port's own `crates/gpu` convention of deriving group counts from companion
plane sizes rather than trusting a packed shape directly):

| field | value |
|---|---|
| `model_type` | `qwen3_vl_text` |
| `num_hidden_layers` | 36 |
| `hidden_size` | 2560 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 (note: `32 * 128 = 4096 != hidden_size`; q/k/v/o projections do NOT preserve `hidden_size` width, which this port's `ArchConfig` already treats as an independent field rather than derived) |
| `intermediate_size` | 9728 |
| `vocab_size` | 151,936 |
| `tie_word_embeddings` | `true` |
| `rms_norm_eps` | 1e-6 |
| `rope_theta` | 5,000,000 |
| `attention_bias` | `false` |
| `rope_scaling.mrope_interleaved` | `true` |
| `rope_scaling.mrope_section` | `[24, 20, 20]` (sums to 64 = `head_dim / 2`) |
| `rope_scaling.rope_type` | `"default"` |

**qk-norm per head, confirmed on the real tensors**: every layer carries
`self_attn.q_norm.weight` and `self_attn.k_norm.weight`, each shape `[128]`
(the head dim) -- the SAME per-head-norm-before-rope shape
`families/llama/`'s `Qwen3Moe` arm already implements
(`crates/runtime/AGENTS.md`'s `families/llama/` entry: "it norms q and k
PER HEAD before RoPE"). A `qwen3_vl_text` decode flow is that arm's
attention block plus the DENSE half's plain gated FFN (`mlp.gate_proj` /
`mlp.up_proj` / `silu_mul` / `mlp.down_proj`, already how this port's
`families/llama/dense.rs` and `families/qwen/dense.rs` both do a dense FFN)
-- no third flow, a variant reusing pieces of two existing ones, the same
shape `qwen_gdn_dense_27b()` is documented as being for `families/qwen/`.

**No `partial_rotary_factor` anywhere in `text_config`**, unlike `qwen3_5`'s
`0.25`. Read plainly, that means FULL rotary over the whole 128-wide head
(`rotary_dim = head_dim`) rather than a partial one -- but this page does
NOT confirm that against the reference's actual RoPE application code, only
against the config's silence, which per `AGENTS.md` Gotcha 39 is a claim
about what absence means and needs the FORMAT's own default rather than an
assumption carried over from a sibling family. **Open item for a real
bring-up**: read `transformers`' or `mlx-vlm`'s `Qwen3VLTextRotaryEmbedding`
construction directly (the same discipline `docs/VISION_PHASE0.md` item 2
used to settle `qwen3_5`'s own mRoPE question) before assuming full rotary.

**mRoPE dispatch is the SAME shape already built.** `mrope_section
[24, 20, 20]` summing to `head_dim / 2 = 64` is `docs/VISION_PHASE0.md`
item 2's exact derivation one level up (there: `[11,11,10]` summing to 32);
`rope_type: "default"` and `mrope_interleaved: true` match the qwen3_5
trunk's own values field for field. The existing
`rope_mrope_interleaved` kernel and its `t == h == w` dispatch condition
(`families/qwen/attn.rs`, `crates/gpu`) are section-COUNT-parametrized
already (the two `min()` clamps this port implements rather than the
narrower `i % 3` collapse, per `docs/VISION_PHASE0.md` item 2's own
caveat), so a different section triple summing to a different `freq_dim`
is a data change, not a new kernel. What is NEW is the dispatch SITE:
`families/qwen/attn.rs` is where this lives today, scoped to that one
family; a `qwen3_vl_text` flow living in (or beside) `families/llama/`
would need the same seam threaded there, which does not exist yet.

**Quantization**: the published 4-bit checkpoint's `quantization` block
reads `{"group_size": 64, "bits": 4, "mode": "affine"}` -- INT4 affine at
group 64, the width and group size EVERY existing MLX-sourced family this
port runs already uses (Gemma 4, Qwen 3.6, `qwen3_5`). No new kernel width.

**New family/name-table rows**, per `docs/NEW_MODEL.md`'s checklist:
`ModelFamily::Qwen3Vl` (or however the enum is spelled) and the two HF
`model_type` strings (`qwen3_vl` for the multimodal wrapper,
`qwen3_vl_text` for the trunk alone -- `arch_registry.rs`'s existing
`qwen3_5`/`qwen3_5_moe` pair is the precedent for two closely-named strings
that must NOT collapse into one family, `AGENTS.md` Gotcha 61's exact
shape), a name table for `language_model.model.*` /
`vision_tower.*` (confirmed by section 0 above to be the ACTUAL naming of
at least this one checkpoint -- do not assume `model.language_model.*` /
`model.visual.*` from the stale index without re-checking whichever
checkpoint a real pull targets), and the three `match family` sites
`docs/NEW_MODEL.md` names in `crates/catalog` (gate, name resolution,
baseline lookup).

## 3. Memory estimate at 4,096 context

Rough, not measured -- a real bring-up should replace every number here
with a header-probe-derived one before quoting it, per `AGENTS.md` Gotcha
36's rule of doing this arithmetic in Phase 0 rather than after a
download.

- **Mapped weights**: dense, no expert cache, so the whole INT4 trunk maps
  resident. 36 layers at hidden 2560 / intermediate 9728 is a small
  fraction of `qwen3_5`'s 64 layers at hidden 5120 -- call it **~3 GB**
  against the existing dense install's ~14 GB, an order of magnitude
  smaller (`AGENTS.md` Gotcha 40's rule applies here too: a dense
  install's mapped weights are NOT counted by `phys_footprint`, so this
  term does not appear in a measured ceiling the way KV does).
- **KV cache** at 4,096 context: 36 layers, 8 KV heads, head_dim 128, no
  sliding window declared in `text_config` (full attention throughout, per
  this config's silence on any window field) -- `36 * 8 * 128 * 2 * 2
  bytes/token = 147,456 bytes/token`, so 4,096 tokens is **~604 MB**. Cross-
  check this the way `crates/model-io` Gotcha 15's dense-7B arm was
  cross-checked (against `mistral_memory_oracle`'s measured KV), once a
  real install exists.
- **Vision tower**: at this hidden width (1024, a factor of ~1.3 smaller
  than `qwen3_5`'s 1152) and WITH Part B1/B2 of this same feature already
  landed, scratch is smaller per page than the existing tower's, and the
  three extra deepstack mergers add a fixed, small resident cost (18
  tensors at widths in the low thousands of elements each -- megabytes,
  not gigabytes).
- **Counted footprint** (the KV-dominated term `phys_footprint` actually
  charges for on a dense install, per Gotcha 40): under 1 GB at 4,096
  context, by the same reasoning `mistral_memory_oracle`'s 684 MiB at
  4,096 already demonstrates for a dense family of comparable KV shape.

## 4. What a real bring-up would gate on

The same four vision stages `docs/VISION.md` already runs for `qwen3_5`,
against mlx-vlm's `qwen3_vl` implementation instead of `qwen3_5`'s:
synthetic wiring check, mlx-vlm cosine parity (both the tower alone and
with deepstack engaged -- TWO parity runs, since deepstack is a second
injection this page has not verified end to end even on paper), the
injection/splice synthetic suite, and a full cross-engine logit dump on a
real text+image prompt. Plus the family's own quality gate and memory
oracle, per `docs/NEW_MODEL.md`'s standing checklist for any new family.

## 5. Open items

- **RoPE scope** (full vs. partial rotary): config is silent; read the
  reference's actual construction rather than assuming.
- **Deepstack's exact fusion**: raw add, gated, or scaled into the trunk
  residual -- read the reference's forward pass, do not infer from tensor
  shapes alone.
- **Which checkpoint's naming is authoritative**: this page verified ONE
  snapshot of ONE repo (unpinned `main`) and found its own index file
  already stale; re-verify against whichever revision a real pull
  actually pins, per the revision-pin discipline `docs/VISION.md`'s
  parity section states for the existing tower.
- **No sliding-window / hybrid-attention field found** in `text_config` --
  read as full attention throughout, unconfirmed against reference source.
- **The 2B variant** (`out_hidden_size` 2048 per public model cards) was
  not independently re-verified here; this page's numbers are the 4B's
  only.

This page settles none of the above; it exists to say the class of
questions is small and enumerable, which is the actual claim Part D of
`docs/VISION.md`'s sidecar feature makes: this is a scoping document, not
a green light to start `families/qwen3vl/`.
