# LFM3-Omni vision: Phase 0 and bring-up record

Facts about the `lfm3_omni` checkpoint, what this port does with its IMAGE
half, and what was measured. The image path is the EXISTING `lfm3` trunk plus
the existing `VisionTowerKind::Lfm3Vl` tower
([docs/LFM3_VL_VISION_PHASE0.md](LFM3_VL_VISION_PHASE0.md)), so nothing there
is repeated here; this page records the omni-specific nesting, the audio
refusal, and the gates. The AUDIO half (LFM3-Audio encoder, depth decoder,
Liquid codec) is out of scope and refused by construction.

## Sources

| What | Where | Pin |
| --- | --- | --- |
| Checkpoint installed and run | `LiquidAI/LFM3-Omni-2B-MLX-8bit` | `5901d16cbc5f1774243db907cd407f825591f743` |
| Other builds seen | `LiquidAI/LFM3-Omni-2B` (unquantized) `c01718a568e53ae936e6b5e1ff5ab2de7d7f388b`, `-MLX-4bit` (group 64 with per-tensor 6-bit overrides), `-MLX-bf16` | configs and file listings only, except 8-bit |
| Decision-model sibling | `LiquidAI/d1-Omni-2B` | `51918b53052eacfd24062f99f13556c3c56fcf63` |
| Reference code read | mlx-vlm `mlx_vlm/models/lfm3_omni/{config,lfm3_omni,processing_lfm3_omni}.py` and `lfm3_vl/` | commit `e8035ae`, shallow clone |

Every fact below came from the real `config.json`, `processor_config.json`,
`tokenizer.json`, `chat_template.jinja` and the published
`model.safetensors.index.json` at the pinned revisions.

## What the checkpoint is

The omni config is LFM3-VL's config plus an audio half, all at the root:

- `model_type: "lfm3_omni"`, `text_config.model_type: "lfm3"`. The trunk
  fields are the 1.7B baseline field for field (hidden 2048, 24 layers, 16
  heads over 4 KV, head_dim 128, FFN 7680, the same 13 conv / 6 full / 5
  sliding mask, partial rope 0.5 at theta 1e6, sliding theta 1e4), so the
  trunk installs and runs through the existing `lfm3` flow with no change.
- `vision_config` (`model_type: "lfm3_vl_vision"`) and the four projector
  fields are byte-identical to `LFM3-VL-2B-MLX-8bit`'s (verified by
  comparing the two fetched configs). The tower is the same 14-block
  alternating conv/attention kind.
- The audio sections: `audio_encoder_config` (inner `model_type
  lfm3_omni_encoder`, a 17-layer `lfm2`-spelled encoder) and `depth_config`
  (a 3-layer MaskGIT depth decoder), plus the audio token ids and the
  codec subfolder in the repo. The reference `ModelConfig.__post_init__`
  refuses an omni config without both sections.
- The omni config does NOT declare `vision_start_token_id` /
  `vision_end_token_id` (checked on all three builds). Its `tokenizer.json`
  maps `<|vision_start|>` to 125009 and `<|vision_end|>` to 125010 - the
  exact ids the VL config declares - and its chat template renders the same
  `<|vision_start|><|image|><|vision_end|>` span, so the reader defaults
  those two fields to the tokenizer's ids and a config that declares other
  values still wins.
- The `processor_config.json`'s `image_processor` object is the same flat
  shape the VL install writes (`patch 16, temporal 2, merge 2, 0.5/0.5,
  bicubic`), with `longest_edge` 2097152 against the VL repo's 16777216.
  `catalog::stream::lfm3_preprocessor_json` consumes it unchanged.

## Tensor inventory (omni, image half)

Prefixes verified against the published `model.safetensors.index.json`
(973 tensors). The image half carries the SAME prefixes the VL checkpoint
uses, so the reader needed no rename:

- `language_model.model.embed_tokens`, `.embedding_norm`, `.layers.*` - the
  trunk (590 layer tensors plus 4).
- `vision_tower.patchifier.weight`, `.embedding_norm.weight`, `.layers.*` -
  the tower, same role names per block as VL.
- `multi_modal_projector.layer_norm.*`, `.linear_1.*`, `.linear_2.*` - the
  projector, same six tensors as VL.

The audio half (241 tensors) carries four prefixes nothing in the image path
claims: `audio_tower.` (170), `audio_projector.` (6), `depth_decoder.` (64),
`codebook_embedding.` (1).

## What intake does

`crates/repack/src/gemma4_checkpoint/vision_lfm3.rs`:

- `parse_lfm3_vl_vision_config` now dispatches on the root `model_type`:
  `lfm3_vl` and `lfm3_omni` (and a config that omits the key, as the trimmed
  fixtures do). Any other string is refused by name. The omni branch checks
  the audio sections' shape and derives the same `VisionConfig` the VL
  branch derives, with the two span-id defaults above.
- `refuse_foreign_lfm3_omni_audio` requires both audio sections to be
  PRESENT and to name the published models (`lfm3_omni_encoder`, `lfm3`),
  and refuses with `Gemma4Error::UnsupportedTensor` naming the section when
  they are missing or foreign. Nothing from either section reaches the
  returned `VisionConfig` (pinned by the
  `audio_section_values_never_reach_the_tower` mutation test).
- `parse_lfm3_config` (the trunk parser, the FIRST gate every flow passes)
  and the vision parser call `refuse_lfm3_omni_d1`, for `lfm3_vl` and
  `lfm3_omni` alike. It refuses a config whose `architectures` names
  `D1Model` or whose `auto_map.AutoModel` ends with it (how mlx-vlm
  `utils.py` selects D1), or that carries any of the seven `d1_*` keys
  mlx-vlm `d1_lfm3_vl/config.py` defines (all optional there).
  `LiquidAI/d1-Omni-2B` declares the chat omni's `model_type` and trunk
  fields; without this check the probe would offer to install it as chat.
- `classify_for_family` (Lfm3 arm) drops the four audio prefixes by name
  into `ExcludedMultimodal`, and the streamed writer prints one progress
  line naming them when any were dropped, so an omni install SAYS its audio
  half is out rather than dropping 241 tensors in silence.

## Gates

**Probe (headers only).** `turbospark-model probe
LiquidAI/LFM3-Omni-2B-MLX-8bit` at the pinned revision: RUNNABLE,
`declares lfm3_omni`, `family lfm3`, MLX affine 8-bit group 64, 24 layers
2048 hidden, 2026-11-27.

**Install.** `turbospark-model pull --repo LiquidAI/LFM3-Omni-2B-MLX-8bit@5901d16c...`
succeeded: 2.1 GiB install, `packed_vision/blobs.bin` 338 MB (the same size
the VL install writes), the flattened `preprocessor_config.json` present,
and the manifest declaring the full tower (`visionTower lfm3_vl`, depth 14,
`visionStartTokenId 125009`, `visionEndTokenId 125010`,
`visionImageTokenId 125030`). A scan of the install's resident index found
zero audio, depth-decoder or codebook names; the tower and projector
residents are the eight `vision.*` entries the VL install carries.

**Text smoke.** `turbospark-check --model lfm3-omni-2b-8bit
--messages-file` with "What is 17 plus 26?": answers `43`, EndOfTurn
(M4 Max, greedy, 2026-11-27).

**Image smoke.** The deterministic page (`scripts/make_vision_test_page.py
--size 896 1152`, seed 11), prompt "Transcribe the first three lines":

```text
output: 00102 | parity streaming streaming oracle fixture tensor kernel oracle manifest gradient kernel expert
truth:  00012 | parity streaming streaming oracle fixture tensor kernel oracle manifest gradient kernel expert | 8316.48
```

All thirteen body words of line 1 in exact order, prefill 1,034 tokens (the
expanded image run included), stop at the model's own end of text turn
after one line at both 18 and 96 token budgets. The left-column id garbling
(`00102` for `00012`) is the SAME failure class the VL gate records on the
same generator; the body-word match is what a model that never saw the
picture cannot produce.

**Negative control.** The same prompt with no image: "The image provided
does not contain any text" (prefill 24 tokens, no tower run).

## Not done, and why

- **No tower-stage parity probe.** The tower is the ONE the VL Phase 0
  already gated against an independent numpy reference
  (`runtime/tests/lfm3_vl_vision_real.rs`), and the omni config declares the
  same `vision_config` and the same prefixes, so no second probe was run.
  What the omni gate adds is the config path, the intake walk and the
  end-to-end read, not new tower arithmetic.
- **No quality gate, memory oracle or server/FFI run.** The image smoke ran
  through the CLI only. The install is dense (no routed experts), so the
  expert-cache oracle does not apply; no perplexity gate was run.
- **No video.** The omni processor supports video spans
  (`<|vision_start|><|video|><|vision_end|>`); this port's Lfm3Vl path is
  stills only, as for VL.
- **Audio is out of scope and refused.** No LFM3-Audio encoder, talker,
  codec or depth-decoder path exists; the audio tensors are dropped at
  intake with a named report, and the audio sections in a config are checked
  and never ingested. The omni processor's additional stop ids
  (`<|audio_start|>`, `<|mixed_start|>`) are not wired, so a text
  generation could run past the point where the reference's plain-text
  generation stops; the smoke above hit EndOfTurn normally.
- **The d1 refusal is config-only.** A repo that stripped `D1Model` and
  every `d1_*` key from its config would resolve as a chat omni.
- **Speech placeholders are not suppressed.** The reference sets the logits
  of `<|audio|>` and `<|audio_out|>` to minus infinity on every step; this
  port does not, so a text turn can in principle sample them.
