# Idefics3 and SmolVLM vision: Phase 0 and bring-up record

One port covers both wrappers: mlx-vlm's SmolVLM is an Idefics3 subclass that
reuses its tower, connector and processing. The port was
built from published metadata and reference code against synthetic fixtures,
then smoked once on real weights (SmolVLM-256M, see "Real install" under
Gates). The Idefics3-8B wrapper itself has not been run.

## Sources

| What | Where |
| --- | --- |
| Config, processor files, index and safetensors headers | `mlx-community/SmolVLM-256M-Instruct-4bit`, `SmolVLM-Instruct-4bit`, `Idefics3-8B-Llama3-4bit` (read at `main`, 2026-11-27; not pinned) |
| Config and index only | `mlx-community/SmolVLM-500M-Instruct-4bit`, `SmolVLM-256M-Instruct-bf16` |
| Tower, connector, prompt layout | mlx-vlm `mlx_vlm/models/idefics3/{vision,idefics3,processing_idefics3}.py` |
| Image processor, tower epsilons, position rule | transformers v4.51.3 `models/idefics3/{image_processing,modeling}_idefics3.py` |

## What the checkpoints are

- Root `model_type` `idefics3`, `text_config.model_type` `llama`. The
  published SmolVLM builds say `idefics3` too. The llama parser reads
  `text_config` for these two wrapper strings and no other.
- Trunk names are `language_model.layers.*` (no `model.` level);
  `canonicalize_idefics3_header` adds it. `lm_head` is already canonical.
- Tower `vision_model.*` with separate q/k/v, projector
  `connector.modality_projection.proj.weight`, no connector bias or norm.
- 4-bit affine at group 64. The SmolVLM builds quantize the tower matrices and
  the position table too (SmolVLM-Instruct leaves `fc2` unquantized);
  Idefics3-8B leaves the tower in FP16. The reader dequantizes to FP16.

| Build | Tower | Tile / patch | Position rows | Shuffle | Tokens per frame | Longest edge |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SmolVLM-256M / 500M | 768 x 12 | 512 / 16 | 1024 | 4 | 64 | 2048 |
| SmolVLM-Instruct | 1152 x 27 | 384 / 14 | 729 | 3 | 81 | 1920 |
| Idefics3-8B | 1152 x 27 | 364 / 14 | 676 | 2 | 169 | 1456 |

384 is not a multiple of 14: the unpadded patch conv drops 6 pixels.

## Decisions

- **Position ids are the identity.** The reference buckets fractional
  coordinates over each frame's valid patches. Its processor emits only full
  square tiles, where the bucketing returns `0..side` (checked in float32
  arithmetic for sides 26, 27 and 32), so no bucketing code exists.
- **Post-norm epsilon is 1e-6**, as HF builds it. mlx-vlm leaves it at 1e-5,
  so an mlx-vlm tower-stage comparison will differ slightly there.
- **Text turns are wrapped as one text part.** The chat template indexes
  `message['content'][0]['type']` and renders a bare string as nothing.
- **`<end_of_utterance>` is added to the stop set.** It is the processor's
  extra EOS; no sidecar lists it.
- **Separator ids are tokenizer-resolved.** `<global-img>` and the row tags are
  single tokens on SmolVLM-256M and BPE pieces on the other two.

## Wired

Probe and streamed MLX install (4-bit or 8-bit conversions), manifest
(`VisionTowerKind::SigLipNaViT`), tiling and prompt expansion
(`crates/vision-io/src/idefics3.rs`), the tower
(`crates/runtime/src/vision/idefics3.rs`, no new kernel), CLI
`run --image`, and FFI `ts_generate` image parts.

## Refused

- Images through `turbospark serve` and the FFI server: no splice there.
- Vision sidecars (`pull-vision`, `--vision-sidecar`).
- Quantize-on-install for these wrappers.
- `use_resampler`, a connector with extra tensors, any other `hidden_act`,
  `layer_norm_eps` or resample filter.
- GGUF vision (`mmproj`) for these models.

## Gates

Synthetic only: config and reader tests (including packed planes), header
rename and bucketing, the resize arithmetic, tile order, shuffle and prompt
layout, the template and stop set, and
`crates/runtime/tests/idefics3_vision_synthetic.rs`, which runs the real
repack walk and open and compares the GPU tower with an independent CPU
forward pass over the source tensors (cosine 0.9999997).

### Real install (SmolVLM-256M, 2026-11-27)

`mlx-community/SmolVLM-256M-Instruct-4bit` (`main` resolved to `69cb5195`),
release CLI, greedy: declares `idefics3`, family `llama`, 138.7 MiB download,
270.7 MiB install. Text answers Jupiter (prefill 18). A large-text page
(`MARIGOLD` / `HARBOR 4821` / `GATE PAGE` / `magic word:` / `zephyr`) reads
back as `MARGINO HARBOR 4821 GATE PAGE: magic word: zephyr` (prefill 1145).
With no image the model invents unrelated prose with none of the page's
strings (prefill 17). One page, one size, one checkpoint; the install was
removed afterwards.

## Not verified

- Idefics3-8B was never downloaded: no install or smoke for it. No quality or
  memory gate on any checkpoint.
- No pixel parity against HF's processor (the resize arithmetic was
  transcribed; the Lanczos filter is the crate's existing PIL port).
- No tower comparison against mlx-vlm or transformers.
- Tokenizer loading and dialect detection on the 8B's real `tokenizer.json`
  (Llama 3 is inferred from its added tokens).
- Memory: all frames of an image run in one pass. SmolVLM-Instruct can reach
  26 frames of 729 rows, several hundred MiB of tower scratch that the load
  guard does not budget.
