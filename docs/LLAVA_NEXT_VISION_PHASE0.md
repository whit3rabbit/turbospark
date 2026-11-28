# LLaVA 1.6 (LLaVA-NeXT) vision: Phase 0 and what is wired

`model_type llava_next`: the LLaVA-1.5 CLIP tower and projector
([LLAVA_VISION_PHASE0.md](LLAVA_VISION_PHASE0.md)), a learned `image_newline`,
and an anyres image layout. Only what differs from 1.5 is on this page.

**Status: wired and tested on synthetic weights only.** No real checkpoint has
been downloaded or run, so there is no parity, quality or memory result.

## Published forms

The route is 1.5's: a GGUF `llama` trunk plus a `pull-vision` tower sidecar from
the `llava-hf` safetensors repository. Facts below were read from configs,
indexes and ranged header reads.

| Role | Pin | Size |
| --- | --- | --- |
| Trunk | `second-state/Llava-v1.6-Vicuna-7B-GGUF@42e50c599382343d2ba2899f1ca86022f4ca9469`, `llava-v1.6-vicuna-7b-Q4_K_M.gguf` | 4,081,004,352 B |
| Tower, tokenizer, template | `llava-hf/llava-v1.6-vicuna-7b-hf@c916e6cdcd760b4cecd1dd4907f84ac649f93b23` | tower up to 648,982,528 B ranged from shard 1; tokenizer.json 1.8 MB |
| Mistral trunk | `cjpais/llava-1.6-mistral-7b-gguf@6019df415777605a8364e2668aa08b7e354bf0ba`, `llava-v1.6-mistral-7b.Q4_K_M.gguf` | 4,368,439,552 B |
| Mistral tower | `llava-hf/llava-v1.6-mistral-7b-hf@2424fdd47412fccc66d91719126b420e9fbd7065` | same shapes |

Both trunks declare `general.architecture llama`, 32 blocks, hidden 4096, Q4_K
and Q6_K blocks, and 32,000 embedding rows against `<image>` = 32000 (1.5's
fill-id rule applies). Shard 1 holds all 396 non-language tensors, F16.

```sh
turbospark-model pull-vision --repo \
  llava-hf/llava-v1.6-vicuna-7b-hf@c916e6cdcd760b4cecd1dd4907f84ac649f93b23 \
  --alias llava16-vision-tower
turbospark-model pull --repo second-state/Llava-v1.6-Vicuna-7B-GGUF \
  --file llava-v1.6-vicuna-7b-Q4_K_M.gguf \
  --sidecar-repo llava-hf/llava-v1.6-vicuna-7b-hf@c916e6cdcd760b4cecd1dd4907f84ac649f93b23 \
  --alias llava16-7b
turbospark-check --model llava16-7b --vision-sidecar <tower dir> \
  --messages-file msgs.json --image shapes.png --temperature 0 --top-k 1
```

## The rule

The reference is `transformers` 4.57.6 (`LlavaNextImageProcessor`,
`pack_image_features`), read and not executed. mlx-vlm's
`models/llava_next/llava_next.py` does **no** anyres packing and was not
followed.

- Base tile: the image squashed to 336 x 336 (not 1.5's center crop).
- Grid: `select_best_resolution` over the five pinpoints (336x672, 672x336,
  672x672, 1008x336, 336x1008, as height x width), an aspect-kept resize
  (`ceil`), centred on zero raw pixels, cut into 336 tiles row-major.
- Tower and projector: 1.5's, once per tile (23 blocks, quick-GELU, eps 1e-5,
  class row dropped, exact-GELU projector).
- Run: the base tile's 576 rows with no newline, then the grid map unpadded to
  the image's aspect, with `image_newline` after every row. No pooling (the
  difference from OneVision's `anyres_max_9`).
- Tokens per image = 576 + H x W + H. At most 2928.

`vision-io/src/llava_next.rs` shares the grid choice, the canvas and the unpad
rule with `onevision.rs`.

## Guards

- A tower kind of its own, `VisionTowerKind::ClipNext` (`clip_next`): 1.5 and
  1.6 have the same patch size and position table. The runtime also refuses
  anyres patches on a 1.5 tower and untiled patches on a 1.6 one, so a
  mismatched `preprocessor_config.json` cannot run.
- `LlavaNextImageProcessor` configs previously parsed as the 1.5 crop; they now
  have their own layout.
- `--vision-sidecar auto` never attaches a 1.6 tower. Pairing is still only
  `(family, hidden_size)`, so **an explicit path attaches it to any 4096-wide
  `llama` trunk** (Mistral 7B, a 1.5 trunk, the other 1.6 trunk). Nothing
  detects that.
- Refused by name at intake: a non llama/mistral `text_config.model_type`
  (the Qwen LLaVA-NeXT builds), a tile other than 336, a disabled newline.

## Prompts

Both templates (`chat_template.json`) read `content` only as a part list, so a
plain-string turn rendered empty. The CLI, FFI and server set
`MfTokenizer::set_string_content_as_parts` when a 1.6 tower is attached.
Neither template emits `<s>`: image turns get it from `clip_prepend_bos`;
**text-only turns run without BOS**, as on 1.5. Without the sidecar attached,
the trunk still renders string turns empty.

## Tests

Synthetic, no real weights:

- `vision-io` unit tests: processor config, token counts against a python
  transcription of the processor's formula, base squash and pad, packing order.
- `repack` unit tests: config and reader, including that a 1.5 walk still
  refuses `image_newline`.
- `catalog/tests/llava_next_sidecar.rs`: `pull-vision` against a loopback
  server, `auto`, and a 1.5 install beside it.
- `runtime/tests/llava_next_vision_synthetic.rs`: GPU tower and packing against
  a CPU computation, batched prefill injection, and the 1.5 path.
- `server` and `ffi` `tests/llava_next_vision_synthetic.rs`: text and image
  turns end to end.

## Not done or unverified

- No real-weights run: parity against `transformers`, output quality, memory
  and speed are all unmeasured. The resize is not compared against PIL here.
- The CLI `--image` path compiles (it is the CLIP arm plus the template flag)
  and has not been run on a 1.6 install.
- The Swift app attaches sidecars (`visionSidecar` `auto`, added 2026-11-28) but
  has not been run with this tower. No catalog rows.
- 1008-wide or tall grids and multiple images have not been run end to end.
- A 2928-token image does not fit twice in the Vicuna trunk's 4096 context.
- The 34B and Llama-3 builds were not checked.
