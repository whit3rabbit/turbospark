# Abliterated equivalents for the recommended models

This page is the research shortlist for offering pre-abliterated (uncensored)
checkpoints as alternatives to the curated catalog rows. It is the weight-side
companion to [`OBLITERATION.md`](OBLITERATION.md), which covers the
runtime-directional (control vector) form, and it follows the admission rule
in [`MODELS.md`](MODELS.md): a row exists only after the exact repository and
revision were streamed and run here. Nothing on this page has been streamed or
run by this engine yet. Everything here is a candidate, ranked and annotated
so the pull-and-gate pass has a queue to work from.

Research date: 2026-09-29, via the Hugging Face repository API (model search
plus per-repo file listings). Download counts are the API's rolling 30-day
figures, not lifetime totals. "Verified layout" below means the repository's
file list was read and the shard or GGUF structure confirmed; it is not an
engine gate.

## What the words mean (they are not the same method)

- **abliterated**: activation-ablation orthogonalization of refusal
  directions (huihui-ai is the highest-volume publisher; mlabonne originated
  the technique). Weights diverge from base only in the ablated projections.
- **Heretic**: GDM's memory-decoding-guided ablation. Different edit, same
  intent, usually branded "heretic" in the repo name.
- **Uncensored (HauhauCS and similar)**: behavior-patched or lightly tuned
  variants; marketed as "lossless". Treat as a fine-tune, not a pure edit.
- **Distill variants** (Claude/Gemini/Opus-named, e.g. the
  `Qwen3.6-...-Claude-4.7-Opus-abliterated` line): same architecture, but a
  different training trajectory. They run on the same family, but they are
  not an "equivalent" of the base row and should be labeled as distills.

The engine does not care which method produced the weights: an equivalent of
a checkpoint that already runs here keeps its family, its intake path, and
its expert granularity. That is the main argument for curating equivalents of
our own rows rather than whichever abliterated model is globally popular.

## The popularity landscape (and how much of it already runs here)

Ranked by 30-day downloads across the abliterated/uncensored search space:

| rank | repository | 30d dl | family here |
|---|---|---|---|
| 1 | `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF` | 2,064,412 | qwen35 (`qwen38-27b`) |
| 2 | `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF` | 1,482,951 | qwen35 (`qwen38-27b`) |
| 3 | `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` | 889,856 | qwen36 (`qwen36`) |
| 4 | `Bahushruth/Qwen3.6-35B-A3B-abliterated-v4` | 731,705 | qwen36 (`qwen36`) |
| 5 | `mradermacher/Qwen3-VL-8B-Instruct-abliterated-GGUF` | 662,637 | qwen3_vl (4B row runs; 8B not cataloged) |
| 6 | `DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-...-GGUF` | 600,128 | dense Qwen3.6-27B, not a catalog row; probe first |
| 7 | `LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-Final-GGUF` | 444,251 | qwen36 (`qwen36`) |
| 8 | `huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF` | 430,186 | DeepSeek-V4 is NOT deepseek2; new family work |
| 9 | `huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF` | 212,581 | qwen36, distill, MTP preserved |
| 10 | `wangzhang/gemma-4-31B-it-abliterated` | 308,570 | gemma4 family, 31B dense variant |
| 11 | `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF` | 166,886 | qwen4exp (`qwen4-reap288` base model) |
| 12 | `huihui-ai/Huihui-Qwen3.5-27B-abliterated` | 155,550 | qwen35 family, 27B sibling of our 3.8 row |
| 13 | `culturerevolt/gemma-4-12b-heretic-abliterated-GGUF` | 133,189 | gemma4 family, 12B variant |
| 14 | `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF` | 110,888 | qwen35 |
| 15 | `mradermacher/Huihui-Ornith-1.5-9B-abliterated-i1-GGUF` | 101,949 | qwen35 (`ornith9b` base) |
| 16 | `DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf` | 41,471 | gptOss (`gptoss-20b`) |

The top of the market is concentrated on exactly the checkpoints this engine
already serves. That is the opportunity: the five most popular abliterated
models in the wild are the bases (or siblings) of `qwen38-27b`, `qwen36`,
and `gptoss-20b`.

## Per-row equivalents

For each curated row: the best weight-side equivalent(s), the MLX-format
option if one exists, and the intake notes. "GGUF single-file" matters
because several families here refuse split GGUF; "sidecar" is the HF
repository a GGUF pull names for tokenizer files
([MODELS.md](MODELS.md#installing-something-not-in-the-catalog)).

### MoE rows (the priority)

| row (family) | equivalent | format | 30d dl | notes |
|---|---|---|---|---|
| `gemma4` (gemma4, 26B A4B) | `huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated-GGUF` via `groxaxo/Huihui-gemma-4-26B-A4B-it-abliterated-GGUF` | GGUF (mxfp4) | 26,339 | huihui also ships a qat-q4_0-unquantized GGUF (8,998 dl) |
| same | `vanch007/Huihui-gemma-4-26B-A4B-it-abliterated-mlx-4bit` | MLX 4-bit, 3 shards, 15.6 GB | 625 | verified layout; `mlx-community/...-qat-q4_0-unquantized-abliterated-4bit-msq` is the mlx-community conversion but uses the msq variant; leonsarmiento ships 3bit-XL |
| `qwen36` (qwen36, 35B A3B) | `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` | GGUF | 889,856 | the single most popular uncensored MoE; a fine-tune-style patch, not pure ablation |
| same | `froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit` | MLX 4-bit, 4 shards, 20.4 GB | 9,249 | verified layout, plain 4-bit (safest MLX pick), Apache-2.0 |
| same | `dawncr0w/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-OptiQ-4bit-MLX` (+ 5/6 bpw) | MLX OptiQ mixed | 1,457 | carries `optiq_metadata.json`; mixed per-layer widths need a probe before any claim |
| `qwen3moe` (qwen3moe, 30B A3B, 128 experts) | `huihui-ai/Qwen3-30B-A3B-abliterated` via `mradermacher/Qwen3-30B-A3B-abliterated-GGUF` | GGUF | 5,045 / 14,026 | exact base match; the Instruct-2507 refresh is 8x more popular (26,147) |
| same | `mlx-community/Josiefied-Qwen3-30B-A3B-abliterated-v2-4bit` (+ 6/8bit, bf16) | MLX | 418 | Josiefied fine-tune + abliteration, not the raw base; `Eric1227/Qwen3-30B-A3B-abliterated-MLX-8bit` (88) is closer to raw |
| `gptoss-20b` (gptOss) | `huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated` (v2 also published) via `DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf` (41,471 dl, 550 likes) or `bartowski/huihui-ai_Huihui-gpt-oss-20b-BF16-abliterated-GGUF` (8,156) | GGUF | 13,322 | GGUF is the proven intake here (the curated row is GGUF); the `mxfp4-abliterated-v2` source matches the row's quant family |
| same | `nightmedia/Huihui-gpt-oss-20b-mxfp4-abliterated-v2-qx86-hi-mlx` | MLX, 3 shards, 11.9 GB | 318 | verified layout but qx86 quant variant; gptOss MLX intake is unproven here; probe before trusting |
| `ornith35b` (qwen36 retrain) | `huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated` via `gbuzhf/Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF` (45,187) or `PocketAiHub/Ornith-1.5-35B-Abliterated-GGUF` (20,203) | GGUF | 3,092 | gbuzhf preserves MTP; APEX/UD naming means dynamic/importance-matrix quants, check block types |
| same | `PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-4bit` | MLX 4-bit, 4 shards, 20.4 GB | 6,068 | verified layout, MIT, exact-base MLX equivalent; shard sizes byte-match a plain 4-bit of the same shape |
| `dsv2lite-16b` (deepseek2) | `mradermacher/DeepSeek-Coder-V2-Lite-Instruct-abliterated-GGUF` | GGUF | 10,334 | Coder twin of the same deepseek2 architecture; the non-coder `DeepSeek-V2-Lite-Instruct-Abliterated-15-1.2` is 1,279 dl and a DavidAU-style merge; no MLX found |
| `qwen4-reap288` (qwen4exp) | `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF` | GGUF | 166,886 | the base model of the REAP row, unpruned, abliterated |
| same | `grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit` | MLX 4-bit, 23 shards, 114 GB | 1,386 | verified layout; full expert set (not REAP-pruned), so it leans hard on streaming; compare tok/s against the REAP row before promoting |
| `qwen4exp-swift-iq2-xs` (qwen4exp) | none found | - | - | only a NVFP4/NInfer release exists (`Dragoy/Swift-1.5-...`), no format this engine reads; use the Flash-Next equivalents above |

### Dense rows

| row (family) | equivalent | format | 30d dl | notes |
|---|---|---|---|---|
| `qwen38-27b` (qwen35) | `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF` | GGUF single-file, many quants | 2,064,412 | verified: Q3_K 13.5 GB, Q4_K 16.8 GB, Q8_0 29.1 GB, Unsloth UD/IQ sets, GSQ-RCO IQ3 with MTP, plus ternary Bonsai PTQ1_0/PQ2_0 quants; GGUF metadata arch is literally `qwen35`; a BF16 mmproj (931 MB) exists for vision pairing with the `qwen38-27b-vision` tower pattern |
| same | `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | MLX 4-bit | 2,303 | the safetensors source (`huihui-ai/Huihui-Qwen3.8-27B-abliterated`, 58,148 dl, 491 likes) is the high-signal artifact; MLX conversions at 2/3/4/6/8 bit from AutisticAF, ailexleon, EgorKodin (text-only), mlasli (heretic 6/8bit) |
| same (MTP) | `KostkaIT/Qwen3.8-27B-Huihui-Abliterated-oQ4e-MTP-MLX` (+ oQ6e) | MLX oQ4e | 1,697 | claims native MTP embedded in the shards; oQ4e is a variant quant, probe first; `soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` (43,934) is the GGUF MTP-preserving option |
| `ternary27b`, `bonsai2`, `bonsai27b` (qwen35 ternary) | `BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-GGUF` (57,970) and `...-PQ2_0-MTP-GGUF` (40,398); `Hikari07jp/Ternary-Bonsai-2-27B-Abliterated-GGUF` (30,869) | GGUF ternary | 57,970 | ternary GGUF block types (PTQ1_0/PQ2_0) have no proven kernels here; the curated ternary rows are MLX 2-bit, and the MLX abliterated pick is `KridgeDookie/...-PHILADELPHIA-CLASS-MLX-Mixed-2-8bit` (1,463, mixed-precision, probe). Note huihui's Qwen3.8-27B-abliterated GGUF repo above also ships Ternary-Bonsai quants of the abliterated weights |
| `qwen25-7b-4bit` (qwen2) | `huihui-ai/Huihui-7B...` line: `Huihui-Qwen2.5-7B-Instruct-abliterated-v2` via `mradermacher/...-v2-GGUF` (9,527) | GGUF | 7,011 | v1 GGUF 16,231 dl; the GGUF rows here already take Q3_K_M/Q4_K_M, sidecar from the huihui safetensors repo itself |
| same | `mlx-community/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2-4-bit` | MLX 4-bit | 325 | Josiefied fine-tune again; no plain-base MLX conversion found, one is a `mlx_lm.convert` away from the huihui source |
| `mistral7b` (llama) | `mradermacher/Mistral-7B-Instruct-v0.3-abliterated-GGUF` | GGUF | 1,375 | low demand; fine |
| `museglimmer` | `0bserverx/Muse-Glimmer-30B-Heretic-Uncensored-GGUF` | GGUF | 21,291 | no MLX abliterated exists; and museGlimmer GGUF intake is unproven (curated row is MLX) so this needs a probe first |
| `spark25` (spark2_5) | `InfinimindCreations/Spark-X2.5-4B-uncensored-GGUF` | GGUF | 4,997 | MLX abliterated exists (`mondk`, 1,887; `xunlinkx`, 2,090) but spark2_5 MLX intake is not built; the curated row is GGUF-only |
| `qwen3vl-4b` (qwen3_vl) | `huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated` via `noctrex/...-GGUF` (23,481) or `mradermacher` quants | safetensors / GGUF | 14,756 | no MLX conversion found; engine row is MLX, so either convert the huihui source with mlx-vlm or probe the GGUF path |
| `ornith9b` (qwen35) | `mradermacher/Huihui-Ornith-1.5-9B-abliterated-i1-GGUF` | GGUF | 101,949 | exact-base, GGUF intake already proven by the curated row |
| `tinyllama` (llama) | `mradermacher/tinyllama-1.1b-abliterated-GGUF` | GGUF | 1,035 | trivial, cheap smoke-test row for the whole uncensored path |
| `mixtral` (llama, caveat row) | none found | - | - | two searches returned zero; the row is a caveat row anyway (slot-cache refusal), so no loss |

Rows that are quant-duplicates of the same base (`gemma4-gguf`, `gemma4-iq3`,
`qwen36-gguf`, `qwen25-7b-q3km`, `qwen38-27b-mtp/-vision/-vision-tower`) are
covered by the equivalents of their base row above; in particular the
huihui GGUF repo's GSQ-RCO IQ3-with-MTP files and the BF16 mmproj map
one-to-one onto what `qwen38-27b-mtp` and `qwen38-27b-vision` do for the
censored base.

## Recommended shortlist

Tier 1, the ones worth gating first (popularity x exact-base fit x proven
intake path):

1. `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF` (qwen35, single-file GGUF
   quants, 2.06M dl). Pull Q4_K or Q3_K first; sidecar is the huihui
   safetensors repo. The MLX 4-bit conversions follow once the GGUF row is
   green.
2. `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` GGUF (qwen36,
   890K dl) plus `froggeric/...-Heretic-MLX-4bit` as the plain-4bit MLX arm
   of the same A/B.
3. `vanch007/Huihui-gemma-4-26B-A4B-it-abliterated-mlx-4bit` (gemma4, MLX,
   exact base of the "install this one" row).
4. `DavidAU` or `bartowski` GGUF of `Huihui-gpt-oss-20b-BF16-abliterated`
   (gptOss, proven GGUF intake).
5. `PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-4bit` (exact-base MLX of
   `ornith35b`).

Tier 2, after Tier 1 is green: Qwen3-30B-A3B-abliterated (GGUF, sidecar
huihui), the Qwen3.8-Flash-Next abliterated pair (167K-dl GGUF; the 114 GB
grant-ai MLX needs a streaming-cost comparison against `qwen4-reap288`),
Muse-Glimmer and Spark-X2.5 uncensored GGUFs (each needs its family's GGUF
intake probed), and TinyLlama as the cheap pathfinder.

## Integration plan

Follow [MODELS.md](MODELS.md) exactly; nothing here relaxes it:

1. `turbospark-model probe <repo>` first. It settles the open format
   questions this research could not from metadata alone: OptiQ/oQ4e/qx86
   mixed-width MLX layers, UD/APEX/IQ GGUF block types, ternary GGUF blocks,
   and the museGlimmer/spark GGUF arch strings.
2. `pull --repo ... --alias ... --sidecar-repo ...` (GGUF) or plain
   `--repo` (MLX repos carry their own tokenizer). Pin a commit sha at pull
   time; this page deliberately records no shas because they would be stale
   by the time a row is added.
3. Generate with it, then set status honestly. `verified` needs a frozen
   row in [`BENCHMARKS.md`](BENCHMARKS.md). An abliteration changes behavior,
   so the quality gate must be re-run on the abliterated weights; do not
   inherit the base row's frozen numbers, and expect some perplexity drift.
4. Remember the pull cannot resume ([MODELS.md](MODELS.md#before-anything-else-a-pull-cannot-resume)).
   The Tier 1 set is 13 to 21 GB per arm; queue them one at a time.
5. MoE equivalents keep their base row's expert granularity, so the slot
   cache arithmetic is unchanged (`gemma4` ~3.3 GiB, `qwen36` ~2.2 GiB,
   `qwen3moe` 128 x 2.5 MiB streaming, Flash-Next full = heavy streaming at
   114 GB). Dense equivalents stream nothing; budget disk size as free RAM.
6. Desktop surfacing: once rows exist they appear in Model Hub automatically
   through the recommend ranker. An "uncensored alternative" affordance
   should pair each base row with its equivalent row by explicit catalog
   alias, never by name matching, and should label the method (abliterated /
   heretic / uncensored-patch / distill) because the behavior claims differ.
7. Users who want this before curation can add the repos as local rows via
   `$TURBOSPARK_HOME/models.json` ([MODELS.md](MODELS.md#local-rows-without-a-rebuild)),
   after running probe themselves.

## Relationship to runtime steering

These offers are weights-side and the steering pane is runtime-side, and
they must not silently combine: [OBLITERATION.md](OBLITERATION.md) already
warns that applying a refusal direction to a model that was already
abliterated can remove useful behavior or collapse generation. Concretely:

- A curated abliterated row should open with steering available (it is the
  same family) but the app should warn if a refusal-direction preset is
  enabled on an abliterated install.
- A curated offer of direction files for the censored bases (the OBLITERATION
  "future curated catalog") and a curated offer of abliterated weights are
  alternatives for the same user goal; the weight-side one is what the
  ecosystem's popularity lives on, the runtime one is reversible and does
  not double disk usage. Both belong in the catalog eventually, clearly
  separated.

## Gaps and negative findings

- No abliterated Mixtral-8x7B exists in current search results (two queries,
  zero hits). The caveat row stays censored.
- No abliterated Swift-1.5 build in any format this engine reads (NVFP4
  only).
- No MLX abliterated for: Muse-Glimmer-30B, Qwen3-VL-4B, gpt-oss-20b in a
  plain (non-qx86) quant, or a plain-base Qwen2.5-7B (Josiefied only). All
  are one mlx-vlm/`mlx_lm` conversion away from a safetensors source that
  exists, which is an option if curation wants to own the conversion.
- No REAP-pruned abliterated Flash-Next; the grant-ai MLX is the full expert
  set and pays for it in streamed bytes.
- Several high-download artifacts are adjacent but out of scope until family
  work exists: DeepSeek-V4-Flash abliterated (430K dl, new architecture),
  the dense Qwen3.6-27B uncensored family (600K+ dl combined, no catalog
  row), and Qwen3-VL-8B abliterated (663K dl, family runs at 4B).
