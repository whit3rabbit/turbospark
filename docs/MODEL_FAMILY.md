# Supported Model Families & Architecture Detection

This document describes how `turbospark` detects, registers, and executes supported large language model families, how automatic architecture detection works during GGUF and Hugging Face downloads, and provides a parity comparison against upstream engines like `llama.cpp`, `mlx-lm`, and `turbo-fieldfare`.

---

## 1. Automatic Architecture Detection

When given a Hugging Face URL, local `.gturbo` directory, or GGUF checkpoint, `turbospark` detects the model architecture automatically before fetching large weight payloads.

**This is a command, not just an internal step.** `turbospark-model probe <repo>` runs the detection below against any Hugging Face repository, reading headers only, and reports what it concluded: the architecture verdict, each block type against the kernels that exist, the affine `(bits, group_size)` pair for an MLX checkpoint, the expert-slot arithmetic of section 4, and which tokenizer sidecars the repository has. It costs KB and seconds and exits 0 only if the checkpoint would run. See [`MODELS.md`](MODELS.md).

```
                    +--------------------------------+
                    |  GGUF / HF Checkpoint Download |
                    +---------------+----------------+
                                    |
                                    v
                +----------------------------------------+
                | Read Metadata Header                   |
                | - GGUF: `general.architecture`         |
                | - HF: `config.json -> model_type`      |
                +-------------------+--------------------+
                                    |
                                    v
            +------------------------------------------------+
            |  `family_for_architecture(arch)`               |
            |  Map string -> `ModelFamily` enum              |
            +-----------------------+------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
    [ Known Family ]                                [ Unknown Family ]
  - Parse `ArchConfig`                       - Reject with error:
  - Transcode & Stream to `.gturbo`            `UnsupportedArchitecture`
  - Execute via Metal shaders                - (See `docs/NEW_MODEL.md`)
```

### Detection Strategy (GGUF vs. Hugging Face)

Both tables live in one place, `crates/repack/src/arch_registry.rs`. **Every key
in it was read off a real published file**, and `tests/arch_registry_network.rs`
re-reads each one; a row without such a witness cannot be added. The three
naming schemes genuinely differ and none is derivable from another: Qwen 3.6 is
`qwen3_5_moe` in `config.json`, `qwen35moe` in GGUF, and `qwen36` in a
`.gturbo` manifest.

- **GGUF Checkpoints**: `turbospark-repack` fetches the initial ~512 KB metadata header via `HttpRangeSource` and inspects `general.architecture`:
  - `"gemma4"` -> `ModelFamily::Gemma4`
  - `"qwen35moe"` -> `ModelFamily::QwenGdnMoe`
  - `"llama"` -> `ModelFamily::Llama`, and partially: this string is both
    Mixtral and dense Llama, only the MoE half has a decode flow, and the
    refusal for the dense half therefore lives at `RealForwardRunner::open`
    rather than here (nothing in the string says which half a file is)
  - `"qwen35"` -> `ModelFamily::QwenGdnDense`, the dense sibling of
    `qwen35moe` and one suffix away from it -- exact equality is load-bearing
    (see the HF table below)
  - `"qwen3moe"` -> `ModelFamily::Qwen3Moe`, which runs through the same
    decode flow as `Llama`: the layer graph is identical, and the two
    differences (per-head q/k norms, RMS epsilon 1e-6 against 1e-5) are
    keyed on the family inside that flow rather than given a fourth copy
    of it
  - `"gpt-oss"` -> `ModelFamily::GptOss` (MXFP4 experts, attention sinks)
  - `"llama4"`, `"deepseek2"`, `"phi3"` are recognized but unported: the
    refusal names what each would need, and `tests/arch_registry_network.rs`
    re-reads every row's witness header so the string cannot rot silently
  - anything else -> refused as unknown. The audited candidate strings for
    future bring-ups are in ROADMAP.md section 13, not here.
- **Hugging Face Safetensors**: the family is chosen by the caller, which picks
  the per-family install writer (`write_gemma4_install`,
  `write_qwen_gdn_moe_install`, and one sibling per family below).
  `config.json`'s `model_type`
  is a guard on that choice rather than a dispatcher: each parser refuses a
  config that positively claims another family, since without it the wrong
  parser silently produces an `ArchConfig` labelled with the family it
  hardcodes. A config claiming nothing recognized is accepted.
  - `"gemma4"` / `"gemma4_text"` -> `ModelFamily::Gemma4`
  - `"qwen3_5_moe"` / `"qwen3_5_moe_text"` -> `ModelFamily::QwenGdnMoe`
  - `"qwen3_5"` / `"qwen3_5_text"` -> `ModelFamily::QwenGdnDense` (the dense
    sibling). **The lookup is exact equality and not a prefix match, and
    these two rows are why**: `qwen3_5` and `qwen3_5_moe` are one suffix
    apart, so a prefix match resolves every dense checkpoint to the MoE
    family -- a baseline with 256 experts and a decode flow with a router in
    it, i.e. fluent wrong output rather than an error. Three published
    checkpoints report `qwen3_5`: `prism-ml/Bonsai-27B-mlx-1bit`,
    `Qwen/Qwen3.8-27B` and `prism-ml/Ternary-Bonsai-27B-mlx-2bit`, at 1, 4
    and 2 bits, and they share one `ArchConfig` exactly. The last two needed
    no field, no kernel-independent change and no decode flow, only their
    own affine width.
  - `"muse_glimmer"` / `"muse_glimmer_text"` -> `ModelFamily::MuseGlimmer`
  - `"qwen4_exp"` / `"qwen4_exp_text"` -> `ModelFamily::Qwen4Exp`
  - The last two families are HF-ONLY in the registry: published GGUF
    conversions of both now exist (`muse-glimmer` and `qwen4exp`,
    witnessed off unsloth's conversions, 2026-09-06 -- note the third
    naming drift, HF underscores where GGUF hyphenates or drops the
    underscore), but the GGUF table above has no row for either, so
    `pull`ing those files refuses at the registry until rows land.
  - The `_text` spellings are what the multimodal checkpoints' `text_config`
    carries. `architectures` (class names like
    `Gemma4ForConditionalGeneration`) is deliberately not consulted: it is a
    fourth naming scheme and would need a fourth table to buy nothing.

---

## 2. Complete Model Family Parity Matrix

The table records what the code does TODAY: the nine `ModelFamily` variants
(eight running, one scaffolded), the three registered-but-unported strings,
and the closest comparisons across `llama.cpp`, `mlx-lm`, and
`turbo-fieldfare`. **The forward-looking family list is deliberately not this
page's job**: candidates, witnessed `general.architecture` strings, expert-slot
arithmetic, and bring-up notes live in ROADMAP.md section 13 (the oMLX and
Unsloth catalog audits, 2026-09-06). A family appears here when a
`ModelFamily` variant, a baseline and a decode flow do.

**Read the footprint column as an MoE result, not a general one.** The
~1.6-2.2 GiB figures come from streaming routed experts: only the resident core
is mapped, and its mapped weights are pinned by `newBufferWithBytesNoCopy`
(AGENTS.md Gotcha 19), so those rows are `resident core + KV + slot cache`.
The slot term is `slots x layers x expert_stride` and it dominates them, which
is why `qwen3moe` sits at 2.7 GiB and `gpt-oss` at 5.4 rather than inside the
band (Gotcha 36).

**The dense rows are low for an entirely different reason, and an earlier
version of this note had it backwards.** It said a dense family sits near its
own on-disk size "by construction", reasoning from Gotcha 19 that every mapped
byte is counted. Measured, that is false: a dense install's resident weights do
not appear in `phys_footprint` at all (AGENTS.md Gotcha 40, on Mistral 7B --
4.07 GiB of weights against a 684 MiB peak, agreeing on two independent
counters). Re-derived 2026-08-14 on a dense install four times that size,
Qwen3.8-27B: 15.1 GB of resident weights, 660 MiB of counted peak. So a dense
row here is KV plus whatever fixed per-layer state the architecture carries,
and it is the MoE rows that are large. Do not quote either number without the
context window, which is most of what a dense row asserts.

One row's architecture string is not what its name suggests, and it is measured
rather than assumed (`tests/arch_registry_network.rs`): **Mixtral reports
`general.architecture = "llama"`**, identical to a dense Llama 3.1, and
expresses its MoE through `llama.expert_count = 8`. The two halves need very
different work, so they are two rows below even though they are one string.
Rows marked *Registered, planned* have had their architecture string read
off a real published file and carry a row in `arch_registry.rs`. Every other
turbospark-column status on this table means the string is NOT in the
registry, whether or not it has been witnessed in the wild -- pointing a
checkpoint at one of those gets the "not in this port's registry" message
rather than the "recognized, needs X" one.

| Model Family / GGUF `general.architecture` | Key Architectural Features | `turbospark` (Rust) | `turbo-fieldfare` (Swift) | `llama.cpp` | `mlx-lm` | Peak RAM Footprint in `turbospark` |
| --- | --- | :---: | :---: | :---: | :---: | ---: |
| **Gemma 4 26B-A4B** (`gemma4`) | SWA/Full Attention, MoE (128 experts, top-8), Tied Embeddings | **Full Support** | **Full Support** | Full Support | Full Support | **~2.1 GiB RAM** |
| **Qwen 3.6 35B-A3B** (`qwen35moe`) | Gated-DeltaNet Linear Attention + MoE (256 experts, top-8) | **Full Support** | **Full Support** | Full Support | Full Support | **~1.6 GiB RAM** |
| **DeepSeek V3** (`deepseek2`, confirmed; the same string also reports Kimi K2.5/K2.6, GLM-4.7-Flash and Mistral-Large-3) | Multi-head Latent Attention (MLA), DeepSeek MoE | *Registered, planned* | *Planned* | Full Support | Full Support | *MoE, keeps the ceiling* |
| **DeepSeek V4 Flash / Pro** (`deepseek4`, witnessed 2026-09-06) | MLA, hyper connections, SWA (window 128), 256-384 experts top-6; the Flash variant carries a VISION tower | *Scaffolded* (`DeepseekV4Flash`) | *Scaffolded* | Full Support | Full Support | *TBD* |
| **Mixtral 8x7B / 8x22B** (`llama` + `expert_count`) | Plain GQA attention + MoE (8 experts, top-2), no shared expert, untied head | **Full Support** | *Planned* | Full Support | Full Support | *MoE, keeps the ceiling* |
| **Llama 2, Mistral 7B, TinyLlama** (`llama`, dense) | Standard Dense Transformer, GQA | **Full Support** (ROADMAP M4) | *Planned* | Full Support | Full Support | *dense: whole model resident* |
| **Qwen3-MoE 30B-A3B** (`qwen3moe`) | Plain GQA + per-head QK-norm, MoE (128 experts, top-8), no linear attention, no shared expert, untied head | **Full Support** | *Planned* | Full Support | Full Support | *MoE, keeps the ceiling* |
| **Qwen3.8-27B / Bonsai-27B / Ternary-Bonsai-27B** (`qwen3_5`, dense) | Gated-DeltaNet Linear Attention (48 of 64 layers) + DENSE SwiGLU FFN, packed q/gate, untied head | **Full Support** | *Not supported* | Full Support | Full Support | **~660 MiB RAM** (dense; see note) |
| **Qwen3.8-Flash-Next / REAP-288** (`qwen4_exp`, HF only) | Fine-grained MoE (288-512 experts, top-10), GDN + sigmoid-gated norm, QSA block-sparse attention, PLE n-gram head, hyper-connections | **Full Support** | *Planned* | Full Support (`qwen4exp`) | Full Support | **~2.5 GiB RAM** (oracle peak at the 2,048 bench window; the 68G install streams) |
| **Llama 3.1 / 3.2 / 3.3** (`llama`, dense) | The above plus LEARNED RoPE frequency scaling, which ships as a TENSOR (`rope_freqs.weight`) and has no kernel input here | *Refused at open, by name* | *Planned* | Full Support | Full Support | *dense: whole model resident* |
| **Llama 4 Scout / Maverick** (`llama4`) | MoE with interleaved chunked attention | *Registered, planned* | *Planned* | Full Support | Full Support | *MoE, keeps the ceiling* |
| **gpt-oss 20B / 120B** (`gpt-oss`) | MXFP4 experts, attention sinks, per-projection biases, YaRN, clamped SwiGLU | **Full Support** | *Planned* | Full Support | Full Support | *MoE at 12.6 MiB per expert; 20B keeps the ceiling at 4.73 GiB of slot cache, 120B does not stream usefully* |
| **Muse Glimmer 30B** (`muse_glimmer`, HF only) | Dense GQA, 3-sliding/1-full 2048 window, **NoPE on the full layers**, separate attention output gate, CENTERED per-layer norms against a PLAIN final one, TWO RMS epsilons, logit softcap behind an output multiplier | **Full Support** | *Not supported* | Full Support (`muse-glimmer`, published after this row was written) | Full Support (mlx-vlm) | **~535 MiB RAM** (dense at 8,192 context; see note) |
| **Phi-3 / Phi-3.5** (`phi3`; Phi-4 reports the same string) | SuScaled (longrope) RoPE, dense FFN | *Registered, planned* | *Planned* | Full Support | Full Support | *dense: whole model resident* |

Every family not named above is UNREGISTERED here and refused as unknown --
including Command-R, Grok, DBRX, StarCoder, Falcon, Baichuan, InternLM,
MiniCPM, OLMo, Exaone and the GPT-2/NeoX/MPT/Bloom legacy lines, which an
earlier version of this table carried as "*Planned*" without a registry row,
a baseline, or a roadmap entry behind them. The candidate list for future
bring-ups, each entry with its witnessed `general.architecture` string,
expert-slot arithmetic, and what it would need in this engine, is
[ROADMAP.md section 13](../ROADMAP.md) and is not duplicated here.

**Vision is per-tower, not per-family, so this page records it once**: the
engine runs exactly one vision tower (the `qwen3_5` tower, the Qwen3-VL-lineage
ViT with mRoPE), and only the `qwen3_5` family consumes it end to end --
[`docs/VISION.md`](VISION.md) is the home for how it works. A VLM checkpoint of
any other running family parses to a TEXT-only config today: Gemma 4's and
Muse Glimmer's vision tensors are dropped at repack, and "Full Support" above
therefore never means vision support. The per-model vision inventory -- which
vision-capable models `mlx-vlm` ships and which of them this port lacks --
lives in [ROADMAP.md section 14](../ROADMAP.md) and is not duplicated here.

---


## 3. How `llama.cpp` Handles Architectures vs. `turbospark`

`llama.cpp` handles architecture discovery using a centralized enum and dynamic graph construction:

1. **Architecture Registry (`enum llm_arch`)**: `llama.cpp` defines enum variants for every supported family (e.g. `LLM_ARCH_LLAMA`, `LLM_ARCH_GEMMA2`, `LLM_ARCH_GEMMA4`, `LLM_ARCH_QWEN35MOE`).
2. **Metadata Key Lookup**: GGUF files declare `general.architecture`. `llama.cpp` matches this string against its lookup table.
3. **Graph Builder**: `llama.cpp` constructs a C++ compute graph dynamically based on `llm_arch`.
4. **Hardcoded Fallbacks**: Because GGUF metadata omits hyper-parameter behavioral flags (like sandwich norms or attention scaling formulas), `llama.cpp` hardcodes these in its internal graph builder per `llm_arch`.

### How `turbospark` Implements This Strategy
`turbospark` follows a clean, strongly-typed Rust implementation of the same pattern:
- **Architecture Registry** (`crates/repack/src/arch_registry.rs`): the string tables, split into what runs and what is merely recognized. llama.cpp's `llm_arch` enum conflates the two because every variant it names has a graph builder; here they are separate, so a recognized-but-unported architecture is a better error rather than a half-wired family.
- **`ModelFamily` Enum** (`crates/model-io/src/arch_config/family.rs`): nine
  discriminators -- `Gemma4`, `QwenGdnMoe`, `Llama`, `Qwen3Moe`, `GptOss`,
  `QwenGdnDense`, `MuseGlimmer`, `Qwen4Exp`, and `DeepseekV4Flash`
  (declared, scaffolded, no baseline yet). The planned strings deliberately
  get NO variant: `known_architecture` is exhaustive and `arch_validation`
  compares its result field by field, so a placeholder would validate
  installs against invented numbers (`arch_registry.rs`'s own doc).

**One architecture string can cover two models, and support is then partial in a way no table column expresses.** `llama` is both Mixtral and dense Llama; only the MoE half has a decode flow, and nothing in the architecture string says which half a file is -- only `expert_count` does. So the registry calls `llama` supported, and `RealForwardRunner::open` refuses the dense half by name. A parity matrix row per MODEL rather than per string is the honest rendering, which is why the two rows above are split.
- **Baseline Specifications** (`crates/model-io/src/arch_baselines.rs`): Provides compile-time defaults for behavioral architecture flags missing from GGUF metadata.
- **Tensor Mapping Engine** (`crates/repack/src/gguf_names.rs`): Maps GGUF tensor naming conventions to canonical parameter names.
- **Dedicated Metal Forward Passes** (`crates/runtime/src/families/<family>/`): Each family owns an optimized Metal execution flow tuned for its layer graph; seven of them carry a chunked-prefill driver (all but the MoE half of `qwenGdnMoe`).

---

## 4. Extending Support to New Families

To add a new model family to `turbospark`:
1. Read the architecture string off a real published file and add a row to
   `crates/repack/src/arch_registry.rs` with the URL you read it from. That row
   is recognition only: it changes the error message and nothing else.
2. Register the new variant in `ModelFamily` (`crates/model-io/src/arch_config.rs`).
3. Add baseline specs in `arch_baselines.rs`.
4. Follow the 7-phase step-by-step checklist in [`docs/NEW_MODEL.md`](docs/NEW_MODEL.md).

Steps 2 and 3 belong to the bring-up, not to step 1. A `ModelFamily` variant
with an invented baseline is worse than no variant: `known_architecture` is
exhaustive and `arch_validation` compares its result against a manifest field by
field, so the placeholder would validate installs against fiction. Recognition
is keyed by string precisely so it can land without that risk.

**Order the families by what this engine is, not by llama.cpp's list.** The
memory result comes from streaming routed experts, so an MoE architecture reuses
the machinery that produces it, while a dense one gives up the ceiling and
competes with llama.cpp on ground where this port has no advantage. That is why
the first bring-up (ROADMAP Phase M2, complete) was the `llama` architecture's
MoE half (Mixtral) rather than its dense half, even though dense Llama is the
cheaper of the two. Which family to bring up NEXT is a ROADMAP.md question,
not a this-page question: section 13 carries the audited candidate list with
witnessed strings and slot arithmetic.

**Being MoE turned out to be necessary and not sufficient, and the correction
is what chose the family after it.** The slot cache is
`slots x layers x expert_stride`, so what decides whether a checkpoint streams
here is how finely it splits its experts, not how big it is: Mixtral 8x7B is 8
experts of 108.9 MiB and wants 54.5 GiB at 16 slots, while Gemma 4 is 128 of
~3.2 MiB and wants 1.5. Mixtral runs, is correct, and cannot stream usefully.
`qwen3moe` (Qwen3-30B-A3B) was picked next for exactly that reason -- 128
experts of 2.5 MiB, 1.90 GiB at 16 slots -- and it reuses the flow Mixtral's
bring-up wrote. Both numbers come off the GGUF header before any download
(AGENTS.md Gotcha 36, `docs/NEW_MODEL.md` Phase 0).

---

## 5. Document References

- Forward-looking family candidates, with witnessed strings and slot arithmetic: [ROADMAP.md section 13](../ROADMAP.md)
- Per-model vision inventory (which VLMs `mlx-vlm` supports and this port does not): [ROADMAP.md section 14](../ROADMAP.md); the one running tower's mechanics: [`docs/VISION.md`](VISION.md)
- Installing a model, and probing one that is not listed: [`docs/MODELS.md`](MODELS.md)
- Architecture Bring-up Guide: [`docs/NEW_MODEL.md`](docs/NEW_MODEL.md)
- `.gturbo` Format Specification: [`docs/GTURBO.md`](docs/GTURBO.md)
- Benchmark Parity & Measurements: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
