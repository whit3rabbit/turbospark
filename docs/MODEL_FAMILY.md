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
  - `"qwen3"` -> `ModelFamily::Qwen3Dense` (per-head Q/K norms and dense
    FFN in the shared Llama flow; the pinned 0.6B Q8_0 checkpoint passes
    real smoke, memory, and quality checks, see the
    [regression record](MINIMAX_M2_PHASE0.md#shared-flow-regression-checks))
  - `"qwen2"` -> `ModelFamily::Qwen2Dense` (standard full-attention GQA
    with Q/K/V projection biases; the dense flow carries the Qwen2 RMS
    epsilon and refuses Qwen2 sliding-window configs)
  - `"gpt-oss"` -> `ModelFamily::GptOss` (MXFP4 experts, attention sinks)
  - `"spark2_5"` -> `ModelFamily::Spark25` (fused QKV, per-class RoPE,
    headwise output gate; GGUF intake, HF safetensors intake deferred)
  - `"minimax-m2"` -> `ModelFamily::MiniMaxM2` (GGUF text execution implemented;
    real-checkpoint gates pending, see [Phase 0](MINIMAX_M2_PHASE0.md))
  - `"llama4"`, `"deepseek2"`, `"phi3"` are recognized but unported: the
    refusal names what each would need, and `tests/arch_registry_network.rs`
    re-reads every row's witness header so the string cannot rot silently
  - anything else -> refused as unknown. The audited candidate strings for
    future bring-ups are in section 2's mlx-lm census below, not here.
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
  - `"qwen2"` -> `ModelFamily::Qwen2Dense` (Qwen2/Qwen2.5 dense models;
    the MLX source namespace is normalized from `model.*` to the canonical
    `language_model.*` install namespace)
  - The last two families are HF-ONLY in the registry: published GGUF
    conversions of both now exist (`muse-glimmer` and `qwen4exp`,
    witnessed off unsloth's conversions, 2026-09-06 -- note the third
    naming drift, HF underscores where GGUF hyphenates or drops the
    underscore), but the GGUF table above has no row for either, so
    `pull`ing those files refuses at the registry until rows land.
  - `"minimax_m2"` -> `ModelFamily::MiniMaxM2` for recognition and foreign-config
    rejection; safetensors intake remains explicitly refused.
  - The `_text` spellings are what the multimodal checkpoints' `text_config`
    carries. `architectures` (class names like
    `Gemma4ForConditionalGeneration`) is deliberately not consulted: it is a
    fourth naming scheme and would need a fourth table to buy nothing.

---

## 2. Complete Model Family Parity Matrix

The table records completed family bring-ups, scaffolded support,
registered-but-unported strings, and the closest comparisons across
`llama.cpp`, `mlx-lm`, and
`turbo-fieldfare`. **The forward-looking family list is deliberately not this
page's job**: bring-up scoping and ordering are ROADMAP priority questions.
What this page does carry is STATUS: the census subsection below folds in the
witnessed-string inventory of the 2026-09-06 oMLX and Unsloth catalog audits
(pruned from ROADMAP.md on 2026-09-07) and extends it with a 2026-09-08
enumeration of mlx-lm's own model registry. A family appears here when a
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
off a real published file and carry a row in `arch_registry.rs`. Implemented
families are registered too. An unregistered candidate gets the "not in this
port's registry" message; a registered but unported family gets the more
specific "recognized, needs X" refusal.

| Model Family / GGUF `general.architecture` | Key Architectural Features | `turbospark` (Rust) | `turbo-fieldfare` (Swift) | `llama.cpp` | `mlx-lm` | Peak RAM Footprint in `turbospark` |
| --- | --- | :---: | :---: | :---: | :---: | ---: |
| **Gemma 4 26B-A4B** (`gemma4`) | SWA/Full Attention, MoE (128 experts, top-8), Tied Embeddings | **Full Support** | **Full Support** | Full Support | Full Support | **~2.1 GiB RAM** |
| **Qwen 3.6 35B-A3B** (`qwen35moe`) | Gated-DeltaNet Linear Attention + MoE (256 experts, top-8) | **Full Support** | **Full Support** | Full Support | Full Support | **~1.6 GiB RAM** |
| **DeepSeek V3** (`deepseek2`, confirmed; the same string also reports Kimi K2.5/K2.6, GLM-4.7-Flash and Mistral-Large-3) | Multi-head Latent Attention (MLA), DeepSeek MoE | *Registered, planned* | *Planned* | Full Support | Full Support | *MoE, keeps the ceiling* |
| **DeepSeek V4 Flash / Pro** (`deepseek4`, witnessed 2026-09-06) | MLA, hyper connections, SWA (window 128), 256-384 experts top-6; the Flash variant carries a VISION tower | *Scaffolded* (`DeepseekV4Flash`) | *Scaffolded* | Full Support | Not supported (absent from mlx-lm, checked 2026-09-08) | *TBD* |
| **Mixtral 8x7B / 8x22B** (`llama` + `expert_count`) | Plain GQA attention + MoE (8 experts, top-2), no shared expert, untied head | **Full Support** | *Planned* | Full Support | Full Support | *MoE, keeps the ceiling* |
| **Llama 2, Mistral 7B, TinyLlama** (`llama`, dense) | Standard Dense Transformer, GQA | **Full Support** (ROADMAP M4) | *Planned* | Full Support | Full Support | *dense: whole model resident* |
| **Qwen3-MoE 30B-A3B** (`qwen3moe`) | Plain GQA + per-head QK-norm, MoE (128 experts, top-8), no linear attention, no shared expert, untied head | **Full Support** | *Planned* | Full Support | Full Support | *MoE, keeps the ceiling* |
| **Qwen3 dense** (`qwen3`) | Plain GQA, per-head Q/K norm, dense SwiGLU, tied or untied head | GGUF implemented; 0.6B Q8_0 gates passed | Not assessed | [Implemented](https://github.com/ggml-org/llama.cpp/blob/e5a8d439cef31f27fad6938233da10dae1ba5631/src/models/qwen3.cpp) | [Implemented](https://github.com/ml-explore/mlx-lm/blob/745352405f0909540760fd9b9ff16d933fd9c82b/mlx_lm/models/qwen3.py) | [0.6B measurement only](MINIMAX_M2_PHASE0.md#shared-flow-regression-checks); dense |
| **Qwen2 / Qwen2.5 dense** (`qwen2`) | Standard full-attention GQA, Q/K/V projection biases, no Q/K norm, dense SwiGLU, Qwen2 RMS epsilon 1e-6 | GGUF + HF intake and shared-flow execution implemented; synthetic GGUF/HF gates passed. Q4_K/Q8_0 paths are runnable; the pinned Q3_K_M artifact still needs a Q3_K resident kernel | Not assessed | Full Support | Full Support | *real artifact gate pending; dense* |
| **Qwen3.8-27B / Bonsai-27B / Ternary-Bonsai-27B** (`qwen3_5`, dense) | Gated-DeltaNet Linear Attention (48 of 64 layers) + DENSE SwiGLU FFN, packed q/gate, untied head | **Full Support** | *Not supported* | Full Support | Full Support | **~660 MiB RAM** (dense; see note) |
| **Qwen3.8-Flash-Next / REAP-288** (`qwen4_exp`, HF only) | Fine-grained MoE (288-512 experts, top-10), GDN + sigmoid-gated norm, QSA block-sparse attention, PLE n-gram head, hyper-connections | **Full Support** | *Planned* | Full Support (`qwen4exp`) | Not supported (absent from mlx-lm, checked 2026-09-08) | **~2.5 GiB RAM** (oracle peak at the 2,048 bench window; the 68G install streams) |
| **Llama 3.1 / 3.2 / 3.3** (`llama`, dense) | The above plus LEARNED RoPE frequency scaling, which ships as a TENSOR (`rope_freqs.weight`) and has no kernel input here | *Refused at open, by name* | *Planned* | Full Support | Full Support | *dense: whole model resident* |
| **Llama 4 Scout / Maverick** (`llama4`) | MoE with interleaved chunked attention | *Registered, planned* | *Planned* | Full Support | Full Support | *MoE, keeps the ceiling* |
| **gpt-oss 20B / 120B** (`gpt-oss`) | MXFP4 experts, attention sinks, per-projection biases, YaRN, clamped SwiGLU | **Full Support** | *Planned* | Full Support | Full Support | *MoE at 12.6 MiB per expert; 20B keeps the ceiling at 4.73 GiB of slot cache, 120B does not stream usefully* |
| **Muse Glimmer 30B** (`muse_glimmer`, HF only) | Dense GQA, 3-sliding/1-full 2048 window, **NoPE on the full layers**, separate attention output gate, CENTERED per-layer norms against a PLAIN final one, TWO RMS epsilons, logit softcap behind an output multiplier | **Full Support** | *Not supported* | Full Support (`muse-glimmer`, published after this row was written) | Full Support (mlx-vlm) | **~535 MiB RAM** (dense at 8,192 context; see note) |
| **Spark-X2.5-4B** (`spark2_5`) | Dense GQA, the muse window at 512 (3 sliding : 1 full), **per-class RoPE** (full: theta 5e6 over the leading quarter; SWA: theta 1e4 whole head), fused `q_k_v_proj`, **headwise scalar sigmoid output gate**, exact-erf GELU, tied head, 1M trained context | **Full Support** (GGUF intake; HF safetensors intake deferred) | *Not supported* | Full Support (upstream PR 27868, 2026-09-06; this port mirrors its conventions) | Full Support (community MLX conversions) | *dense at 8,192 context; oracle passed. See [measured peak and KV verification](TRUBOQUANT.md#spark-real-install-probe)* |
| **MiniMax-M2** (`minimax-m2`) | Full GQA, whole-projection Q/K RMS norm, 64-of-128 RoPE, 256 experts top-8, sigmoid router with selection-only bias | *Implemented; low-temperature smoke fails* ([record](MINIMAX_M2_PHASE0.md)) | Not assessed | Implemented | Implemented | Oracle passed at 8192 context / 8 slots; no frozen baseline ([record](MINIMAX_M2_PHASE0.md)) |
| **Phi-3 / Phi-3.5** (`phi3`; Phi-4 reports the same string) | SuScaled (longrope) RoPE, dense FFN | *Registered, planned* | *Planned* | Full Support | Full Support | *dense: whole model resident* |

**Execution support and verification status are separate.** The P0 verification
follow-up completed Spark's KV-quantization fixture suite and real-model oracle,
plus the remaining real-install probes and CLI smokes. It also recorded open
gpt-oss KV4 greedy-completion, museGlimmer FP16 sampled-golden, and Qwen4 KV4
sampled-answer findings. "Full Support" denotes the implemented execution
path, not a clean result on every quality check. Read
[TurboQuant verification](TRUBOQUANT.md#p0-verification-follow-up-2026-09-09)
for those findings and framing limits, and
[Qwen4 phase measurements](QWEN4_EXP.md#p0-phase-follow-up-2026-09-09-pread-measured-speedup-inconclusive)
for the unresolved throughput effect. Frozen benchmark windows and baselines
are unchanged.

The registry is authoritative for evolving bring-ups; a working-tree enum or
registry addition alone does not establish a completed parity-matrix row.
Families outside both this matrix and the registry are refused as unknown --
including Command-R, Grok, DBRX, StarCoder, Falcon, Baichuan, InternLM,
MiniCPM, OLMo, Exaone and the GPT-2/NeoX/MPT/Bloom legacy lines, which an
earlier version of this table carried as "*Planned*" without a registry row,
a baseline, or a roadmap entry behind them. The full census of what mlx-lm
runs that this port does not -- grouped by registry status and witness,
with provenance on every entry -- is the subsection below. Bring-up SCOPING
(what to build next, and in what order) stays a ROADMAP priority question
and is deliberately not answered on this page.

### The mlx-lm model census, 2026-09-08

The systematic version of the paragraph above: every model implementation
in mlx-lm's `mlx_lm/models/` directory, enumerated 2026-09-08 (129 files,
of which 11 are shared modules -- `base`, `cache`, `mla`, `ssm`,
`gated_delta`, `rope_utils`, `switch_layers`, `activations`,
`bitlinear_layers`, `pipeline`, `__init__` -- leaving 118 model files),
each classified against this repo's registry and the 2026-09-06 oMLX and
Unsloth catalog audits. mlx-lm additionally carries a `MODEL_REMAPPING`
dict in `mlx_lm/utils.py`; a remap counts as support there, and is named
below where it matters. Re-running the census is one upstream directory
listing plus, for any candidate worth scoping, one `turbospark-model
probe` against a real repo.

**Being in mlx-lm's list is not a witness for this registry.** The
admission rule above (every key read off a real published file) is
unchanged. What the census buys is the shape of the gap: how many
families a peer engine runs that this port refuses, which of them already
carry a witnessed string, and which run here but not there. The Unsloth
audit's strings ARE witnesses in the registry's own sense -- read off
real published files, re-derivable in seconds with the probe -- so the
GGUF-witnessed group below is row-eligible, not merely plausible.

| Verdict | Families (mlx-lm file names, minus `.py`) |
| --- | --- |
| Running in BOTH engines | `gemma4` / `gemma4_text`, `qwen3_5`, `qwen3_5_moe`, `qwen3` (GGUF only here; 0.6B Q8_0 verified), `qwen3_moe`, `gpt_oss`, `muse_glimmer`, `llama` / `mixtral` -- with the two splits the matrix rows above record (Llama 3.1+ refused on `rope_freqs.weight`; Mixtral runs on the GGUF path only, no HF writer exists here) |
| Running HERE, ABSENT from mlx-lm | `qwen4_exp` (running here) and `deepseek4` (scaffolded here): no model file and no remap entry in mlx-lm on 2026-09-08. The other direction is format-level, not a family: this port reads GGUF natively, mlx-lm reads MLX safetensors only |
| Implemented here, real-checkpoint gates pending | `minimax` (GGUF `minimax-m2`; HF `minimax_m2` intake deferred; [record](MINIMAX_M2_PHASE0.md)) |
| Registered, planned here; running there | `phi3` (the row's string also covers Phi-4, witnessed), `llama4` / `llama4_text`, `deepseek2` (mlx-lm's `deepseek_v2` / `deepseek_v3`; the audit's highest-leverage unlock -- one MLA bring-up covers Kimi K2.5/K2.6, GLM-4.7-Flash and Mistral-Large-3, and mlx-lm's own `kimi_k2 -> deepseek_v3` remap corroborates the shape), `kimi_k25` (mlx-lm ships a native file; here it reports the witnessed `deepseek2` string and is unported) |
| GGUF-witnessed, unregistered here (Unsloth audit) | `qwen2_moe` (the MoE line remains out of scope), `gemma3` / `gemma3_text` / `gemma2` / `gemma3n`, `qwen3_next` (`qwen3next`, 512 top-10 GDN -- its OWN string despite sharing `qwen36`'s layer graph), `glm4_moe` (`glm4moe`), `glm_moe_dsa` (`glm-dsa`, the GLM-5 DSA line), `nemotron_h` (`nemotron_h_moe`, 128 top-6 up to 512 top-22), `hunyuan` / `hunyuan_v1_dense` (`hunyuan-moe`), `ernie4_5` / `ernie4_5_moe` (`ernie4_5-moe`), `mistral3` / `ministral3` (`mistral3` -- dense, and a separate string from `llama`, so Devstral Small 2 needs its own row despite the name) |
| Audit-noted model_type, no GGUF witness here | `deepseek_v32` (DeepSeek V3.2 -- rides omlx's `glm_moe_dsa` patch; needs MLA latent KV plus a token-level sparse indexer, where `qwen4_exp`'s QSA indexes blocks), `bailing_moe` / `bailing_moe_linear` / `bailing_moe_v3` (the Ling line; omlx audits `bailing_hybrid`, Ling 3.0 Flash -- MLA and KDA in one model), `laguna`, `longcat_flash` / `longcat_flash_ngram`, `mimo` / `mimo_v2_flash`, `step3p5` (omlx notes `step3p7`, the same vendor line) |
| Refused by measurement, not by absence | `kimi_k3`: witnessed `kimi-k3` -- 896 experts top-16, about 727M parameters per expert, roughly 7x Mixtral's blob -- so Gotcha 36's header arithmetic says unstreamable here at any legal slot count; recorded so nobody re-derives it after a download |
| In mlx-lm, no witness here, unregistered (refused as unknown) | `Klear`, `afm7`, `afmoe`, `apertus`, `baichuan_m1`, `bitnet`, `cohere` (Command-R), `cohere2`, `dbrx`, `deepseek` (V1), `dots1`, `exaone` / `exaone4` / `exaone_moe`, `falcon_h1`, `gear`, `glm` / `glm4` / `glm4_moe_lite`, `gpt2`, `gpt_bigcode`, `gpt_neox`, `gptj`, `granite` / `granitemoe` / `granitemoehybrid`, `helium`, `internlm2` / `internlm3`, `iquestloopcoder`, `jamba`, `kimi_linear`, `lfm2` / `lfm2_moe` / `lfm2-vl`, `lille-130m`, `mamba` / `mamba2`, `mellum`, `minicpm` / `minicpm3`, `nanbeige`, `nanochat`, `nemotron` / `nemotron-nas`, `olmo` / `olmo2` / `olmo3` / `olmoe`, `openelm`, `phi` / `phi3small` / `phimoe` / `phixtral`, `plamo` / `plamo2` / `plamo3`, `qwen` (Qwen 1), `recurrent_gemma`, `rwkv7`, `seed_oss`, `smollm3`, `solar_open`, `stablelm`, `starcoder2`, `talkie`, `telechat3`, `youtu_llm` |

The vision-capable files (`qwen2_vl`, `qwen3_vl`, `qwen3_vl_moe`,
`kimi_vl`, `lfm2-vl`, `pixtral`) are tower gaps, not text-family gaps --
see the vision note below. Grok is absent from mlx-lm's directory on this
date but keeps its Unsloth-witnessed `grok` string (Grok-2 270B, 8
experts top-2 at ffn 32768: the Mixtral-unstreamable shape at 4x the
size). `diffusion-gemma` is likewise witnessed upstream and out of class
here: a block-diffusion generation LOOP, not a decode flow.

Two structural readings fall out of the census:

- **Dense families remain a substantial gap.** Dense `qwen3` now has a
  GGUF execution path and a verified 0.6B Q8_0 regression checkpoint, and
  dense Qwen2/Qwen2.5 now has both GGUF and MLX/HF intake through the shared
  Llama flow. Real Qwen2.5 artifact smokes, memory, quality, and catalog
  promotion remain open; dense Gemma (`gemma3`/`gemma2`/`gemma3n`) remains
  unported here, and `deepseek2` remains the multi-model MLA candidate.
  MiniMax-M2's top-8-of-256 structure motivated its streaming bring-up;
  that structure alone makes no measured footprint or throughput claim.
- **A class this engine has no machinery for at all.** The recurrent and
  hybrid lines -- `mamba`, `mamba2`, `falcon_h1`, `jamba`, `nemotron_h`,
  `recurrent_gemma`, `rwkv7`, `bailing_moe_linear`, `kimi_linear` (KDA)
  -- need recurrent state machinery beside the GDN state manager, not a
  new attention kernel; this port has none of it today.

**Vision is per-tower, not per-family, so this page records it once**: the
engine runs exactly one vision tower (the `qwen3_5` tower, the Qwen3-VL-lineage
ViT with mRoPE), and only the `qwen3_5` family consumes it end to end --
[`docs/VISION.md`](VISION.md) is the home for how it works. A VLM checkpoint of
any other running family parses to a TEXT-only config today: Gemma 4's and
Muse Glimmer's vision tensors are dropped at repack, and "Full Support" above
therefore never means vision support. The per-model vision inventory -- which
vision-capable models `mlx-vlm` ships and which of them this port lacks --
was recorded in the 2026-09-06 mlx-vlm audit; ROADMAP.md pruned it on
2026-09-07 and it is recoverable from git history
(`git show 3d58b83^:ROADMAP.md`, section 14). Its headline for this page:
this port runs exactly one tower, and the nearest-term vision gap is Gemma
4 vision (`gemma4_unified`), whose text half already runs here and whose
~815 vision tensors are dropped at repack today.

---


MiniMax-M2 now has implemented split-GGUF intake and a text execution path.
The pinned three-shard Q4_K_M install is complete. Synthetic Metal inference,
a short-answer EOS check, and the real memory oracle pass. The 400-token
low-temperature smokes repeat planning and truncate; quality and shared-flow
regression gates remain pending. It has no catalog row or frozen performance
baseline. [Phase 0](MINIMAX_M2_PHASE0.md) records the pinned contracts,
storage calculations, tokenizer handling, and gate results. HF/MLX intake
and native tool-call parsing remain deferred.

## 3. How `llama.cpp` Handles Architectures vs. `turbospark`

`llama.cpp` handles architecture discovery using a centralized enum and dynamic graph construction:

1. **Architecture Registry (`enum llm_arch`)**: `llama.cpp` defines enum variants for every supported family (e.g. `LLM_ARCH_LLAMA`, `LLM_ARCH_GEMMA2`, `LLM_ARCH_GEMMA4`, `LLM_ARCH_QWEN35MOE`).
2. **Metadata Key Lookup**: GGUF files declare `general.architecture`. `llama.cpp` matches this string against its lookup table.
3. **Graph Builder**: `llama.cpp` constructs a C++ compute graph dynamically based on `llm_arch`.
4. **Hardcoded Fallbacks**: Because GGUF metadata omits hyper-parameter behavioral flags (like sandwich norms or attention scaling formulas), `llama.cpp` hardcodes these in its internal graph builder per `llm_arch`.

### How `turbospark` Implements This Strategy
`turbospark` follows a clean, strongly-typed Rust implementation of the same pattern:
- **Architecture Registry** (`crates/repack/src/arch_registry.rs`): the string tables, split into what runs and what is merely recognized. llama.cpp's `llm_arch` enum conflates the two because every variant it names has a graph builder; here they are separate, so a recognized-but-unported architecture is a better error rather than a half-wired family.
- **`ModelFamily` Enum** (`crates/model-io/src/arch_config/family.rs`):
  `ModelFamily::ALL` is the authoritative discriminator list, including
  `Spark25` for `spark2_5`. `DeepseekV4Flash` remains declared and scaffolded,
  without a baseline. The planned strings deliberately
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
not a this-page question: the census in section 2 carries the audited
candidates with their witnessed strings, and the slot arithmetic is the
Phase 0 multiplication of Gotcha 36.

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

- The mlx-lm model census (2026-09-08) and the witnessed candidate strings: section 2 above
- Per-model vision inventory: the 2026-09-06 mlx-vlm audit, pruned from ROADMAP.md on 2026-09-07 (`git show 3d58b83^:ROADMAP.md`, section 14); the one running tower's mechanics: [`docs/VISION.md`](VISION.md)
- Installing a model, and probing one that is not listed: [`docs/MODELS.md`](MODELS.md)
- Architecture Bring-up Guide: [`docs/NEW_MODEL.md`](docs/NEW_MODEL.md)
- `.gturbo` Format Specification: [`docs/GTURBO.md`](docs/GTURBO.md)
- Benchmark Parity & Measurements: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
