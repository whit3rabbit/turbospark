# Hugging Face text-model coverage

Snapshot date: 2026-11-18. This page measures how much of the Hugging Face
text-generation list this port can install and run, and ranks the gaps that
fit on a Mac. It is a coverage map, not a support claim: a row counts as
"runs" only when its family, format and quant type are on the executable paths
recorded in [MODEL_FAMILY.md](MODEL_FAMILY.md) and [GGUF.md](GGUF.md). No row
here was re-gated on hardware for this snapshot.

## How the snapshot was taken

Lists (top 100 each, `pipeline_tag=text-generation`):

```text
https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=100&expand[]=config&expand[]=safetensors&expand[]=gguf&expand[]=tags
https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trendingScore&direction=-1&limit=100&expand[]=config&expand[]=safetensors&expand[]=gguf&expand[]=tags
```

- Family comes from `config.model_type`, or `gguf.architecture` for GGUF-only
  repos (which publish no `config.json`).
- Size comes from `safetensors.total` or `gguf.total` (parameters).
- HF quant method comes from `config.quantization_config.quant_method`.
- GGUF quant types come from the `.gguf` file names in each repo
  (`/api/models/<id>`, `siblings`).
- Support status is read from `crates/repack/src/arch_registry.rs`
  (`SUPPORTED_HF`, `SUPPORTED_GGUF`, `PLANNED_GGUF`),
  `crates/repack/src/quantize_on_install.rs` and
  `crates/repack/src/gguf_header/ggml.rs`.

## Summary, top 100 by downloads

| Bucket | Count | Examples |
| --- | --- | --- |
| Runs today (HF via quantize-on-install, GGUF, or MLX) | ~39 | Qwen3 0.6B-32B, Qwen2.5 0.5B-14B and Coder, Ornith `qwen35`/`qwen35moe` GGUF, Qwen3.8-27B GGUF finetunes, Bonsai MLX, Spark GGUF, Gemma 4 GGUF |
| Family runs, only from another format | ~7 | gpt-oss 20B/120B (GGUF only), Qwen3-30B-A3B and Qwen3-Coder-30B-A3B (HF `qwen3_moe` unregistered, GGUF runs), Mistral-7B, TinyLlama, dolphin-yi (GGUF only) |
| Supported family, unsupported HF quant | ~16 | AWQ (9), NVFP4 modelopt (6), FP8, bitsandbytes |
| Unsupported family | ~38 | Llama 3.x HF (7), gpt2/opt/pythia (6), Nemotron-H (6), DeepSeek V3/R1/V3.2/V4 (6), GLM 4.5/4.7/5/5.3 (5), LFM2 (3), MiniMax M2.5, Hy3 |

GGUF quant types are effectively fully covered. Every type found in the 53
GGUF repos on both lists (Q2_K to Q8_0, IQ1_S to IQ4_XS, the Unsloth UD-*
mixes, TQ1_0, TQ2_0, Q2_0, NVFP4, BF16, F16, F32) is executable. The quant gaps
are all on the HF side: `quantization_config` is never read, so AWQ, GPTQ,
NVFP4 (modelopt), FP8, compressed-tensors and bitsandbytes checkpoints fall
into the unquantized-safetensors refusal.

## Missing coverage that fits a Mac

Fit estimate: about 0.55 bytes per parameter at 4 bits, plus KV and runtime
overhead. Tiers: 16 GB holds up to about 14 B parameters, 36 GB up to about
50 B, and 64-128 GB up to about 200 B (MoE with expert streaming).

### A. Supported family, missing intake or rope form

These reuse an existing forward pass.

| Models | Params | Tier | Missing | Reuse |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B(-Instruct), Llama-3.2-1B/3B(-Instruct) | 1.2-8 B | 16 GB | llama3 rope scaling (`rope_freqs.weight` refused by name) and HF `llama` intake | `rope_neox_freqs` kernel, the Qwen2 HF path |
| Meta-Llama-3-8B-Instruct | 8 B | 16 GB | HF `llama` intake (no rope scaling) | same |
| SmolLM2-135M(-Instruct), MiniCPM5-2B, TinyLlama HF | 0.1-2.5 B | 16 GB | HF `llama` intake | same |
| Mistral-7B-Instruct-v0.2 HF | 7.2 B | 16 GB | HF `mistral` intake | same |
| Nanbeige4.3-33B, dolphin-2.9.1-yi-1.5-34b | 33-34 B | 36 GB | HF `llama` intake; check `rope_scaling` and eps | same |
| Qwen3-30B-A3B, Qwen3-Coder-30B-A3B HF | 30.5 B | 36 GB | HF `qwen3_moe` row and MoE quantize-on-install | GGUF `qwen3moe` flow |
| gpt-oss-20b HF (MXFP4 safetensors) | 20.9 B | 36 GB | HF `gpt_oss` intake mapping `*_blocks`/`*_scales` to the MXFP4 tag | gpt-oss flow, MXFP4 expert kernels |
| granite-4.1-3b/8b, TwIL-8B HF | 3.7-8.8 B | 16 GB | HF `granite` intake (GGUF runs) | llama flow |
| Spark-X2.6-4B HF | 4.1 B | 16 GB | HF intake (ROADMAP P2.5 descope) | spark flow |
| Whittle-Next-27B-A3B (`qwen4_exp_text`) | 27.6 B | 36 GB | Qwen4Exp not admitted to quantize-on-install | qwen4 flow |
| Nemotron-3.5-Bolt-E4B GGUF (`gemma4`, per-layer embeddings) | 7.5 B | 16 GB | Unverified: PLE is refused on the HF path; the GGUF path is unprobed | gemma4 flow |
| Gemma 3 4B/12B/27B GGUF | 4-27 B | 16-36 GB | linear rope scaling refused (`gguf_config/mod.rs`) | `rope_neox_freqs` (linear scaling is a constant factor table) |

### B. Supported family, HF quantization format not read

One intake feature per row covers every model in it.

| Format | Models | Params | Reuse |
| --- | --- | --- | --- |
| AWQ / GPTQ int4 g128 | Qwen2.5-3B/7B/14B/32B(-Coder)-AWQ, Qwen3-8B/14B-AWQ | 3.4-33 B | g128 INT4 kernels (in progress on another branch), or dequantize then quantize-on-install |
| NVFP4 (modelopt) | Qwen3.6-35B-A3B, Qwen3.8-27B, Ornith-1.5-9B and 35B-A3B, Gemma-4-26B-A4B and 31B | 6.7-21 B | GGUF NVFP4 kernels, if the block and scale layout matches (unverified) |
| FP8 (e4m3, block scales) | Qwen3-4B-Instruct-2507-FP8, Qwen3-Coder-30B-A3B-FP8 | 4.4-30.5 B | dequantize at install, then quantize-on-install |
| compressed-tensors, bitsandbytes | OTel-2.0 26B-A4B 4bit, OTel 31B QLoRA | 26-31 B | low priority |

### C. New family that fits a Mac

| Family | Models | Params | Tier | Closest existing path | Work |
| --- | --- | --- | --- | --- | --- |
| `lfm2` / `lfm2moe` | LFM2.5-24B-A2B (HF and 3 GGUF repos) | 24 B, 2 B active | 36 GB | lfm3 short-conv state plus MoE routing. A different layer graph from lfm3 (no SWA, no sandwich norms, no output gate) | medium |
| `nemotron_h` (Mamba2 hybrid) | Nemotron-3-Nano-4B, lumi-4b, Nano-30B-A3B, 3.5-Lightning-30B-A3B | 4-32 B | 16-36 GB | GDN recurrent state in the qwen flow; the scan differs | large |
| `glm4_moe_lite` | GLM-4.7-Flash and finetunes | 31 B, 3 B active | 36 GB | deepseek2 MLA flow plus the four ROADMAP P2.4 descopes | medium-large |
| `gemma2` | gemma-2-9b-it | 9.2 B | 16 GB | gemma4 dense (softcap, alternating SWA) | medium |
| `maxwell_moe` | Maxwell-V1-Flash (HF and GGUF) | 36 B | 36 GB | needs a Phase 0 | unknown |
| `raon1_nano` | Raon-1-nano (HF and GGUF) | 3.1-3.5 B | 16 GB | needs a Phase 0 | unknown |
| `dflash` drafts | GLM-5.3-Flash-Maya, Qwen3.8-27B-DSpine-b16 | 1.2-2 B | 16 GB | speculative-decode drafts for targets that run here | medium, speed only |
| one-offs | embla (7 B MLX), lara_mini (7.3 B), dory (4 B), k2-horizon (9 B GGUF), aion_instruct (2.3 B), strata-ple (51 B) | 2-51 B | varies | each needs a Phase 0 | low reach each |

### D. Fits only large-memory Macs (64-128 GB)

- gpt-oss-120b (117 B, MXFP4). The GGUF route already runs; HF intake is the A-row gap.
- GLM-4.5-Air (`glm4_moe`, 110 B).
- Nemotron-3-Super-120B-A12B (`nemotron_h`).
- Ornith-1.5-122B GGUF: supported, memory is the only limit.

### Out of reach on any Mac

DeepSeek V3, R1, V3.2, V4 and V4.1 (290-760 B), GLM-5 and 5.3 (753 B),
Kimi K3.1 (2.8 T), Hy3 and Hy4, MiniMax-M2.5 (229 B FP8), ERNIE-5.5,
Marin-Mochi-535B, Qwen3.8-2.4T, Nemotron-3-Ultra.

### Not worth closing

`gpt2`, `opt`, `gpt_neox` (pythia): their download counts are dominated by CI
and test traffic, not chat use.

## Status of gap 1 (Llama 3), 2026-11-18

Runs and has been checked against two other engines on the same bytes, but
has no frozen memory or quality rows yet (the numbers are in the Llama 3.1
row of [MODEL_FAMILY.md](MODEL_FAMILY.md)):

- GGUF Llama 3.x installs: `rope_freqs.weight` is read by value into
  `arch.ropeFreqFactors` and bound as a `rope_neox_freqs` angle table. These
  installs are written at manifest minor 2, so an older reader refuses them
  by version instead of decoding them with unscaled frequencies.
- MLX-quantized `llama` and `mistral` safetensors (mlx-community conversions)
  install through the shared walk. `rope_scaling.rope_type: llama3` builds
  the same divisor vector.

Still refused, by design:

- BF16 HF Llama and Mistral repos. Quantize-on-install admits a family only
  after a parity gate against a published MLX conversion, and Llama has none
  yet.
- `openbmb/MiniCPM5-2B`: its `model_type` is `llama` with
  `rms_norm_eps: 1e-6`. The llama flow hardcodes 1e-5 for this family, so
  the install is refused rather than run with the wrong epsilon.
- Mistral v0.1 (live `sliding_window`), attention or MLP biases, and
  `rope_scaling` kinds other than `llama3` (linear, dynamic, yarn, longrope).

Found and fixed while doing this:

- GGUF intake left a `llama` install's `attention_scale` at the Mixtral
  baseline's 128^-0.5, so every 64-wide head attended at 0.71x the right
  scale. That covers Llama 3.2 1B, SmolLM2 and TinyLlama. Position 0 was exact
  and every later position was off: TinyLlama Q6_K read 0.339 nats from
  llama.cpp (perplexity 17.61 against 12.52) before the fix and 6.0e-6
  (12.519) after (`docs/verification/tinyllama-q6k-attention-scale-*.json`).
  TinyLlama's frozen quality row (17.6084 and its two digests) was recorded
  under this bug. It was re-frozen on 2026-11-18 at 12.5191
  (`crates/bench/tests/tinyllama_quality_gate.rs` records the evidence).

Found while doing this, not fixed:

- `gguf_config` reads YaRN for every family, but the llama flow (Qwen2,
  Qwen3, MiniMax, dense Llama) never applies `rope_scaling`. A GGUF of one of
  those families that declares YaRN installs and decodes with unscaled
  frequencies.
- `qwen2_config.rs` ignores `rope_scaling` and hardcodes none. This is the
  same silent class on the HF path.

## Ranked order (reach against cost)

1. Llama 3 rope scaling plus HF `llama`/`mistral` intake. About 11 entries
   across both lists, no new kernel.
2. HF `quantization_config` intake (NVFP4, FP8, then AWQ/GPTQ). About 16
   entries, all alternate builds of families that run.
3. HF `qwen3_moe` intake.
4. `lfm2` / `lfm2moe`.
5. `nemotron_h`.
6. GLM-4.7-Flash.
