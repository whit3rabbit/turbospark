# Z-Image-Turbo Phase 0 evidence

Status: IG0 resource evidence remains open. Native IG1 parity is closed for the
available fixtures: the full-width native checkpoint block, complete nine-step
DiT gate, and real 1024-by-1024 VAE decode gate pass their frozen contracts.
The optional raw-pixel arrays are absent from this checkout, so the VAE test's
conditional pixel comparisons were not exercised. This page records measured
facts and unresolved gates for
[the image-generation design](IMAGE_GENERATION.md). It does not establish
a supported RAM minimum, general image-quality guarantee, or production disk schema.
The next work is the remaining IG0 resource closure. IG2 production runtime
work follows that evidence, and app work remains IG4 after IG3 proves bounded
lifetimes; neither a CLI nor app launch belongs to the current evidence gate.

## Inputs and reproducibility

| Input | Immutable revision | License |
| --- | --- | --- |
| Tongyi-MAI/Z-Image-Turbo | `f332072aa78be7aecdf3ee76d5c247082da564a6` | Apache-2.0, model-card declaration |
| huggingface/diffusers | `a71e62e0d226c284b86abf518791a5ffbba064bf` | Apache-2.0, pinned LICENSE |
| mflux-community/mflux | `051ba9ff25c9a8a8703356053a012a0dfad3fe39` | MIT, pinned LICENSE |

[Input evidence](verification/z-image-ig0-inputs.json) records all component
configs, 1,163 tensor names/shapes/dtypes/offsets, shard payload sizes and
published SHA-256s, tokenizer asset hashes, and reference source hashes.
The probe validates shard indexes against actual safetensors headers and
rejects range responses that would download unbounded weights. Published
payload hashes are distinguished from locally verified payload hashes.
Stage captures verify the complete component payload before loading it.
All seven downloaded shards now pass local SHA-256 verification; see
[the download receipt](verification/z-image-ig0-download.json).

The transformer, text encoder, VAE, scheduler, tokenizer, and pipeline index
are all required inputs. There is no separate CLIP or image encoder in this
text-to-image pipeline. Text embedding/output weights are tied; a second
LM-head payload is not required. The canonical VAE includes encoder tensors,
although v1 text-to-image execution only needs its decoder.

Python 3.13.15 was used. The isolated reference dependency lock is
[`z_image_reference_requirements.txt`](../scripts/z_image_reference_requirements.txt).
It pins Diffusers by commit and every installed dependency, including Torch,
Transformers, NumPy, MLX, and safetensors. MFLUX block sources are imported
directly from the pinned, hash-checked cache. Unrelated package initializers
are bypassed; the block implementations themselves are unchanged.

## Component contract

### Conditioning

- Qwen3: 36 blocks, hidden size 2560, FFN 9728, 32 Q heads and eight KV
  heads, head width 128, vocabulary 151936. RMS epsilon is 1e-6 and RoPE
  theta is 1,000,000. Attention is causal with a padding mask.
- Apply the checkpoint chat template to one user message with
  `add_generation_prompt=True`, `enable_thinking=True`. Pad and truncate
  to 512 tokens. Extract `hidden_states[-2]`, then remove padding rows.
  This is the state after block 35 of 36, before the final block and final
  normalization. Do not substitute final normalized text-generation states.
- The base Qwen model suffices: logits and generation KV caching are unused.
  The capture explicitly disables cache and avoids the vocabulary head.
- MFLUX's pinned text attention constructs Q/K RMSNorm without an epsilon
  argument, so MLX 0.32.2 supplies 1e-5. Canonical Qwen3 and the pinned
  Transformers encoder use the checkpoint's 1e-6. Do not treat MFLUX text
  conditioning as a numerically identical oracle without resolving this
  difference. The native contract follows the canonical checkpoint.
- Seven prompt cases are captured. Empty input still contains eight framing
  tokens; Unicode contains 19; the overlong case truncates 2,409 tokens to
  512. The four image-review cases cover composition, typography, detail,
  and lighting. Token IDs and masks are exact fixtures, not handpicked IDs.
- Native Rust FP32 CPU forward pass (in `crates/image`) confirms exact framing
  string parity across all seven prompt cases, and exact token IDs and
  attention mask agreement against captured arrays.
- Native text encoder forward parity against captured BF16 MPS conditioning:
  - lighting (27 tokens): max absolute error 1.2142e2, relative L2 8.6892e-3
  - empty (8 tokens): max absolute error 1.2142e2, relative L2 8.8581e-3
  - unicode (19 tokens): max absolute error 1.2142e2, relative L2 8.7833e-3
  Achieved relative L2 error is consistently 8.69e-3 to 8.86e-3 (~0.88%),
  reflecting FP32 CPU accumulation versus BF16 MPS capture across 35 layers.
  Freeze text encoder tolerance: relative L2 <= 0.015 for Rust FP32 CPU vs
  captured BF16 MPS. This tolerance is measured, not inherited from bounded
  FP32 cross-engine comparisons.
- Native scheduler parity (`crates/image::scheduler`): exact schedule bit-parity
  for (1,1), (8,8), (9,9) against contracts JSON, exact timesteps/sigmas capture
  match, and Euler step parity across latent_00..08 + final_latents within 1e-6
  float roundoff. All six native tests are mutation-checked in
  `docs/verification/z-image-ig1-mutations.json`.
- The native crate now contains the staged FP32 DiT reference, including exact
  image/caption sequence construction, learned padding tokens that remain
  attendable, two noise-refiner blocks, two context-refiner blocks, thirty main
  blocks, final unpatchification, and the pinned negative output convention.
  The canonical 1024-by-1024 gate uses `[1,16,128,128]` initial and final
  latents, not 64-by-64 latents. This code is not evidence of full parity until
  the opt-in nine-step checkpoint gate completes.
- All nine captured-input updates pass the unchanged local scheduler relative-L2
  ceiling of `0.02`. Their transformer-output and scheduler-output relative-L2
  values, followed by the accumulated rollout values, are:

  | Update | Input | Expected | Transformer | Captured-input scheduler | Accumulated rollout |
  | ---: | --- | --- | ---: | ---: | ---: |
  | 1 | `initial_noise.npy` | `latent_00.npy` | `3.08770984e-2` | `2.02384288e-3` | `2.02384288e-3` |
  | 2 | `latent_00.npy` | `latent_01.npy` | `2.65911520e-2` | `1.93704967e-3` | `1.12695470e-2` |
  | 3 | `latent_01.npy` | `latent_02.npy` | `2.09559463e-2` | `1.79356441e-3` | `2.42157504e-2` |
  | 4 | `latent_02.npy` | `latent_03.npy` | `1.61662959e-2` | `1.73750182e-3` | `3.88680734e-2` |
  | 5 | `latent_03.npy` | `latent_04.npy` | `1.57005340e-2` | `2.15785531e-3` | `5.96708842e-2` |
  | 6 | `latent_04.npy` | `latent_05.npy` | `1.11899236e-2` | `1.98123301e-3` | `8.85860100e-2` |
  | 7 | `latent_05.npy` | `latent_06.npy` | `1.08153522e-2` | `2.44240882e-3` | `1.24803871e-1` |
  | 8 | `latent_06.npy` | `latent_07.npy` | `8.89623817e-3` | `2.47731130e-3` | `1.63017213e-1` |
  | 9 | `latent_07.npy` | `latent_08.npy` | `9.89344344e-3` | `3.13403807e-3` | `1.95015728e-1` |

  The maximum accumulated error is `1.95015728e-1`. Rounding that maximum
  upward to the next `0.001` freezes the cumulative BF16 envelope at `0.196`.
  The local scheduler maximum is only `3.13403807e-3`, so the curve identifies
  accumulated FP32 CPU versus BF16 MPS state drift rather than a bad local
  timestep. `latent_08.npy` and `final_latents.npy` are byte-identical capture
  aliases, not a tenth scheduler transition.
- Native VAE output conversion maps finite decoded `[3,H,W]` floats from
  `[-1,1]` to interleaved RGB8 and validates a decodable PNG. The real
  1024-by-1024 Rust decode gate passes and produces `[3,1024,1024]`; its
  optional raw-pixel comparison files are absent from this checkout, so only
  the real decode and geometry assertions ran.

### Diffusion transformer

- 30 main blocks, two noise-refiner blocks, two context-refiner blocks;
  hidden size 3840, 30 Q/K/V heads of 128, FFN width 10240. Conditioning
  width is 2560. RMS epsilon is 1e-5; attention scale is `128^-0.5`.
- Latents have 16 channels. Spatial patches are 2 by 2, temporal patch size
  is one. For input `(C,F,H,W)`, reshape to
  `(C,Ft,pF,Ht,pH,Wt,pW)`, permute to `(Ft,Ht,Wt,pF,pH,pW,C)`, then flatten.
  The inverse must restore this ordering, not a channel-first patch vector.
- Pad caption and image sequences separately to multiples of 32. Padding
  positions use learned pad tokens after embedding. Real batch-one padded
  sequences attend to their pad tokens; padding is not simply deleted.
  Batch-padding masks and sequence-padding tokens are different contracts.
- Three-axis RoPE uses dimensions `[32,48,48]`, lengths `[1536,512,512]`,
  theta 256, and adjacent real/imaginary pairs. Caption IDs start at
  `(1,0,0)`; image IDs start at `(padded_caption_length+1,0,0)`. Padding
  positions use zero IDs. This differs from the vision kernel's NeoX pairing.
- Normalize Q/K per head. Noise and main blocks modulate pre-attention and
  pre-FFN RMS-normalized inputs by `1+scale`, normalize each branch output,
  and gate its residual with `tanh(gate)`. FFN is `w2(silu(w1(x))*w3(x))`.
  Context-refiner blocks omit time modulation. Concatenate image then
  caption tokens for the main blocks.
- Embed time after multiplying normalized time by 1000. The final layer
  uses non-affine LayerNorm, epsilon 1e-6, followed by `1+linear(silu(time))`
  modulation and a biased projection back to patch values.

### Noise, scheduler, and the model-card discrepancy

The capture stores the actual CPU Torch FP32 noise tensor, generated with
seed 42, rather than treating a seed as portable across backends. At
1024 by 1024 its shape is `[1,16,128,128]`. Latents and scheduler updates
remain FP32; transformer weights/inputs in the staged reference use BF16.

The pinned scheduler has 1000 training timesteps, static shift 3, and
dynamic shifting disabled. Diffusers supplies `linspace(1,1/N,N)` sigmas,
then transforms each sigma with `3*sigma/(1+2*sigma)` and appends zero.
Transformer time is `1-sigma`. The pipeline negates transformer output
before Euler's `x_next = x + (sigma_next-sigma)*prediction` update.

**Nine requested steps produce nine forwards at this reference revision.**
The canonical model-card example says nine steps produce eight forwards;
that comment does not describe the pinned implementation. A counting
transformer running inside the actual Diffusers pipeline observes 1, 8,
and 9 forwards for requests of 1, 8, and 9 respectively. Both complete
sigma sequences are in [contract evidence](verification/z-image-ig0-contracts.json).
Do not silently drop the final update to reproduce the comment. The
capture defaults to nine to test the published invocation. The resumed
1024-by-1024 reference produces a coherent image with nine actual forwards;
nine is the pinned reference contract. An eight-forward optimization is
a separate comparison, not an interpretation of the model-card comment.

### VAE and dimensions

The VAE has 16 latent channels, spatial scale eight, channel widths
`[128,256,512,512]`, 32-group normalization, SiLU, convolution, upsampling,
and middle-block attention. Decode `latent/0.3611 + 0.1159`. Capture the
raw decoded tensor before image postprocessing; PNG conversion maps and
clamps the decoder output through Diffusers' image processor.

The reference accepts positive dimensions divisible by 16; 1023 fails.
Its zero-height argument falls back to 1024 and negative dimensions fail
later during allocation. The capture intentionally rejects nonpositive
dimensions up front. These boundary probes use a stub. Separate full 1024 captures now establish
execution on this development machine, without qualifying smaller RAM tiers. RoPE table bounds
also constrain maximum dimensions; no arbitrary upper limit is advertised.

## Operator reuse and gaps

| Required operation | Current foundation | IG1 disposition |
| --- | --- | --- |
| Group-64 affine INT4 matrix products | `gpu/dequant_int4_gemv.rs`, `gpu/dequant_int4_batch.rs`, CPU `quant.rs` | Compatible packed rows and BF16 scale/bias; validate diffusion token counts and scratch. Scalar batched path caps at 16 rows; do not treat it as a whole-image GEMM. |
| Qwen causal GQA, Q/K RMSNorm, FFN | Dense Qwen family and CPU/Metal RMSNorm/RoPE | Adapt into a hidden-state encoder; no autoregressive runner or final head. |
| Dense bidirectional attention | `gpu/vision.rs` | Candidate math, head width 128 supported; FP16-only and maskless today. Dtype, mask, and dense sequence footprint require adaptation. |
| Three-axis adjacent-pair RoPE | Existing text and vision RoPE primitives | New position-table/packing adapter and compatible pairwise kernel; vision NeoX rotation is not interchangeable. |
| RMSNorm and gated residuals | Existing RMSNorm and elementwise kernels | Adapt per-token modulation, tanh gates, and post-branch norms. |
| Patchify, unpatchify, learned padding | Vision preprocessing provides only a precedent | New image-generation layout implementation. |
| Timestep embedding and Euler updates | Portable arithmetic | New scheduler and time-embedding implementation, pinned fixture tests. |
| Final non-affine LayerNorm | Vision LayerNorm requires affine FP16 operands | Adapt the contract and dtype explicitly. |
| VAE 2D convolution, group norm, upsampling | No general VAE operator pipeline found | New CPU references and Metal operators. DFlash causal convolution is not a 2D VAE convolution. |
| Staged weight ownership | Streamer/load-guard foundations | Reuse later; no changes to text cache or slot policy in IG0. |

## Comparisons and quantization decision

[Bounded cross-reference evidence](verification/z-image-ig0-blocks.json)
compares unchanged Diffusers and MFLUX transformer-block implementations
on identical FP32 weights, inputs, modulation, masks, and position IDs.
The fixture uses hidden width 384, three heads of the real 128 dimensions,
and sequence lengths 1, 16, and 35. It is synthetic, not a checkpoint gate.
Maximum block absolute error is 3.58e-6; relative L2 is at most 3.83e-7.
RoPE maximum absolute error is 1.91e-6.

Frozen tolerances for **these bounded FP32 cases only**: block absolute
error <=1e-5 and relative L2 <=1e-6; RoPE absolute error <=3e-6. Token IDs,
masks, dimensions, inventory bytes, and fixture hashes require exact
agreement. The resumed full-width checkpoint-block comparison (3840 hidden units,
30 heads, first 64 captured image tokens) observes FP32 maximum absolute
error 1.72e-5 and relative L2 2.29e-7 between Diffusers and MFLUX. Its
affine-64 output error is 0.0326 relative L2. These are bounded-attention
component results, not full-image pixel parity. The full-width FP32 gate is absolute error <=3e-5 and relative L2 <=1e-6.
[Checkpoint-block evidence](verification/z-image-ig0-checkpoint-block.json)
retains the restricted-attention scope.

The native Rust checkpoint-block gate now passes that frozen threshold. Its
FP32 tree reduction for RMSNorm avoids the low-bit loss of a serial 3840-value
sum, and biased projections perform the matrix product before the bias add.
On the reproduced 64-token fixture, maximum absolute error is 5.72e-6 versus
Diffusers and 1.53e-5 versus MFLUX; relative L2 is 8.57e-8 and 2.52e-7,
respectively. This closes the bounded native block gate. The complete
nine-step transformer and real Rust VAE decode gates also pass; the VAE test's
optional raw-pixel assertions remain conditional on ignored binary fixtures.

The independent full 1024 FP32 VAE comparison uses all 138 canonical decoder
tensors, transposes convolution weights from OIHW to OHWI, and explicitly
sets MFLUX's precision to FP32. Maximum absolute error is 3.77e-5 and
relative L2 is 1.34e-6 (rounded up). Freeze absolute error <=6e-5 and
relative L2 <=3e-6 for this fixed latent fixture only; see
[VAE evidence](verification/z-image-ig0-vae.json).
Full-width BF16 cross-engine and wider image-quality tolerances remain open.
They must not inherit the synthetic FP32 tolerance without measurement.

[Quantization evidence](verification/z-image-ig0-quant.json) samples the first
64 rows of Qwen Q projection and transformer Q, FFN w1, and modulation
projections. It compares group sizes 32/64/128 against canonical weights
and synthetic 16-row activations. Group-64 weight relative L2 is about
0.090-0.094; projection relative L2 about 0.091-0.095. Group 32 reduces
error but does not match the existing group-64 kernel contract.

**Candidate representation: affine INT4, group 64, BF16 scales and biases,
low nibble first.** The sample payloads agree byte-for-byte with the actual
Rust `quantize_int4_affine` implementation, not only a Python formula.
The four-bit candidate passes the limited fixed-prompt visual review below,
but is not broadly quality-approved. Keep norms, biases, learned
pad tokens, timestep/modulation, final output projection, and VAE at higher
precision until component/image comparisons justify changing them.
Embeddings and protected projections remain BF16 in the candidate policy.

The quality-emulation policy quantizes 252 text-encoder matrices and 238
transformer matrices directly from canonical source values. It preserves
embeddings, normalization, biases, timestep/modulation, input/output
projections, and the entire VAE. `z_image_quantization.py` deliberately
executes dequantized BF16 weights in Diffusers: this tests quantization
quality and does not measure a packed INT4 runtime. The candidate weight
arithmetic is 2,822,044,672 bytes for text encoding and 3,661,523,072 bytes
for the transformer, including BF16 exceptions but excluding activations,
staging, and allocator capacity. These are calculated bytes, not peaks.

[Real comparison evidence](verification/z-image-ig0-real.json) records eight
complete captures, four BF16 references and four INT4 emulations. Every pair
has identical token IDs, masks, explicit noise, timesteps, and sigmas.
Conditioning relative L2 is 0.0808-0.0830; final-latent relative L2 is
0.442-0.922 and pixel relative L2 is 0.520-1.005. These large downstream
differences rule out pixel-parity claims for quantization.

[Visual review](verification/z-image-ig0-image-review.json) inspected all eight
original PNGs. Both composition images put the teapot left of the cup; both
spell FRESH BREAD exactly; both detail images retain hairs and wing veins;
both lighting images contain warm windows and cold snow. The quantized
lighting structure and typography presentation change substantially.
A second independent lighting run is byte-identical for every saved array,
including conditioning, block captures, scheduler updates, final latents,
and decoded pixels. Real empty/Unicode/overlong encodes are finite with
8/19/512 conditioning rows respectively. This is fixed-environment
reproducibility, not a cross-platform bitwise guarantee.

This supports group64 as the development candidate with the stated
exceptions, not a general quality guarantee or a packed-runtime speed claim.

The canonical transformer is FP32, not prequantized. Conversion must happen
once, then preserve the chosen packed representation. The probe does not
certify any community GGUF/MLX conversion or permit repeated requantization.

## Resource evidence and remaining gates

Host identified during preparation: Apple M4 Max, 14 CPU cores, 36 GB
unified RAM. Approximately 149 GiB of disk space was free after cleanup.
Canonical tensor payloads, excluding file headers:

| Component | Tensors | Stored dtype | Payload bytes |
| --- | ---: | --- | ---: |
| Text encoder | 398 | BF16 | 8,044,936,192 |
| Transformer | 521 | FP32 | 24,619,634,944 |
| VAE | 244 | BF16 | 167,639,366 |

These are disk/header arithmetic, not measured peak RAM. A BF16 transformer
alone has 12,309,817,472 weight bytes before activations and scratch. That
does not prove it fits safely alongside the OS and other applications.

The initial weight-backed preflight refused battery power
([recorded refusal](verification/z-image-ig0-preflight.json)). On resume, host
AC was confirmed, but Codex background processes exceeded the quiet-load
threshold; the [final preflight](verification/z-image-ig0-resumed-preflight.json)
also fails the quiet-load requirement. Correctness captures now run with `--capture-only`: AC remains
mandatory, load is recorded, and `benchmark_eligible` is false. These
instrumented timings do not establish a cold/warm performance baseline or
a supported minimum RAM figure.

The four busy-AC BF16 references recorded the following **instrumented
observations**, not a qualified latency or memory limit:

| Stage | Elapsed range (seconds) | Largest sampled process footprint (decimal GB) |
| --- | ---: | ---: |
| Text encoding | 1.82-3.26 | 8.97 |
| Denoising | 69.65-78.34 | 17.40 |
| VAE decoding | 1.63-2.38 | 11.51 |

Sampling every two seconds can miss short-lived peaks, especially decoder
runs. Final live MPS tensors are zero, but driver allocations remain as high
as 8.31/14.02/10.60 GB for encoder/transformer/VAE until process exit. A future
single-process staged runtime must explicitly account for allocator retention.
Swap already existed before these runs and grew during some busy captures;
no isolated attribution is justified. Physical reads, per-stage samples,
and fixture byte counts are retained in the
[capture manifests](verification/z-image-ig0-captures/lighting/denoise.json).
INT4 emulation still allocates BF16 weights and is not a packed-memory test.

The monitor records process footprint, RSS, physical read counters, swap,
MPS live/driver bytes, power, and competing process CPU. Its preflight
requires AC plus three two-second samples with external CPU total <=50%
and no external process >20% of one core. `--capture-only` permits busy
AC execution for correctness and explicitly disqualifies benchmark timings.
Captures record whether quiet AC
conditions persist. Instrumented capture timings include fixture copies
and are not production throughput. A separate cold/warm reference series
and explicit activation/scratch accounting are still required.

### Quiet-AC component benchmark protocol

The benchmark runs each component in a fresh process, loads its pinned
weights, records the first execution, then measures three executions with
weights resident. All outputs must exactly equal the lighting reference;
the transformer must perform nine forwards per execution. Tensor exports,
CPU output comparisons, checkpoint verification, and cold-copy preparation
are outside the execution timers. Load time is reported separately.

For the cold-file arm, write a fresh checkpoint copy with macOS `F_NOCACHE`.
Verify its payload after measurement, since a verification read before load
would warm the very cache being measured. Follow it with a fresh-process
load of the same files. Physical read counters classify the observed regime:
at least 90% of stored bytes is disk-read, at most 1% is cached, otherwise
mixed. This does not claim a cold SSD hardware cache or a system-wide purge.

Resource sampling is every 100 ms; competing CPU and AC power are checked
at one-second intervals. `/usr/bin/time -l` adds the kernel's whole-process
peak footprint. A process is excluded from the aggregate if any monitored
interval fails the existing quiet/AC requirement. Whole-process peaks
include preparation and verification; phase samples are retained separately.
Exact MPS operator scratch bytes remain unavailable, and driver allocation
minus live tensor allocation is not labeled as scratch.

The suite temporarily pauses user-owned photo-analysis, indexing, App Store,
and Codex display helpers. It leaves system/session/network services running,
restores the exact PID identities in `finally`, and has an independent
30-minute restoration watchdog. Its receipt records every paused process.
No app data is deleted or service permanently disabled.

```sh
target/ig0/venv/bin/python scripts/z_image_benchmark_suite.py \
  --out target/ig0/benchmarks/quiet-02 --pause-display-and-photo-work
target/ig0/venv/bin/python scripts/z_image_benchmark_summary.py \
  target/ig0/benchmarks/quiet-01 target/ig0/benchmarks/quiet-02
```

- [x] Pin model/reference inputs and inventory complete component headers.
- [x] Capture tokenization, dimensions, scheduler behavior, and bounded
  independent FP32 block/RoPE comparisons.
- [x] Identify operator gaps and a kernel-compatible quantization candidate.
- [x] Capture real conditioning, representative checkpoint blocks, identical
  noise, each scheduler update, final latents, and decoded pixels.
- [ ] Compare full-width/higher-precision and quantized components; approve
  precision exceptions and freeze their numerical tolerances.
- [x] Review composition/typography/detail/lighting images; resolve the
  eight-versus-nine evaluation schedule from pinned-reference evidence.
- [ ] Measure repeated stages, cold/warm storage, retained memory, swap,
  and a useful target memory/latency envelope on quiet AC hardware.
- [ ] Finalize the image manifest contract from those measurements.

## Reproduction and handoff

Run from the repository root. Weights, environments, and large fixtures
stay under ignored `target/ig0/`. Use new run directories for new requests.

```sh
uv venv --python 3.13 target/ig0/venv
uv pip install --python target/ig0/venv/bin/python -r scripts/z_image_reference_requirements.txt
python3 scripts/z_image_probe.py --output docs/verification/z-image-ig0-inputs.json
target/ig0/venv/bin/python scripts/z_image_capture.py contracts \
  --model target/ig0/inputs --out target/ig0/runs/contracts --device cpu
target/ig0/venv/bin/python scripts/z_image_compare.py blocks --out target/ig0/runs/blocks
target/ig0/venv/bin/python scripts/z_image_compare.py quant --out target/ig0/runs/quant
```

Download and verify the checkpoint with
`target/ig0/venv/bin/python scripts/z_image_download.py`. It pins the revision,
fetches all required folders into `target/ig0/model`, and checks payload hashes.
Then, on quiet AC hardware, run each stage in a fresh process:

```sh
target/ig0/venv/bin/python scripts/z_image_capture.py encode --out target/ig0/runs/lighting
target/ig0/venv/bin/python scripts/z_image_capture.py denoise --out target/ig0/runs/lighting
target/ig0/venv/bin/python scripts/z_image_capture.py decode --out target/ig0/runs/lighting
target/ig0/venv/bin/python scripts/z_image_capture.py validate --out target/ig0/runs/lighting
target/ig0/venv/bin/python -m unittest discover -s scripts -p test_z_image_evidence.py -v
target/ig0/venv/bin/python scripts/test_z_image_evidence.py \
  --mutation-report docs/verification/z-image-ig0-mutations.json
```

For the fixed review suite, repeat the three stages with `--case composition`,
`--case typography`, and `--case detail`, each in its own run directory.
Repeat all four with `--quantize` and the `-int4` directory suffix. Add
`--capture-only` to all stages when AC is available but load is not quiet;
this disqualifies benchmark use. Then run:

```sh
target/ig0/venv/bin/python scripts/z_image_summarize.py
target/ig0/venv/bin/python scripts/z_image_checkpoint_block.py \
  --run target/ig0/runs/lighting --out target/ig0/runs/checkpoint-block-verified
target/ig0/venv/bin/python scripts/z_image_vae_compare.py \
  --run target/ig0/runs/lighting --out target/ig0/runs/vae-compare

cargo test --release -p turbospark-image --test transformer_math_parity -- \
  --ignored --nocapture test_transformer_checkpoint_block_parity
cargo test --release -p turbospark-image --test pipeline_parity -- \
  --ignored --nocapture test_z_image_all_steps_from_captured_input_parity
cargo test --release -p turbospark-image --test pipeline_parity -- \
  --ignored --nocapture test_z_image_full_nine_step_checkpoint_parity
cargo test --release -p turbospark-image --test vae_parity -- \
  --ignored --nocapture test_vae_real_decode_parity
```

The checkpoint-block Rust command, complete nine-step DiT command, and real
1024-by-1024 VAE decode command are closed for the available fixtures. The
captured-input command records all nine local rows, and the full rollout
command passes the frozen `0.196` cumulative BF16 envelope. The VAE mutation
record is in `z-image-ig1-mutations.json`; its latent geometry assertion was
tightened from 128 to 127 and failed in isolation before the expensive decode.

The local gallery is `target/ig0/review.html`; it links original PNGs.
Empty, Unicode, and overlong conditioning can be regenerated with `encode`
and their corresponding `--case` values. A repeated lighting baseline uses
`--out target/ig0/runs/lighting-repeat` with the same default inputs.

Capture manifests contain tool/package/source provenance and array hashes.
The checked-in JSON copies reference external fixture filenames; validation
runs against the original run directory containing those arrays. Regenerate
them with the commands above rather than treating JSON alone as a fixture.
The real encode/denoise/decode path has completed all eight 1024-by-1024
review captures, including nine actual forwards, finite intermediate arrays,
PNG export, and recursive fixture validation. Quiet-hardware measurements
remain a separate gate.

The proposed image inventory requires the following component roles:

| Role | Required manifest information |
| --- | --- |
| Pipeline | Distinct image-generation capability and schema version; ordered component references; supported dimensions and batch policy. |
| Tokenizer | Asset hashes, framing template, thinking flag, truncation/padding policy, maximum 512 tokens. |
| Text encoder | Canonical revision, tensor inventory, hidden-state extraction point, causal mask, Q/K norm epsilon, quantized and protected tensor lists. |
| Transformer | Tensor inventory, patch ordering, sequence padding, time modulation, three-axis RoPE, dtype and packed group/nibble/scale/bias conventions. |
| Scheduler | Euler equations, static shift, sigma endpoints, actual evaluation count, guidance policy, explicit noise provenance. |
| VAE decoder | Decoder tensor subset, layout, normalization, scale/shift, output range and pixel conversion. |
| Verification | Reference revisions, fixture hashes, measured tolerance scope, resource methodology and qualification status. |

Text-only consumers must reject this capability. Cross-component dimensions
must agree before loading weights; every required tensor needs a shape,
storage dtype, byte span, hash, and component owner. The packed tensor list
must be explicit so protected projections cannot silently become INT4.
Measured activation/scratch limits and final allocation ownership remain
open. These are proposed requirements, not production serialization changes.

Thirteen tests pass. Eleven validator/preflight mutations (component omission,
header shape/bytes, fixture shape, corruption, non-finite values, revision,
missing fixture, forward count, busy-load modes, and battery refusal) each
fail only their intended case;
[mutation evidence](verification/z-image-ig0-mutations.json) records them.
Two additional quantization-policy mutations prove that canonical source
weights are used and the higher-precision exceptions are preserved.
The final workspace check results are retained in
[validation evidence](verification/z-image-ig0-validation.json).
No shared production text/kernel implementation was changed.
