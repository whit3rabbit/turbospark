# LookAtMyQuant (LAMQ) formats

STATUS: **runs on the 5.4 GB Inkling release; its loss against the original
is in the release's weights (step 5).** This page
is the spec the intake and kernels follow, and the record of their gates. The `inkling2-packed` section was checked on 2026-11-21 against the
5.4 GB release (container layout, dtypes, shapes, maps, the name map). The
math comes from the upstream source and is marked "from source" where no file
can confirm it. The CPU reference and the logits gate are the real check.

Upstream: `github.com/ShinMegamiBoson/lookatmyquant` (MIT). LAMQ is a
quantization compiler: it measures the KL cost of each tensor (and of row
buckets of the embedding and head) under candidate codes, then picks one code
per piece under a size budget. The upstream numbers (KL against Q4_K_M,
benchmark scores) are the authors' own and have not been reproduced here.

Upstream Python is read, never run, in this repository. A bit-level
cross-check against it is something a person runs by hand, outside the build.
Sources: `runtime_mlx/inkling/kernels.py` (reference decoders, RHT,
self-test), `model.py` (wiring), `fast_kernels.py` (Metal fast path).

## Containers

| Container | Releases | Status here |
|---|---|---|
| `inkling2-packed` | `shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB` (1.63 bpw), `-8.9GB` (2.69 bpw) | 5.4 GB: intake, kernels and decode done (workstream A), loss against the original traced to the release's weights, catalog row `inkling2-mini-lamq54` at `verified`; 8.9 GB: not installed |
| `lamq-1` | Gemma 4 E2B exports | planned (workstream C) |
| `opgraph-packed-1` | Qwen3.5-4B (`lm2-qwen35-4b-1.2gb`), `shinmegamiboson/Qwen3.8-27B-LookAtMyQuant-7.2GB` | planned (workstream C) |
| GGUF | Gemma 4 E2B compiled mixes | stock llama.cpp types per tensor, so it goes through the existing GGUF intake (`docs/GGUF.md`); no LAMQ code |

## Shared decode math (from source)

For an `m x n` matrix with per-row scale `scale`, input signs `s_in` (int8
+-1) and a code stream:

```
x~   = H_n (s_in * x) / sqrt(n)            input rotation, signs first
y[o] = scale[o] * sum_j W~[o, j] * x~[j]   code GEMV, scale after the dot
z    = s_out * (H_m^T y) / sqrt(m)         only when out_rot is set, signs last
```

The transform is orthonormal (1/sqrt(n) both ways). Upstream's self-test
checks `out(in(x)) == x` within 3e-3 for n = 2304, 1152, 3072, 1024.

### Hadamard `H_n`

`H_n = kron(A, B)`. For `a` from `floor(log2 n)` down to 0, skip unless
`2^a` divides `n`; let `p = n / 2^a`; take the first `a` where `p` is 1 or 2,
or `p - 1` is a prime `q` with `q % 4 == 3`.

- `p > 2`: `A = Sylvester(2^a)`, `B = Paley(q)`.
- `p = 1`: `A = Sylvester(2^(a/2))`, `B = Sylvester(2^(a - a/2))` (integer
  division).

| n | A | B |
|---|---|---|
| 2304 | Sylvester 32 | Paley q=71 (72) |
| 1152 | Sylvester 16 | Paley q=71 (72) |
| 3072 | Sylvester 256 | Paley q=11 (12) |
| 384 | Sylvester 32 | Paley q=11 (12) |
| 1024 | Sylvester 32 | Sylvester 32 |

- `Sylvester(N)[i][j] = (-1)^popcount(i & j)`, which is symmetric.
- `Paley(q)` is `(q+1) x (q+1)`. `chi(0) = 0`, and `chi(a)` is +1 when `a` is
  a nonzero square mod `q`, else -1. `Q[i][j] = chi((j - i) mod q)`.
  `S[0][1..] = +1`, `S[1..][0] = -1`, `S[1..][1..] = Q`, `S[0][0] = 0`, and the
  result is `S + I`. Field element `a` sits at index `a + 1`. Paley is not
  symmetric, so the transposes in the output rotation matter.
- Flat index is `i * |B| + j` (row-major `x` reshaped to `|A| x |B|`). The
  input transform is `A X B^T` and the output transform is `A^T Y B`.

`crates/gpu/src/hadamard.rs` is power-of-two only and is not reused.

### Code stream

Each row is a little-endian bit stream of `n * k` bits in `n * k / 32` u32
words (every width here is a multiple of 32 bits). A 12-bit state for a group
whose last column is `c` is read at bit `bp = (n - 1 - c) * k`:
`s = ((w[bp >> 5] >> (bp & 31)) | (w[(bp >> 5) + 1] << (32 - (bp & 31)))) & 4095`.
Words at index `>= n * k / 32` read as zero, which matters near column 0,
where the window runs past the row end.

| Code | k | g | LUT (`luts.npz`, f16) | Index |
|---|---|---|---|---|
| `rtr1`..`rtr4` | 1..4 | 1 | `(4096,)`, one table per code | `s` |
| `ht2` | 2 | 4 | `(512, 4)` | `(s * 0x9E3779B1 mod 2^32) >> 23` |
| `ht3` | 3 | 2 | `(1024, 2)` | `(s * 0x9E3779B1 mod 2^32) >> 22` |
| `lm6` | 6 | 1 | per-matrix `levels` | 4 columns in 3 bytes: `v = b0 + (b1 << 8) + (b2 << 16)`, column `4i + t` is `(v >> 6t) & 63` |
| `lm8` | 8 | 1 | per-matrix `levels` | one byte per column |

- For `ht*`, a group covers columns `j..j+g-1`, so `c = j + g - 1`. Entry
  element `t` multiplies column `j + t`.
- `manifest.json` also names `ht1` (fid 0). No `ht1` LUT ships and no tensor
  uses it, so the intake refuses it.
- `levels` are F32 `(64,)` / `(256,)` for linears. lm buckets of the
  embedding and head store theirs the same way (`b0_levels` is `<f4`).

## `inkling2-packed` (Inkling 2 Mini), checked against the 5.4 GB release

Files (63 in all): `manifest.json` (`"format": "inkling2-packed-v1-sqyfr"`,
`fmt_ids`, LUT shapes, steering count 499), `config.json` (the HF config,
text_config identical to `docs/INKLING2_PHASE0.md`), `luts.npz` (42 KB),
`steering.safetensors` (122 MB, 499 uncoded tensors), `embed_meta.json` +
`embed_packed.npz`, `head_meta.json` + `head_packed.npz`, and
`layer_NN_meta.json` + `layer_NN_packed.npz` for 24 layers, plus tokenizer
files. There is no vision tower or drafter, so an install is text-only.

### npz container

Every member of every file is STORED (zip method 0), with no zip64 and
`.npy` v1 headers, little-endian, C order. Dtypes seen: `<f2`, `<f4`,
`<i4`, `<i2`, `<u4`, `|u1`, `|i1`. Signs are int8 +1/-1 (`|i1`).

### Linears (`layer_NN_*`)

Meta: `linears[<hf name>] = {fmt, m, n, k, g, K, hash_mult, safe, out_rot?}`,
plus `experts.gu` / `experts.dn` (count of experts per format). Arrays:
`lin_<safe>_codes` (`<u4` `(m, n*k/32)` for trellis, `|u1` for lm),
`_scales` (`<f2` `(m,)`), `_signs_in` (`(n,)`), and for lm6 `_levels`. Only
`v_proj` uses `out_rot` in this release, and it carries `_signs_out`.

### Experts

- Per format `f`: `gu_<f>_codes` / `_scales` and `dn_<f>_codes` / `_scales`,
  slabs of consecutive experts.
- `gu_map` / `dn_map` are `<i2` `(128, 2)`: `(fid, slot)` per expert, fids
  from `manifest.fmt_ids` (ht2 = 1, ht3 = 2, rtr1 = 3, ...). Slots are dense
  `0..count-1` per fid (checked on layer 7).
- `gu_signs_in` `(2304,)` is applied once per token and shared by all 8
  slots. `dn_signs_in` `(1152,)` is applied per slot to the activation.
- Gate-up rows: gate is rows `[0, 1152)` and up is `[1152, 2304)`, as
  contiguous halves. This differs from the HF checkpoint's interleaved `w13`.
- Activation (from source): `(tanh(g / 4) * 4) * sigmoid(g) * (tanh(u / 25) * 25)`,
  where the sigmoid takes the uncapped `g`. No output rotation on experts.

### Embedding and head buckets

- `order` (`<i4`): packed position -> token id. It has 201024 entries in
  the embedding and 200058 (unpadded vocab) in the head.
- Buckets fill positions in meta order (`kb` 0..7). Bucket `b` covers
  `[lo_b, lo_b + rows_b)`, and its row `r` is position `lo_b + r`. The row
  counts sum to 201024 (embed) and 200058 (head).
- Per bucket: `b<kb>_codes`, `_scales` (`<f2`), `_signs_in` (`(2304,)`), and
  for lm buckets `_levels`.
- Embedding row (from source): decode the row, `scale * W~`, then the output
  rotation with the bucket's `signs_in`, then `embed_norm`.
- Head (from source): per-bucket GEMV on `H(s_b * h)`, logits scattered to
  token `order[p]`, then the final softcap 30.

| Bucket rows | embed (5.4 GB) | head (5.4 GB) |
|---|---|---|
| 1024 | lm6 | lm8 |
| 3072 | rtr4 | lm6 |
| 4096 | rtr4 | lm6 |
| 8192 | rtr3 | rtr4 |
| 16384 | rtr3 | rtr4 |
| 32768 | rtr2 | rtr3 |
| 65536 | rtr1 | rtr2 |
| tail | rtr1 (69952) | rtr2 (68986) |

### Per-layer code mix (5.4 GB)

- Attention projections other than `v_proj` use `ht2`, `ht3`, `rtr4`, and
  `rtr1` twice (layer 2 `relative_position_bias`, layer 3 `q_proj`).
- `v_proj` is `lm6` with `out_rot` on layers 7 to 19 and 23, and `rtr4` on
  the rest.
- Experts use `ht2`, `ht3` and `rtr1` only. `lm8` occurs only in the head.
- Every format named above occurs somewhere, so the kernels need all of
  `ht2`, `ht3`, `rtr1`..`rtr4`, `lm6` and `lm8`.

### Name map

All 646 resident tensors of the HF install map one to one (checked: none
missing, extra or duplicated). Steering and linear names are HF-transformers
names (`model.layers.N.self_attn.q_proj.linear`), which become the mlx-lm
names that `inkling2_mlx.rs::mlx_to_hf` already inverts once these renames
are applied:

| LAMQ | install |
|---|---|
| `self_attn.{q,k,v}_proj.linear` | `attn.{wq_du,wk_dv,wv_dv}` |
| `self_attn.relative_position_bias.linear` | `attn.wr_du` |
| `self_attn.o_proj.gate_proj` | `attn.gate_attn` |
| `self_attn.o_proj.linear` | `attn.wo_ud` |
| `self_attn.q_proj.norm.scale` | `attn.q_l2_clip.scale` |
| `self_attn.relative_position_bias.norm.scale` | `attn.rel_l2_clip.scale` |
| `self_attn.relative_position_bias.proj` | `attn.rel_logits_proj.proj` |
| `self_attn.{k,v}_proj.sconv.*` | `attn.{k,v}_sconv.*` (`static_weight` flipped as in the HF intake) |
| `model.message_input.query` `(12, 4, 2304)` | split into `middle_cycle_cross_pass_{queries,kv_queries}` |

Every F32 steering tensor holds values that are NOT exact BF16 (checked on
all 18 F32 patterns: convs, controller forget/gain planes, router and bias,
every attention-residual and message query). The HF install has the convs
and controller planes in BF16, so LAMQ tuned them in F32. The intake keeps
them F32 rather than narrowing, and the runtime decides per kernel; any
rounding there is explicit and measured, not done silently at intake.

## `lamq-1` (not yet checked against a file)

`model.safetensors` + `lamq.json` (`"format": "lamq-1"`). Per matrix:
`<name>.codes` (u32 stream stored as I32), `<name>.scales` (F16),
`<name>.s_in` (I8), `<name>.s_out` for two-sided rotation. Table buckets:
`<name>.b<k>.rows` (token ids), `.codes`, `.scales`, `.signs`, or
`<name>.b<k>.dense` (BF16) when uncoded. Shared `trellis.k<k>` (F32, 4096).
Everything else keeps its own name in BF16.

## `opgraph-packed-1` (not yet checked against a file)

`artifact.json` (`format`, `blob`, `blob_bytes`, `tensors`, each with
`dtype`, `shape`, `nbytes`, `offset` and a sha256) plus `weights.bin`. Per row
group `g<i>`: `<name>.signs` (I8), `<name>.g<i>.rows` (U32), `.codes` (U32),
`.scales` (F32), `.table` (F32). Tables shared by exact bytes are aliased.
The intake must check every offset bound and every sha256.

## Plan and gates

1. DONE (2026-11-21): Phase 0 inventory of the 5.4 GB release, above. Small
   files are in `~/models/lamq-probe/ink54/` (not release source).
2. DONE (2026-11-21): CPU reference `compute::lamq` (bit windows, hash,
   trellis and level rows, Sylvester x Paley-I RHT, `LamqLinear`,
   `embed_row`) and the stored-only reader `turbospark_repack::npz` (no new
   dependency). Unit tests: windows against an independent shift register,
   exact hash constants, Paley and Kronecker Hadamard property, round trips at
   every release width, npz refusals (deflate, encryption, data descriptors,
   zip64 archives, local header disagreement, CRC, a member hidden inside
   another, npy header rules). Each guard was mutation-checked and fails.
   Real-data oracle (`examples/lamq_embed_probe.rs`, M4 Max): decoded
   embedding rows of the 5.4 GB release against the BF16 HF install, cosine
   per bucket (4 rows each):

   | bucket | code | cosine mean |
   |---|---|---|
   | 0 | lm6 | 0.9997 |
   | 1, 2 | rtr4 | 0.9977 |
   | 3, 4 | rtr3 | 0.9910 |
   | 5 | rtr2 | 0.9649 |
   | 6, 7 | rtr1 | 0.856 |

   Cosine falls with bit width, as quantization error should. A dropped Paley
   transpose in the output rotation gives -0.97 (Paley-I is `I + C` with `C`
   skew, so a transpose error flips the sign, it does not scramble), so the
   probe catches it.

   The hashed codes and the input rotation (`examples/lamq_linear_probe.rs`):
   each layer-0 attention linear on one deterministic input, LAMQ against the
   BF16 install's matrix, output cosine:

   | linear | code | cosine |
   |---|---|---|
   | `wq_du` | ht2 | 0.9352 |
   | `wk_dv` | ht3 | 0.9860 |
   | `wv_dv` | rtr4 | 0.9950 |
   | `wr_du` | ht2 | 0.9281 |
   | `gate_attn` | ht2 | 0.9371 |
   | `wo_ud` | rtr4 | 0.9955 |

   Anchoring `ht*` groups on their first column instead of their last drops
   every `ht` row to about 0 and leaves `rtr4` unchanged, as expected for a
   `g = 1` code. Not covered yet: experts (contiguous gate/up halves, the
   `dn` rotation), `lm6` with `out_rot` (layers 7 to 19, 23), the head.
   Not run: the x86_64 Linux cross-target check (target not installed here).
3. DONE (2026-11-21): intake, `turbospark_repack::write_inkling2_lamq_install`
   (`crates/repack/src/inkling2_lamq.rs`, `examples/inkling2_lamq_install.rs`).
   Install format:
   - Resident tag 36 (`DTYPE_LAMQ`, `docs/GTURBO.md`): payload = codes (or
     an i32 bucket order, or int8 signs), scales = f16 per row, "bias" region
     = aux block opening with `compute::lamq::LamqAuxHeader` (kind, code,
     out_rot, n, m), then signs, then F32 levels for `lm*`.
   - Embedding and head keep their names as order entries; buckets are
     `<name>.b<kb>`. LUTs are raw FP16 `lamq.lut.<code>`. Per-layer expert
     signs are `model.llm.layers.<l>.mlp.experts.lamq_signs`.
   - Expert blobs: `lamq_fmt` (16 B: gate-up and down code ids and row
     bytes), `gate_up`, `gate_up_scales`, `down`, `down_scales`. Code
     sub-tensors are padded to the layer's widest format so sub-offsets are
     uniform (the streamer reads `e * stride`): 4.71 GB of expert codes
     become 6.73 GB on disk (x1.43). A tighter layout is open work.
   - Manifest: `embedding`/`attention` scheme `lamq`, `routedExpert`
     `inkling2_lamq` (`model_io::LAMQ_SCHEME`, `INKLING2_LAMQ_ROUTED_SCHEME`),
     raw F32 router, no shared expert.
   Real intake of the 5.4 GB release (M4 Max): 33 s, 3.4 GB peak RSS, 6.9 GB
   install, 692 resident entries (the HF install's 646 plus 6 LUTs, 16
   buckets, 24 sign sets). Every HF resident name is present with its shape;
   184 are F32 where the HF install has BF16 (the steering, not narrowed).
   Read back through the install (`examples/lamq_expert_probe.rs`), against
   the HF install on the same inputs (output cosine):

   | probe | cosine |
   |---|---|
   | `wv_dv` rtr4 (layers 0, 20) | 0.996, 0.994 |
   | `wv_dv` lm6 + out_rot (layer 7) | 0.998 |
   | head, all 200058 logits | 0.960, same top-1 token |
   | expert gate / up, rtr1 | 0.72 to 0.78 |
   | expert gate / up, ht2 | 0.90 to 0.94 |
   | expert gate / up, ht3 | 0.98 |
   | expert down, rtr1 / ht2 / ht3 | 0.74 to 0.80 / 0.94 to 0.95 / 0.99 |

   The same rows read with the HF gate/up interleave give cosine about 0, so
   the contiguous halves are right; layer 20 is checked against MXFP8, the
   others against NVFP4. Most experts of the 5.4 GB release are rtr1, which
   is where its size comes from; whether that holds up end to end is the
   logits gate's question, not this probe's. Unit tests cover the aux codec,
   the manifest schemes, the name map and the intake refusals (repeated or
   out-of-range token ids, short buckets, stray members and `signs_out`,
   reused or unknown expert slots); each guard was mutation-checked.
   (The runtime opened it from step 4 on.)
4. DONE (2026-11-21): Metal and runtime. `shaders/lamq.metal` (`gpu::lamq`):
   the Kronecker RHT (FWHT over the Sylvester factor, dense small factor),
   trellis and level GEMVs, one-row decode, bucket scatter. Function
   constants are 120-126 (used by no other shader; the expert pair compiles
   this file with `moe.metal`) and every one is in the pipeline key.
   `shaders/moe_inkling2_lamq.metal` (`gpu::moe_inkling2_lamq`): the expert
   pair, streamed and resident, reading each expert's code from its format
   record and every LUT from one pack. Parity (`tests/lamq_parity.rs`,
   `tests/moe_inkling2_lamq_parity.rs`, M4 Max) against `compute::lamq`: every
   code, every release width both ways, 8 mixed-format slots at the Mini's
   dims; 16 shader mutants each fail a test. The flow
   (`families/inkling2/lamq.rs`) parses every tag-36 entry at open, rotates,
   runs the GEMV and the output rotation for coded linears, decodes the
   embedding row through the token's bucket, scatters the head buckets into
   the FP16 logits, and runs the expert pair in both MoE paths. Open admits
   tag 36 and the `lamq.lut.` FP16 tables for this family only.
   Steering: the 184 F32 conv-tap and controller tensors are rounded to BF16
   at intake (round to nearest even, largest relative change 3.9e-3), since
   their kernels read BF16. The router, its bias and every query stay F32.
5. Logits gate (2026-11-21, M4 Max, 5.4 GB release, `inkling2_logits`),
   against the BF16 HF install on the same Metal flow (that install is itself
   at mean KL ~5e-4 against llama.cpp):

   | prompt | top-1 | mean KL | median | max |
   |---|---|---|---|---|
   | 19 tokens (capitals) | 19/19 | 2.6e-2 | 2.2e-2 | 6.0e-2 |
   | 232 tokens (prose) | 182/232 | 0.154 | 0.095 | 2.03 |

   Re-run 2026-11-22 at `07218acf` (after the step 9 to 14 kernel ports,
   both installs dumped by the same rebuilt example): 19/19, mean KL 2.8e-2,
   max 6.9e-2; 183/232, mean KL 0.154, median 0.093, max 2.08. The kernel
   work did not move the gap in either direction.

   KL is flat over position (0.12 to 0.22 per 29-token band), so this reads
   as weight error, not a position-dependent defect. Most misses are the
   reference's second choice. Decode is 64 ms per token against 28 ms for
   the BF16 install (untuned kernels, unpaired, not a protocol number).
   OPEN: the authors report `free_KL` 0.0895 (English/chat/code windows)
   and `all13_KL` 0.2871 (13 languages) for this file against the original
   teacher, their harness, which is not in the upstream repository. (An
   earlier note here read their 89.2% as top-1; it is their "3-task mean vs
   teacher", GSM8K, HumanEval and vision retention, not a token metric.)
   The protocols differ, so the gap is not a measured defect, but it is not
   explained either. Two more windows (2026-11-22, `07218acf`, 450 tokens
   each, no BOS, `inkling2_logits` against the BF16 install): English
   (docs/DEVELOPMENT.md) top-1 363/450, mean KL 0.248, median 0.132; Rust
   source (crates/compute/src/lamq.rs) 357/450, 0.313, median 0.186.
   RULED OUT (2026-11-22), a decode defect in any coded tensor: the
   diagnostic `examples/inkling2_lamq_weight_err.rs` (runtime crate)
   compares every coded matrix through the CPU reference
   (`LamqLinear::forward`, rotations included) with the original weights
   on random probes. Error sits in tight bands by code, with no outlier
   tensor, flat over the 16 columns of a chunk and equal on even and odd
   chunks; the embedding (no activations behind it) lands within 3 to 8%
   of the Gaussian bound 2^-R at every code (rtr4 0.067, rtr3 0.135, rtr2
   0.26, rtr1 0.515), so the windows, tables, rotations and bucket order
   decode right. The head (1.3x the bound) and attention (rtr4 1.9x) sit
   higher, as an activation-aware quantizer places its error. Experts
   (against the NVFP4 / MXFP8 originals times their outer scales; the
   release uses only ht2, ht3 and rtr1 there) match the same bands: rtr1
   0.63 to 0.71, ht3 0.16 to 0.22, ht2 0.31 to 0.50, scale ratios 0.88 to
   1.0. The intake reads every meta field the upstream loader uses
   (`hash_mult` is checked against the kernels' constant). RULED OUT
   (2026-11-22), the Metal runtime on this install as a whole: the CPU f32
   reference forward (`compute::inkling2::forward`, whose `WeightSource`
   now has default `matvec`, `embed_row` and expert hooks; a hook-vs-dense
   test is bit-exact and fails with gate and up swapped) run on the LAMQ
   weights by `examples/inkling2_lamq_reference_logits.rs` (runtime crate,
   every coded tensor through `LamqLinear`, about 2 s a token). Over the
   48-token `ids48` prompt: CPU LAMQ against Metal LAMQ 45/48 top-1, mean
   KL 1.4e-3, max 1.1e-2; BF16 against Metal LAMQ 19/48, 0.762; BF16
   against CPU LAMQ 20/48, 0.768 (that prompt is a synthetic id run, so
   its absolute KL is high). The runtime reproduces the reference about
   500x closer than either sits to BF16, so the gap is in the release's
   weights as this format reads them, and the embedding's near-bound error
   says the reading is right. What is left unexplained is only the
   protocol difference from the authors' 0.0895, whose harness and windows
   are not public. RULED OUT (2026-11-21): the BF16 rounding of the F32
   steering. Rounding every narrowed value to the OTHER BF16 neighbour (0.5
   to 1 ulp, max relative change 7.8e-3, about 3x the nearest-even error)
   moved prose KL from 0.1543 to 0.1556 and top-1 from 182 to 181, and the two
   installs differ from each other by mean KL 1.5e-3. Not yet run:
   the 8.9 GB release, sampled generation. Memory oracle (2026-11-22,
   `inkling2_lamq54_memory_oracle`): 1,587 MiB streamed at 16 slots; it
   drives two of the three protocol cases, because the release reasons past
   the 4,096-token budget on long-synthesis where the original stops at
   2,070. Speed and peak footprint beside the other Inkling installs:
   `docs/INKLING2_PHASE0.md`, "Speed and memory by install".
6. DONE (2026-11-21): Workstream B, Inkling 2 GGUF intake and decode
   (`docs/INKLING2_PHASE0.md`, GGUF sections). The unsloth UD files are not
   gated (not downloaded) and MXFP4 down projections are refused.
7. DONE (2026-11-21): install and run paths. `turbospark model pull-local`,
   `pull --repo` (the probe reports a LAMQ release or an Inkling GGUF as
   runnable, and the install downloads the files whole at the pinned commit,
   resumes by size, then runs the local intake), the FFI
   (`ts_install_local_with_options`, `ts_local_install_kind_json`), Swift
   (`TurboSparkCatalog.install(localPath:)`) and the app's Discover > Install
   from Files sheet. Shared steps: `catalog::install_inkling2_local` and
   `catalog::install_inkling2_repo`. Chat through `turbospark run` was broken
   for every Inkling install until the padded vocabulary rows were masked
   with a finite value: the producer wrote -inf there and the sampler
   refuses non-finite scores. The logits harness drops those rows, which is
   why the gates above never saw it. Checked after the fix: BF16, LAMQ 5.4
   GB, MLX 4-bit and the local IQ2_M GGUF each answer a one-sentence factual
   prompt correctly and stop at end of turn. Real Hub run (2026-11-21):
   `pull --repo shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB` pinned
   commit `9af87091`, downloaded 5.0 GiB, installed 6.9 GiB in 21 minutes
   (mostly download), removed its download folder, and wrote every file
   except `manifest.json` byte-identical to the install converted earlier
   from the same files; it answers the same prompt correctly.
8. DONE (2026-11-21): first decode tuning, output bit-identical (48-token
   logits `cmp`-equal before and after; parity suites pass). The dispatch
   profile (`TURBOSPARK_DISPATCH_PROFILE=1`, rank only) put `lamq_rht` first
   at 34.5% of LAMQ decode: one threadgroup per vector, so a decode
   rotation runs on one core, and its dense loop re-read the 72x72 factor
   from device memory per element. Changes: the factor (exactly +1/-1) is
   staged as int8 in threadgroup memory and the rotation runs 1024
   threads; the coded GEMV takes 32 rows per threadgroup (each threadgroup
   stages the whole table, up to 4096 entries); expert phase 1 stages its
   expert's gate-up table when all 32 of a threadgroup's rows share it.
   tslab `profile`, experiment `lamq-kernels-ab2`, old and new worker
   builds interleaved, 5.4 GB install, `protocol:short-explanation`, 128
   decode tokens, resident experts: old 14.70 to 15.08 tok/s, new 16.06 to
   18.15 (medians 14.91 and 17.95), every new run above every old run. ALL
   12 RUNS ARE DIRTY (a foreign Python job and mediaanalysisd over 90% CPU):
   a direction, not a quotable number. Tried and reverted: a
   lane-contiguous trellis loop with a sliding 64-bit code register (loses
   coalescing; read slower, also under noise). Next levers, not tried:
   batching the per-matrix rotations into one dispatch, and a multi-weight
   inner loop that keeps lanes coalesced.
9. STUDIED (2026-11-21): the authors' fast path, read only, never run
   (`runtime_mlx/inkling/fast_kernels.py` at commit 5106923, MIT). Their
   card claims 63.5 tok/s serial for the 8.9 GB file on an M5 Max, in the
   range of plain MLX 4-bit, so the gap to it is our kernels, not the
   format. Their techniques, none of which this port had:
   - Inner loop: each lane decodes a chunk of 16 consecutive columns per
     step, reading its 1 to 3 code words once and taking every window
     with `extract_bits`; chunks interleave across lanes (`c = lane + L *
     it`), which keeps the word loads coalesced. A lane-contiguous layout
     loses that (measured here, reverted).
   - Multi-weight tables, exact: `rtr1` takes 14 bits for 3 consecutive
     weights (`build_rtr_multitable`, a 16384 x half4 table, 128 KB in
     device memory, relying on L1); `rtr2` and `rtr3` 2 weights; `ht2` and
     `ht3` read a half4 or half2 table row per lookup from threadgroup
     memory (4 KB).
   - Row layout: 16 lanes per row (8 for width 1152), several rows per lane
     group, and each x chunk loaded once and reused across those rows.
     Activations are half and the 16-column partials are summed in half.
   - Rotations: the 32-wide Sylvester factor with `simd_shuffle_xor` (lane
     = outer index, no barriers), then the dense factor after one barrier;
     one launch rotates every vector of a layer (grid y over an `x_map` /
     `s_map` pair: q, k, v, rel and o_gate in one dispatch).
   - Fusion: the expert-input rotation fused with the softcapped SwiGLU,
     norms fused with the rotation that follows them, the attention gate
     with the output rotation, and the whole MoE in one launch across
     formats.
10. DONE (2026-11-21): first port from step 9, expert phase 1 only. Each
    16-lane half of a simdgroup takes one gate/up row pair; each step loads
    a 16-column chunk of x once and decodes that chunk of both rows from
    1 to 3 code words read once (`lamq_chunk16`). `rtr1` reads 14 bits per
    three columns from a 16384 x half4 table appended to the LUT pack
    (`LAMQ_RTR1_MT_OFF`, built from the same f16 values, so the weights are
    exact). Sums stay F32 (theirs use half). Not bit-identical to the
    previous kernel: phase 1 matches the f64 reference to 1e-5 relative
    for all eight format pairs (checked with the parity tolerance tightened,
    then restored), and swapping two table slots fails the parity test. On
    48 tokens against the previous build: mean KL 1.4e-3, max 1.9e-2, 45/48
    top-1, which reads as F32 reordering amplified by 48 applications with
    FP16 activations (about 5% of the LAMQ-vs-BF16 error). tslab profile,
    experiment `lamq-kernels-h16`, interleaved, two passes: committed
    kernels 17.15 to 18.33 tok/s (median 17.8), this kernel 18.23 to 19.21
    (median 18.8). ALL RUNS DIRTY (mediaanalysisd over 90%). Not yet
    ported: phase 2, the coded GEMV, the batched shuffle rotation, fusion.
11. DONE (2026-11-21): batched input rotations, bit-identical. A layer's
    attention group (up to eight projections of width 2304 over three
    inputs) rotates every LAMQ input in one `lamq_rht_in_batch` dispatch,
    one threadgroup per vector, so they run on several cores at once
    instead of one after another on one; each GEMV then reads its own
    rotated slot (`LamqRuntime::gemv_batch`). Each vector's arithmetic is
    the single-vector kernel's: 48-token logits `cmp`-equal to the previous
    build, and `batched_rotation_equals_one_at_a_time_bit_for_bit` (a wrong
    input selection fails it). Single rotations per token 332 -> 112 plus
    80 batched; rotation share of the dispatch profile 19% -> 11%. tslab
    profile `lamq-kernels-3arm` (three distinct workers, interleaved, two
    passes, all dirty from mediaanalysisd): previous build 18.01 to 18.90
    tok/s (median 18.7), this 20.67 to 21.00 (median 20.9).
    METHOD TRAP, recorded because it produced a false null: an A/B worker
    built with `cargo build -p turbospark-lab -p turbospark-runtime
    --example inkling2_logits` builds nothing (the lab crate has no such
    example) and leaves the previous `tslab-worker` in `target/`; both arms
    of `lamq-kernels-c16` were that stale binary. Check that each arm's
    `exe_sha256` in the ledger differs before reading an A/B.
12. DONE (2026-11-21): the step 10 chunk decode in expert phase 2 and the
    coded GEMV. Phase 2 keeps its layout (one threadgroup per output row,
    one simdgroup per slot, slot-order sum) and decodes 16-column chunks
    across the simdgroup (`lamq_chunk16`); the GEMV decodes them against its
    threadgroup table (`lamq_chunk16_tg`, no multi-weight table there yet).
    Both need widths that are multiples of 16 (checked on the host). Parity
    suites pass; a one-bit window shift in the GEMV helper fails them. The
    phase 2 output is FP16, so its parity is to one half-precision step.
    Drift against the original bit-exact kernels with steps 10 to 12: mean
    KL 2.3e-3, 47/48 top-1 over 48 tokens. tslab: `lamq-kernels-3arm`
    (this change alone on the step 10 build: median 18.7 -> 20.1 tok/s) and
    `lamq-kernels-both` (on top of step 11: 20.70 to 20.89 vs 21.99 to
    23.47, medians 20.8 and 22.1), all runs dirty. Day total on the 5.4 GB
    install, same workload: about 14.9 -> 22.1 tok/s.
13. DONE (2026-11-22): the expert chunk decode specialized per format at
    compile time. The code id is read once per row and switched on outside
    the chunk loop, which calls a template per format (`lamq_p1_rows<ID>`,
    `lamq_p2_row<ID>`), so k, g, the hash shift and the table offset are
    constants and the 16 windows of a chunk unroll with fixed offsets.
    Phase 1 also stages the ht2/ht3 gate-up table (2048 halves, the same f16
    values) in threadgroup memory when every row of the threadgroup shares
    one expert (`F % 64 == 0`, true for Inkling's 1152). Staging the wrong
    table fails the parity test. The summation order is unchanged, but the
    48-token logits are not `cmp`-equal (the unrolled loops contract to
    FMAs differently): against step 12, mean KL 3.0e-3, max 2.1e-2, 46/48
    top-1; against the original bit-exact kernels 1.3e-3 and 47/48. tslab
    `profile`, experiment `lamq-kernels-spec`, step 12 and this worker
    interleaved (distinct `exe_sha256`), two passes of three rounds, 5.4 GB
    install, `protocol:short-explanation`, 128 decode tokens, resident
    experts, all 12 runs clean: step 12 22.76 to 23.12 tok/s (median 22.85),
    this 28.78 to 28.90 (median 28.85). A dev A/B, not a frozen row.
    Tried after it and NOT kept, both measured the same way (all runs
    clean): the coded GEMV with one row per 16-lane half (balanced chunks
    at width 2304), median 29.06 -> 29.56 tok/s, under the 2% floor with
    overlapping ranges (`lamq-kernels-gemv16`); and a grid-stride GEMV
    capped at 160 threadgroups so each stages its table once instead of
    once per 32 rows, bit-identical output, median 28.88 -> 28.75
    (`lamq-kernels-gs`). The table re-staging is not what limits the GEMV.
    The dispatch profile after this step (rank only): `lamq_gemv` 22%,
    phase 1 17% (was 25%), the two rotation kernels 15%, phase 2 10%.
    Also tried and NOT kept: the Sylvester stages h < 32 of both rotation
    kernels with `simd_shuffle_xor` (step 9's form) instead of one
    threadgroup barrier per stage, same stages and operands, so 48-token
    logits `cmp`-equal to step 13 (a flipped butterfly sign fails the
    rotation parity test). Median 29.50 -> 29.70 tok/s, under the 2% floor
    (`lamq-kernels-shuf`, 3 of 12 runs clean, spotlightknowledged busy).
    Dispatch profile: `lamq_rht` 2.20 -> 2.06 ms/token, `lamq_rht_in_batch`
    1.67 -> 1.50. The barriers are not the rotation's cost; the likely one
    is the 72-tap dense factor with each vector on one threadgroup (one
    core), so the next lever is splitting a vector across threadgroups or
    fusing the rotation into the kernel that produces its input. Other
    barrier-per-stage butterflies the same rewrite would apply to (not
    measured): `hadamard_butterfly` (hadamard.metal), the TurboQuant KV
    transforms (attention_tq.metal, kv_quantize_tq.metal), and the norm
    tree reduction in mla.metal (`simd_sum` there reorders the sum, so not
    bit-identical). Price each with a dispatch profile first: a kernel that
    is not barrier-bound gains as little as this one.
14. DONE (2026-11-22): split rotations, bit-identical. Every release
    width whose Sylvester factor is 32 or less (2304 = 32 x 72,
    1152 = 16 x 72, 384, 1024) rotates in `lamq_rht_split` /
    `lamq_rht_in_batch_split`: output column j of the dense factor is
    independent, so a vector spreads over ceil(nb / 8) threadgroups of 256
    threads (9 for 2304) instead of one 1024-thread threadgroup on one
    core. Each na-lane group takes one column, lane = outer index: the
    dense sum in the same order, then the FWHT with `simd_shuffle_xor`.
    3072 = 256 x 12 keeps the one-threadgroup kernels. 48-token logits
    `cmp`-equal to step 13; swapping the vector and column-block decode
    fails both rotation parity tests, a flipped butterfly sign fails the
    reference one. Dispatch profile (rank only): rotations 3.87 -> 2.32
    ms/token. tslab profile, experiment `lamq-kernels-split`, interleaved,
    two passes, 11 of 12 runs clean: step 13 28.86 to 29.51 tok/s (median
    29.09), this 30.64 to 31.30 (median 30.71). A dev A/B, not a frozen
    row.
    `lamq_gemv` characterized (2026-11-22, M4 Max, nothing kept). The
    ignored `gemv_bandwidth` test in `tests/lamq_parity.rs` times the GEMV
    in one primed command buffer over about 192 MB of distinct matrices.
    Time does not depend on bits per weight: at 2304 x 2304, rtr1 to rtr4
    all take about 30 us (22 to 84 GB/s of codes), and lm8, a plain byte
    index, is as slow. The kernel is bound by its one threadgroup-table
    gather per weight (about 170 to 240 G weights/s; ht2, one hashed lookup
    per 4 weights, is about 1.5x faster), not by DRAM or the bit extract.
    Per token the Mini runs about 1.1 G coded weights through it (attention
    643 M, head 461 M; codes mostly rtr4 and ht3 in attention, rtr2 to rtr4
    in the head, rtr1 in one matrix), about 5.5 ms of a 32.6 ms token.
    Tried, all bit-identical where they pass parity, none kept: x as float4
    register loads, +1 to 3% interleaved with overlapping ranges; the window
    base specialized to constant 0 / 16 bodies, 50 to 70% slower (likely
    register spills); the FP16 table read from device memory instead of
    staged, about 40% slower; 16 or 8 rows per threadgroup instead of 32,
    +16 to 21% on `wo_ud` (2304 x 3072) and +31 to 37% on `wr_du` (384 rows)
    but -3 to -34% elsewhere, about 0.3 ms per token if chosen per shape,
    under the 2% floor. A three-weight table like the expert rtr1 one does
    not apply (rtr1 is nearly unused here, and wider-code multi-weight
    tables exceed threadgroup memory). The next GEMV lever would have to
    cut gathers per weight, which is a format question, not a kernel one.
    Lossless form of that, tried and NOT kept (2026-11-22): a two-column
    table derived from each direct code's LUT at load, entry W of
    2^(12 + k) = (lut[(W >> k) & 4095], lut[W & 4095]), one device lookup
    per two columns (64 / 128 / 256 KB for rtr2 / rtr3 / rtr4, too large
    for threadgroup memory). Interleaved against `lamq_gemv` at the Mini's
    shapes: -2 to -8% for rtr2 and rtr3, -15 to -34% for rtr4. A device
    gather into a table of that size costs more than two threadgroup
    gathers, so halving the lookups loses. Any further GEMV gain would need
    a lossy re-encode of the weights, which needs its own quality gate.
    Expert pair characterized the same way (2026-11-22, nothing kept): the
    ignored `expert_pair_bandwidth` test in
    `tests/moe_inkling2_lamq_parity.rs` (128 resident experts of one
    format, 16 disjoint 8-slot selections cycled). The release's experts
    use only ht2, ht3 and rtr1 (gate_up 41 / 3 / 56%, down 48 / 8 / 44% of
    3072 expert records). Phase 1: ht2 87 us (122 GB/s), ht3 140 us,
    rtr1 128 to 140 us (41 GB/s); phase 2: 63, 93 and 73 us. Both phases
    are gather-bound like the GEMV. Staging the plain 4096-entry rtr1 table
    in threadgroup memory for phase 1 (16 threadgroup gathers per chunk
    instead of 6 three-weight device gathers) was slower, 151 to 156 us
    against 140, interleaved: the 128 KB three-weight table stays.
    `inkling2_bf16_gemv_multi` is dispatch-bound, not weight-bound: 336
    dispatches per token, 328 of them one segment, on 0.3 to 0.6 MB
    controller matrices (`down_weight` 64 x 4608, `gain_weight` and
    `forget_weight` 2304 x 64). Grouping each controller's gain and forget
    GEMVs (same input) into one dispatch removes 96 dispatches per token,
    48-token logits `cmp`-equal, NOT kept: tslab `profile`, experiment
    `lamq-ctrl-group2` (the first try, `lamq-ctrl-group`, was all dirty
    from another session's Swift build), interleaved, two passes, all 12
    runs clean: 30.47 to 33.22 tok/s against 30.71 to 31.64, medians both
    about 30.8. So 96 fewer dispatches cost under about 0.2% of a token:
    a small dispatch in the real (one encoder) decode is well under 1 us,
    and the dispatch profile's per-dispatch times are its own overhead.
    Cutting dispatch count is not a decode lever here.
15. Workstream C: `lamq-1` (Gemma 4) and `opgraph-packed-1` (Qwen GDN dense)
   on the same kernels, each with its own family quality gate.

No catalog row beyond `runs`, and no change to any family's defaults, without
its gate. The 5.4 GB release's row (`inkling2-mini-lamq54`) is `verified`
(2026-11-22): its memory oracle and quality gate are frozen. Verified means
those gates hold, not that it matches the original (perplexity 5.7391
against 4.7024; re-frozen 2026-11-23 from 5.7382 when decode's trellis
GEMVs moved to the multi-row kernel, `docs/INKLING2_PHASE0.md`).
