# LookAtMyQuant (LAMQ) formats

STATUS: **runs on the 5.4 GB Inkling release, quality gap open.** This page
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
| `inkling2-packed` | `shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB` (1.63 bpw), `-8.9GB` (2.69 bpw) | planned first (workstream A) |
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
   The runtime does not open the install yet.
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

   KL is flat over position (0.12 to 0.22 per 29-token band), so this reads
   as weight error, not a position-dependent defect. Most misses are the
   reference's second choice. Decode is 64 ms per token against 28 ms for
   the BF16 install (untuned kernels, unpaired, not a protocol number).
   OPEN: the authors report `free_KL` 0.0895 and 89.2% top-1 for this file
   (English/chat/code windows against the BF16 teacher, their harness). The
   protocols differ, so the gap is not a measured defect, but it is not
   explained either. First suspect: the BF16 rounding of the F32 steering
   above (the authors' runtime precision for it is not stated). Not yet run:
   the 8.9 GB release, the memory oracle, a tslab profile, sampled
   generation. No catalog row.
6. Workstream B: Inkling 2 GGUF intake for the published ggml-org and unsloth
   files (stock types).
7. Workstream C: `lamq-1` (Gemma 4) and `opgraph-packed-1` (Qwen GDN dense)
   on the same kernels, each with its own family quality gate.

No catalog row, and no change to any family's defaults, without its gate.
