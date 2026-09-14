---
title: 'Python Reference: Oracle Generators and Vision Probes'
description: 'Public API of the scripts/ that probe reference implementations and emit Rust golden fixtures, plus the vision Phase 0 tower probes and the MTP bisection driver.'
diataxisType: reference
---

<!-- generated: python lane, signal: scripts/*.py (20 tracked files; no pyproject.toml/setup.py/setup.cfg) -->

These scripts PROBE a reference implementation (mlx, mlx-vlm) and emit golden
fixtures or measured facts for the Rust test suite, rather than transcribing a
format description: a decoder and the doc it was written from can be wrong
together, and an independent implementation's output cannot agree with a wrong
unpacker by accident. No script defines `__all__`; every module-level name
without a leading underscore is public. Functions marked **(no docstring)**
are public but undocumented.

## `mlx_1bit_oracle.py` — 1-bit affine oracle generator

Regenerates `crates/compute/tests/generated/quant_1bit_oracle.rs`. Recovers
the MLX 1-bit affine layout by PROBING mlx's own `mx.dequantize` on real
published bytes, rather than transcribing a format description. Costs a few
tens of KB of RANGED reads off a 4.8 GiB remote file (~5 s), so it needs no
local checkpoint.

| Constant | Value |
|---|---|
| `REPO` | `prism-ml/Bonsai-27B-mlx-1bit` |
| `FILE` | `model.safetensors` |
| `URL` | derived from `REPO`/`FILE` |
| `TENSOR` | `language_model.model.layers.0.linear_attn.in_proj_a` |
| `GROUP_SIZE` | `128` |
| `BITS` | `1` |
| `ORACLE_ELEMENTS` | `256` |

### Functions

```python
def fetch(start, end_inclusive)
```

**(no docstring)**

```python
def main()
```

**(no docstring)**

## `mlx_2bit_oracle.py` — 2-bit affine oracle generator

Regenerates `crates/compute/tests/generated/quant_2bit_oracle.rs`. The
sibling of `scripts/mlx_1bit_oracle.py`, at TWO bits, against
`prism-ml/Ternary-Bonsai-27B-mlx-2bit`. Same rule and same reason: recover
the MLX affine layout by probing mlx's own `mx.dequantize` on real published
bytes. Costs a few tens of KB of RANGED reads off an 8.5 GiB remote file
(~5 s).

| Constant | Value |
|---|---|
| `REPO` | `prism-ml/Ternary-Bonsai-27B-mlx-2bit` |
| `REVISION` | `70f75f3ad081ab840a42f3304c02c27e7f89bfb7` |
| `FILE` | `model.safetensors` |
| `URL` | derived from `REPO`/`REVISION`/`FILE` |
| `TENSOR` | `language_model.model.layers.0.linear_attn.in_proj_a` |
| `GROUP_SIZE` | `128` |
| `BITS` | `2` |
| `ORACLE_ELEMENTS` | `256` |

### Functions

```python
def fetch(start, end_inclusive)
```

**(no docstring)**

```python
def main()
```

**(no docstring)**

## `qwen3vl_vision_oracle.py` — qwen3_vl preprocessing golden fixtures

Probes mlx-vlm's qwen3_vl vision preprocessing and emits Rust golden
fixtures. The reference is PROBED, never transcribed: every number in
`crates/vision-io/tests/generated/` is produced by calling the vendored
mlx-vlm code, so a fixture cannot encode this port's own misreading of it.
Modes: `smart_resize`, `preprocess`, `pos_embed`, `rope_freqs`, `mrope`.

Notable constants: `REAL_PATCH = 16`, `REAL_MERGE = 2`, `REAL_TPS = 2`,
`REAL_MIN_PIXELS = 65536`, `REAL_MAX_PIXELS = 16777216`, `ROPE_HEAD_DIM = 72`,
and marker ids `VISION_START = 151652`, `IMAGE_PAD = 151655`,
`VIDEO_PAD = 151656`. Case tables `PREPROCESS_CASES`, `POS_EMBED_GRIDS`,
`ROPE_GRIDS`, `SAMPLE_STRIDE = 97` and the emitted-file banner `BANNER`
control fixture coverage; `OUT_DIR` is
`crates/vision-io/tests/generated`.

### Functions

```python
def synth_image(height: int, width: int, seed: int) -> "np.ndarray"
```

> A deterministic, non-flat, non-random (C, H, W) uint8 image.
>
> Deterministic so the fixture is reproducible; non-flat because a constant
> image is invariant under every resize kernel and would prove nothing; not
> white noise because a resample of noise is dominated by aliasing and hides
> a wrong filter window under plausible-looking numbers.

```python
def rust_u8_slice(values) -> str
```

**(no docstring)**

```python
def rust_f32_slice(values) -> str
```

**(no docstring)**

```python
def write(path: Path, mode: str, source: Path, body: str) -> None
```

**(no docstring)**

```python
def emit_smart_resize(mod, source: Path) -> None
```

**(no docstring)**

```python
def emit_preprocess(mod, source: Path) -> None
```

**(no docstring)**

```python
def emit_pos_embed(mod, source: Path) -> None
```

**(no docstring)**

```python
def emit_rope_freqs(mod, source: Path) -> None
```

**(no docstring)**

```python
def mrope_cases()
```

> `(name, ids, grids)`. Ids other than the two markers are arbitrary text.
>
> `grids` are `(t, h, w)` in PATCHES; the walk divides the spatial axes by
> the merge size itself.

```python
def emit_mrope(mod, source: Path) -> None
```

**(no docstring)**

```python
def main() -> int
```

**(no docstring)**

## `vision_tower_probe.py` — vision tower activation and INT4 probes

Vision Phase 0 probes: activation magnitude through the qwen3_5 vision
tower, and INT4-transcode quality, against the REAL reference
implementation (never a reimplementation) vendored at `../mlx-v/mlx-vlm`.
Stubs the top-level `mlx_vlm` package in `sys.modules` before importing, so
loading `mlx_vlm.models.qwen3_vl.vision` does not execute the real
`mlx_vlm/__init__.py` dependency chain. Needs only the `vision_tower.*`
tensors (`fetch_vision_tower.py`), never the text trunk.

### Functions

```python
def load_vision_model(vendor_root: str)
```

**(no docstring)**

```python
def load_vision_config(VisionConfig, config_path: str)
```

**(no docstring)**

```python
def load_tower_weights(mx, tower_dir: str) -> dict
```

> Both published dtypes, both landing in FP16.
>
> `prism-ml/Bonsai-27B-mlx-1bit` ships the tower F16 and
> `mlx-community/Qwen3.8-27B-4bit` ships the SAME 333 tensors at the SAME
> shapes in BF16. The install's repack converts the second to FP16, so a
> parity comparison has to run the reference on the converted values.
> Values above 65504 would become `inf` and the walk refuses those by name.
> `numpy` cannot decode BF16, so it is read as uint16 and shifted into the
> top half of an f32 -- which is what BF16 IS.

```python
def build_model(mx, VisionModel, vision_config, weights: dict)
```

**(no docstring)**

```python
def preprocess(mx, Qwen3VLImageProcessor, vision_config, image_path: str)
```

**(no docstring)**

```python
def run_forward(mx, model, pixel_values, grid_thw, dtype = None)
```

**(no docstring)**

```python
def mode_activation(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor)
```

**(no docstring)**

```python
def mode_int4(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor)
```

**(no docstring)**

```python
def mode_dump(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor)
```

> Write the reference's per-stage tensors for `vision_tower_parity.rs`.
>
> FOUR STAGES, not just the merger: patch embed, block 0, the last block,
> and the merger output. A convention that shows as 0.99 at the merger
> localizes immediately when block 0 is exact and the last block is not.
>
> THE PATCH ROWS ARE DUMPED TOO, and that is the load-bearing part. The
> Rust side replays THESE rows rather than preprocessing the image itself,
> so a preprocessing difference cannot read as a kernel gap. Everything is
> written float32 in a plain `{header.json, <name>.bin}` pair rather than
> `.npy`, so the Rust reader needs no format parser.

```python
def main() -> None
```

**(no docstring)**

## `fetch_vision_tower.py` — ranged fetch of vision tower tensors

Fetches ONLY the `vision_tower.*` tensors out of a real HF safetensors
checkpoint, by ranged HTTP, without downloading the (multi-GiB) text trunk.
Reads the checkpoint's own `model.safetensors.index.json` if present
(multi-shard installs) or falls back to a single `model.safetensors` file,
fetches that shard's size-limited JSON header, validates its tensor offsets,
then streams each `vision_tower.*` tensor into a bounded output file.

### Functions

```python
def main() -> None
```

**(no docstring)**

## `make_vision_test_page.py` — deterministic test page renderer

Renders a deterministic dense-text page for the vision activation probe.
Companion to `vision_tower_probe.py`. Exists so the activation numbers in
`docs/VISION_PHASE0.md` can be reproduced rather than merely quoted: the
probe's answer depends on the INPUT as much as on the tower. The default
4064x4064 is the size that closed Phase 0 item 3's open action item
(16,516,096 px, just under the 16,777,216 ceiling, resizing to a 254x254
patch grid). `WORDS` is the rendered word list.

### Functions

```python
def render(width: int, height: int, seed: int, point_size: int) -> Image.Image
```

**(no docstring)**

```python
def main() -> None
```

**(no docstring)**

## `mtp_bisect.py` — MTP drafter bisection

Bisects this port's MTP draft step against the published drafter weights
(`docs/MTP_SPECULATIVE.md` step 3): the head runs, its installed weights
match the checkpoint to 0.992-0.996, its shapes match a trunk
full-attention layer exactly, and its output is nonetheless ANTI-aligned
with the trunk. Constants: `GROUP = 64`, `BITS = 4`.

### Functions

```python
def load_safetensors(path: Path) -> dict
```

> Minimal reader, because `safetensors.numpy` cannot decode BF16.
>
> Hand-rolled rather than routed through mlx on purpose: this script's job
> is to be an INDEPENDENT decoder of the same bytes, and borrowing the
> reference's own loader would weaken that (AGENTS.md Gotcha 48). The format
> is an 8-byte little-endian header length, that many bytes of JSON, then
> the data region, with every `data_offsets` pair relative to its start.

```python
def f16(path: Path) -> np.ndarray
```

**(no docstring)**

```python
def dequant(w: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray
```

> MLX `affine` dequant: uint32 words, `BITS` per element, LSB first.
>
> The same layout `dequant_int4_gemv_simd` reads, stated here rather than
> imported so this script is a genuinely independent decoder.

```python
def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray
```

**(no docstring)**

```python
def corr(a: np.ndarray, b: np.ndarray) -> float
```

**(no docstring)**

```python
def main() -> int
```

**(no docstring)**
