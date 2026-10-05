# Native Music 3 precision fixtures

Generated independently with mlx-audio at
`feb25a37b07923bae556e59111995071d66afa0d` and MLX `0.32.3`.
The generator verifies the pinned Python source hash before evaluating any
model. `manifest.json` records source, generator, and artifact SHA-256 hashes,
shapes, precision, and seeds. No checkpoint download is required.

Regenerate from the TurboSpark root:

```sh
PYTHONPATH=/tmp/turbospark-music3-mlx0323:/tmp/turbospark-music3-reference \
  ../mlx-audio/.venv/bin/python crates/audio/tools/gen_music3_precision_fixtures.py
```

The seven directories `bf16`, `affine8`, `affine6`, `affine4`, `mxfp8`,
`mxfp4`, and `nvfp4` contain converted tiny model trees with BF16 floating
parameters. Each records three emitted AR frames, eight warmup sampling
decisions, hidden states, and a two-step flow waveform. Both AR and flow were
repeated in the same loaded reference model and required exact equality.

`long201` contains a separate tiny model whose vocoder hop is 512 samples,
supplied 201-frame BF16 hidden states, per-chunk condition/noise/latent dumps,
and the raw waveform. Its two crop branches are required to remove samples.
The chunk trace is checked exactly against the reference `_run_flow` method.

`ops.json` contains independent native operations. Every case records its
logical output dtype, shape, and expected f32 values. Dense linear weights
use `[out, in]`. Convolution inputs and outputs use `[channels, time]`;
normal weights use `[out, in, kernel]` and transpose weights use
`[in, out, kernel]`. Attention and rotary arrays use `[batch, time, heads,
dim]`. Packed weights are uint32 values and float-format scales are uint8
values. Affine scale and offset arrays retain their BF16 descriptors.
Gaussian inputs are float32 uniforms in `(-1, 1)`, before `erfinv` and
`sqrt(2)` scaling. `single_round` fields are diagnostic negative controls
that omit required intermediate rounding.

`.f32` dumps store little-endian planar float32 values. BF16 values are
exactly represented in those buffers; this storage choice does not measure
device memory use. These fixtures qualify numerical behavior in small
models. Real checkpoint parity, music quality, throughput, sustained memory,
and application integration require separate checks.
