# Moonshine Tiny reference profile

The CPU reference follows `mlx_audio/stt/models/moonshine/` from mlx-audio
0.5.7 at `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

| Checkpoint | Revision | Required file | SHA-256 |
| --- | --- | --- | --- |
| [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny/tree/2ccd087d043f77a4a07bd412b6725266355b08f4) | `2ccd087d043f77a4a07bd412b6725266355b08f4` | `model.safetensors` | `867cd2215804859c55aa972d740bd5002be149b4e7526328c895d2408848c736` |
| Same revision | Same revision | `config.json` | `df40d71949d4b3460993b3e7ab743fe957b1a3c3272257c58f373c34dd30177a` |
| Same revision | Same revision | `tokenizer.json` | `6579793438bc4fbafffacf699169ff53e3769c5a0a0f5e71cdee8853e8130deb` |

The local `~/models/moonshine-tiny` witness has matching weights and tokenizer,
but its `config.json` is modified: it adds `pad_token_id` and
`pad_head_dim_to_multiple_of`, and changes `max_position_embeddings` from 512
to 194. Treat measurements on that local directory as diagnostic until the
unmodified pinned config is used. A one-clip transcript and stage parity are
recorded in [the inventory](../../../MODELS.md), but broader quality,
Metal performance, memory, and product gates remain open.

The CPU reference can be run with:

```sh
cargo run --release -p turbospark-audio --example moonshine_transcribe -- \
    <model-dir> <wav>
```

The experimental Metal runner lives in `turbospark-runtime`, leaving
`turbospark-audio` portable. The runner selects CPU at open by default and selects
Metal only with `TURBOSPARK_MOONSHINE_DEVICE=metal`. An unsupported Metal
profile fails at open, and a clip exceeding the Metal attention limit fails
at transcription instead of loading a second CPU model. Metal uses request
sized buffers and reuses them for subsequent clips that fit.
The only accepted explicit device values are `cpu` and `metal`, so a typo
cannot silently change the benchmark arm.

```sh
TURBOSPARK_MOONSHINE_DEVICE=metal cargo run --release \
    -p turbospark-runtime --example moonshine_backend_probe -- <model-dir> <wav>
```

`TURBOSPARK_MOONSHINE_PROFILE=1` reports frontend, transformer, cross-cache,
and decode timing. It uses separate command passes for stage attribution, so
use the ordinary path for end-to-end latency measurement.

The pinned-checkpoint test compares encoder activations and the final
transcript against the CPU reference. It requires a real Metal device:

```sh
TURBOSPARK_MOONSHINE_TEST_MODEL=<model-dir> \
TURBOSPARK_MOONSHINE_TEST_WAV=<wav> \
cargo test -p turbospark-runtime pinned_checkpoint_encoder_and_transcript_parity \
    -- --ignored --nocapture
```

Measure process peak `phys_footprint` with `/usr/bin/time -l` around a
fresh-process run. Release-build timings under concurrent host load are
diagnostic only. Retain this CPU implementation as the independent parity
partner for the Metal path. The GPU path has one known-clip checkpoint witness,
not broader recognition-quality or paired performance evidence. It remains
opt-in and is not connected to the product STT session yet.
The [interleaved benchmark protocol](../../../../../docs/BENCHMARKING.md#moonshine-stt-backend-probe)
uses the uniquely named runtime probe so the portable example cannot replace
the measured executable.
