# Multitalker ASR (Frozen E)

NVIDIA Multitalker Parakeet Streaming 0.6B v1 coupled to Nemotron-3
Diarization, ported from the exact NeMo Speech code path that produced the
"Frozen E" meeting-transcription result. Each active speaker gets its own
cache-aware FastConformer/RNN-T stream over the same mixed audio, conditioned
by that speaker's diarization activity through learned speaker kernels. This
is not ordinary ASR with speaker labels attached afterwards.

## Pins

| Component | Source | Revision | sha256 |
|---|---|---|---|
| Multitalker ASR | `nvidia/multitalker-parakeet-streaming-0.6b-v1` (`.nemo`, FP32) | `8749fc71fd6e2d88ef230159bbf2aea69b524ee1` | `afaefe89829e201a0ee22c67a715eb5a467bd06c0704cc1579fbfb370fb5be73` |
| Diarizer | `nvidia/Nemotron-3-Diarization` (`.nemo`, bfloat16 values) | `f667ed73aee57d40cc39428eb768b4fd87a0a29e` | `867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d` |
| Reference code | `NVIDIA-NeMo/Speech` | `218d0b443441b7d8aa1c832f47df310f186cf0a4` | driver `7c34077ce6f4286e5ad3f390994683ef957d965bb93e05dea1a51fb2c8b0b56e` |

The `.nemo` files are converted locally (tests never download):

```sh
uv sync --project crates/audio/scripts/nemo_218d0b4 --python 3.12
PY=crates/audio/scripts/nemo_218d0b4/.venv/bin/python
$PY -I crates/audio/scripts/convert_multitalker_nemo.py convert \
  --nemo ~/models/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo \
  --out ~/models/multitalker-parakeet-streaming-0.6b-v1-f32
$PY -I crates/audio/scripts/convert_multitalker_nemo.py convert-diar \
  --nemo ~/models/Nemotron-3-Diarization-f667ed73/Nemotron-3-Diarization.nemo \
  --hf-config ~/models/Nemotron-3-Diarization-f667ed73/config.json \
  --out ~/models/Nemotron-3-Diarization-f667ed73-turbospark
```

Both conversions are deterministic. The Multitalker output is
`model.safetensors` sha256 `4a49a63aa71af1173d2dc56760ef75f6d505e08278b784dbb3b6c4a6a65ee65b`
(659 tensors, F32), the diarizer output
`27912e27e8f31d21cae10c9ce076ca01a7b99db78ee563dd5cbd3583801df33d` (363
tensors). Each directory carries a `conversion_manifest.json`.
`check-diar` confirms every tensor of NVIDIA's HF `model.safetensors` has an
exact `.nemo` source. The `mlx-community/Nemotron-3-Diarization` conversion
comes from an older revision (`723e19c6`) and is not a Frozen E source.

## Reference and the mlx-audio rule

The crate's normative reference is mlx-audio 0.5.7, which has no
speaker-kernel model (its `speaker_streaming.py` masks features instead).
This family follows NeMo Speech 218d0b4 instead, and every golden comes from
that code running on CPU in the pinned environment above.

## Architecture

The checkpoint is a fine-tune of Nemotron's cache-aware FastConformer, so
the encoder, predictor and joint are [`nemotron_asr`](../nemotron_asr/)'s,
with streaming methods added there. This module adds the rest:

- Frontend: 128 log-mel bands, 512-point FFT, 400-sample Hann, 160 hop,
  pre-emphasis 0.97, log guard 2^-24, `normalize: NA`, centered STFT with
  zero padding (not the reflect padding the Nemotron 3.5 path uses), and the
  final centered frame zeroed. Features are computed over the whole file
  first, as NeMo's streaming buffer does.
- Speaker kernels, applied to the layer-0 input of each chunk:
  `x += spk_ff(x * spk)` then `x += bg_ff(x * bg)` on the updated `x`, where
  each kernel is Linear, ReLU, Linear (1024 wide). Short masks are left-padded
  (1.0 for the target, 0.0 for background), long ones keep their last rows.
- Streaming geometry for `att_context_size = [70, 13]`: every step is a
  9-frame pre-encode cache (zeros before the start) plus 112 new frames,
  subsampled to 16 rows of which the first 2 are dropped, leaving 14 encoder
  frames. The attention cache keeps 70 rows and the causal conv 8 frames per
  layer. Because the diarizer's pre-encode is feature stacking, the NeMo
  driver forces this `pad_and_drop_preencoded` geometry from step 0.
- Decoding: greedy RNN-T (`max_symbols` 10) resumed per speaker from its
  last token and LSTM state, emitting global frame indices.
- Text: SentencePiece decode, then NeMo's
  `decode_tokens_to_str_with_strip_punctuation`, which deletes one
  whitespace character before any punctuation mark in the vocabulary
  (`and 'cause` becomes `and'cause`). The converter writes that set to
  `config.json` as `supported_punctuation` and loading refuses a config
  without it. The rule glues some words together, which costs cpWER, but
  Frozen E's numbers include it.

The [scheduler](scheduler.rs) ports `perform_parallel_streaming_stt_spk`.
The diarizer steps on the ASR chunk minus its pre-encode cache, speakers 4
and up are zeroed, the gate keeps speakers whose max over the last 28
frames exceeds 0.5, and each active speaker gets its own target plus a
background built from the other active speakers. The step runs with one
shared pre-encode, and inactive speakers keep stale state.
[seglst.rs](seglst.rs) ports NeMo's sentence assembly, including its
word-continuation split.

For speed, the diarizer runs one step ahead of transcription and the
active speakers step concurrently on the kernels' worker pool. Each
component performs the same arithmetic in the same order as a sequential
loop, so the output is bit-identical.

## Evidence

All comparisons are against NeMo Speech 218d0b4 on this repo's reference
environment (CPU, FP32).

| Gate | Result |
|---|---|
| NeMo CPU reproduces the frozen CUDA run | Identical SegLST (356 segments). 355/3814 on the 12-clip IS panel and 160/1740 on the matched six, every clip equal to the frozen counts |
| Frontend (whole 120 s clip) | Max error 3.3e-4 (3.2e-5 of RMS), trailing frame zero |
| Speaker kernels (two-speaker step) | 7.2e-7 of RMS |
| Pre-encode and layer-0 input (6 steps) | 9.5e-6 of RMS (values reach ~1500) |
| Encoder output (6 steps) | 9.8e-7 absolute |
| Tokens, token frames, text, decoded and cache lengths (6 steps) | Identical |
| Full IS1004a clip, replayed NeMo diarization | All 108 steps identical; SegLST identical |
| Full IS1004a clip, live coupled diarizer | All 108 steps identical (chunk predictions within 4.2e-6); SegLST identical |
| Frozen AMI panel, live pipeline | SegLST identical to NeMo CPU on all 12 clips (356 segments: speakers, millisecond times, words). 160/1740 on the matched six (9.20%) and 355/3814 on the 12-clip panel (9.31%), every clip equal to its frozen count |
| Frozen AMI panel, diarizer encoder on Metal (f32, `crates/runtime/src/nemotron_diar_metal.rs`) | Encoder within 1.4e-5 of RMS of the CPU on a full 542-row window; coupled chunk predictions within 2.2e-6 of NeMo over 54 steps, no 0.5 decision flips, identical cache and FIFO lengths; panel SegLST identical to NeMo on all 12 clips, 160/1740 and 355/3814 |
| Frozen AMI panel, ASR encoder layers on Metal too (f32, `crates/runtime/src/multitalker_metal.rs`) | Encoder within 2.4e-5 of RMS of NeMo and the CPU over the 6 NeMo fixture steps; batched slots within 1.7e-4 of independent CPU streams (a silent target with an active background moves the CPU itself 8.5e-5 for a one-ulp input change). Greedy near-ties flip, so the SegLST differs from NeMo on 9 of 12 clips, but cpWER is 160/1740 and 355/3814, equal to Frozen E (two clips move by one error in opposite directions). Precise Metal math does not remove the flips (158/1740, 364/3814) |

Tolerances on large-magnitude tensors are RMS-scaled (`max |a - b| / rms(b)
<= 1e-4`) because f32 accumulation order differs from torch's blocked GEMM.

CPU throughput on an Apple M4 Max (release build, shared host with load
around 9 to 12, not a formal benchmark), live diarizer, the 12-clip panel
interleaved per clip: before the tiled `ops` kernels, 1929 s in total
(mean RTF 1.34). After, 699 s (mean RTF 0.49, worst clip 0.63), with the
SegLST unchanged on every clip. The diarizer is about 60% of the time,
mostly `linear` over its 542-row spkcache + FIFO + chunk window.

Adding the worker pool, the diarizer running one step ahead and
concurrent speakers (medians of three interleaved runs on a quiet host,
2026-10-17): IS1000a_00510 (4 speakers) 40.0 s (RTF 0.33) against 54.5 s
with the tiled kernels alone and 153.9 s before them. IS1004a_00140 (2
speakers) 35.0 s (RTF 0.29) against 47.5 s and 142.7 s. The SegLST is
identical across all three builds.

With the diarizer encoder on Metal (`crates/runtime` example
`frozen_e_metal`, ASR still on the CPU, same host): IS1000a_00510 16.9 and
16.7 s (RTF 0.14) against 36.9 s with the CPU diarizer from the same
binary, and RTF 0.13 to 0.17 across the 12-clip panel.

With the ASR encoder layers on Metal as well (`--asr-device metal`, with
the speaker kernels and the RNN-T decode still on the CPU), interleaved with
the CPU ASR from the same binary while another session compiled on the
CPU (load 6 to 10): IS1000a_00510 11.7, 12.0 and 11.7 s (RTF 0.10)
against 17.2, 19.9 and 21.7 s. The device runs every active speaker in
one command buffer per step and keeps each speaker's caches resident.

## Tests

| Test | Needs |
|---|---|
| `prompt_free_parse_*`, `align_mask_*`, scheduler and seglst known answers | Nothing |
| `frozen_e_step_geometry_known_values`, `speaker_kernels_match_nemo`, `stream_steps_match_nemo` | `TURBOSPEECH_MULTITALKER_MODEL` |
| `frontend_matches_nemo`, `scheduler_with_replayed_diarization_matches_nemo_full_clip` | plus `TURBOSPEECH_FROZEN_E_TRACE` |
| `scheduler_with_live_diarization_matches_nemo_full_clip` | plus `TURBOSPEECH_NEMOTRON_MODEL` |
| `crates/runtime/tests/nemotron_diar_metal.rs`, `crates/gpu/tests/nemotron_diar_parity.rs` | A Metal device, plus `TURBOSPEECH_NEMOTRON_MODEL` for the runtime test |
| `crates/runtime/tests/multitalker_metal.rs`, `crates/gpu/tests/conformer_stream_parity.rs` | A Metal device, plus `TURBOSPEECH_MULTITALKER_MODEL` for the runtime test |

`TURBOSPEECH_FROZEN_E_TRACE` is a trace directory from
`scripts/nemo_multitalker_trace.py` (its `meta.json` names the clip). The
SegLST scorer and the frozen panel references live in
[`crates/bench`](../../../../bench/src/meeting/mod.rs).

## Reproduce

```sh
PY=crates/audio/scripts/nemo_218d0b4/.venv/bin/python
# Frozen clips from the AMI corpus, checked against the frozen sha256.
$PY -I crates/bench/scripts/prepare_frozen_e_clips.py --ami ~/models/frozen-e/ami \
  --manifest ~/models/frozen-e/evidence/evaluation_manifest.json --out ~/models/frozen-e/clips
# NeMo trace for one clip.
$PY -I crates/audio/scripts/nemo_multitalker_trace.py \
  --driver ~/models/frozen-e/nemo-cpu/speech_to_text_multitalker_streaming_infer.py \
  --asr <multitalker .nemo> --diar <diarizer .nemo> \
  --wav ~/models/frozen-e/clips/IS1004a_00140_00260.wav --out <trace dir>
# Rust pipeline and first-divergence report.
cargo run -p turbospark-audio --release --example multitalker_meeting -- \
  --model ~/models/multitalker-parakeet-streaming-0.6b-v1-f32 \
  --diar ~/models/Nemotron-3-Diarization-f667ed73-turbospark \
  --clips ~/models/frozen-e/clips --out <run dir>
python3 -I crates/audio/scripts/diff_multitalker_trace.py <run dir>/<id>.trace.jsonl <trace dir>/trace.jsonl
cargo run -p turbospark-bench --release --bin meeting_eval -- score --seglst <run dir>/seglst.json --clips matched6
```

## Open

- Runtime session, FFI, Swift and catalog `can_run` (Phase 5).
- The speaker kernels, pre-encode and RNN-T decode still run on the CPU.
  Both encoders run on Metal (opt-in, tolerance-gated, not bit-exact).
  Only F32 weights load. No GPU memory or power measurement yet.
- A live incremental frontend. The port computes features over the whole
  recording, as the frozen replay did.
- Recordings outside the frozen panel.
