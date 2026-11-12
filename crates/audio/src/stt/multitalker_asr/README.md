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
| `live_session_with_replayed_diarization_matches_nemo_full_clip` | `TURBOSPEECH_MULTITALKER_MODEL` and `TURBOSPEECH_FROZEN_E_TRACE` |
| `live_session_with_live_diarization_matches_nemo_full_clip`, `crates/runtime` `streams_the_traced_clip_to_nemo_seglst` | plus `TURBOSPEECH_NEMOTRON_MODEL` |
| `live_frames_equal_whole_file_bitwise`, `changes_replay_to_the_same_segments_as_entries` | Nothing |
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

## Live streaming

The batch pipeline above computes features over the whole recording. The
live path steps the same chunks while PCM arrives.

- `LiveFrontend` (`frontend.rs`) turns PCM into the same log-mel frames as
  `MultitalkerAsr::features`. Each frame comes from the same `nemo_mel` call
  on a hop-aligned window of raw samples, so it is bit-identical to its
  whole-file value. Only the last two frames wait for `finish()`, which
  supplies the zero padding and the mask past `samples / hop`. The
  whole-file call refuses input past 2^24 samples (about 17.5 minutes) and
  this path has no such cap.
- `LiveSession` (`scheduler.rs`) drives `CoupledSession`'s steps from that
  frontend. `append()` returns a report per completed chunk and `finish()`
  steps the remaining full chunks and the final partial one, dropping a tail
  shorter than a subsampling group as the batch plan does. Text trails the
  audio by one chunk (1.12 s) plus 16 ms of right context.
- Two differences from the batch run, neither in a value. The diarizer cannot
  run a step ahead, because the next chunk's audio does not exist yet. A
  full chunk steps with `last = false`, because the end of the stream is
  only known at `finish()`. With 14-row chunks `last` changes nothing for a
  full chunk, and the equality tests hold that to NeMo.
- A failed step closes the session, since the caches no longer match any
  prefix of the audio. A non-finite chunk is refused without consuming it.
  A cancel hook is polled before each step.
- `SentenceBuilder::changes` reports the segments new or changed since a
  cursor, keyed by speaker and sentence index. Only a speaker's last
  sentence can change after it is written, so that one is resent when its
  end time or words moved.
- `MultitalkerStreamSession` (`crates/runtime/src/native_audio/`) owns both
  models and the `LiveSession` on one worker thread. `append`, `finish` and
  `cancel` reach it over a channel. A concurrent call is refused as busy and
  cancel is terminal. It takes no heavy-work guard, because a recording can
  outlast a chat reply, and it runs on the CPU. Its `SpeakerStreamUpdate`
  carries the changed `segments` (`speaker`, `index`, `start_seconds`,
  `end_seconds`, `text`), `active_speakers`, `received_seconds`,
  `processed_seconds` and `is_final`. Memory is admitted once at open from
  the two checkpoints' file sizes. That estimate has not been measured.

## Status

Not wired into any user-facing surface. The catalog row is `can_run: false`,
and there is no `ts_multitalker_stream_*` FFI, Swift adapter, CLI command or
server route. The pipeline runs only through the `multitalker_meeting` and
`frozen_e_metal` examples and the runtime session.

Not recommended for end users yet:

- Quality is proven on one panel, the 12 frozen AMI clips. Nothing outside
  it has been tried.
- The Metal diarizer path is intermittent (below), and the Metal ASR path
  differs from NeMo's SegLST on 9 of 12 clips.
- Peak memory and power have not been measured.
- Timings come from a loaded host and are not benchmarks.

## Open

1. **`nemotron_diar_metal` is intermittent.** On a loaded host (load 7 to
   14) `cargo test -p turbospark-runtime --release --test nemotron_diar_metal`
   failed in 3 of the first 4 runs against a 1e-4 bound (measured 2.2e-6 on
   a quiet host): full window max abs diff 6.0e-1, coupled chunk
   predictions max dp 4.5e-2, 2.5e-2 and 2.9e-3. It failed once with the
   status-check fix reverted (1.2e-3), then passed 4 of 4. The 0.5 decision
   flips stayed at 0. The cause is not found. Run-to-run variation on
   identical inputs points at a hazard or race in the diarizer pass.
   `multitalker_metal` and the gpu parity tests passed in every run. Treat
   the diarizer Metal path as unqualified.
2. **Metal default-on gates.** Memory, power and a formal paired benchmark
   on a quiet host on AC power (`docs/BENCHMARKING.md`,
   `.claude/docs/power-measurement.md`). Metal stays opt-in until they pass.
3. **Phase 5 wiring.** The `ts_multitalker_stream_*` FFI inside the ABI
   guard, modeled on `crates/ffi/src/api/audio_stream.rs` (JSON updates,
   busy and cancelled status codes, close refused while a call runs). Then
   the Swift adapter (read `swift/AGENTS.md`, `DESIGN.md` and
   `docs/API_WORKSPACE.md` first), then catalog `can_run: true`. The caller
   supplies 16 kHz mono PCM, so resampling lives in the host. Measure the
   admission estimate before relying on it.
4. **Metal near-tie flips.** With the ASR encoder on Metal the SegLST
   differs from NeMo on 9 of 12 clips (cpWER totals match). Precise math did
   not help (158/1740 and 364/3814). Levers not tried: a tighter GEMM
   reduction order and f32 accumulation in relpos attention.
5. **Still on the CPU.** The speaker kernels, pre-encode (`conv2d`
   `dw_striding`) and RNN-T decode. Profile first. Smaller CPU items not
   done: speaker row-stacking in the CPU encoder, a 16x6 tile, and tuning
   `PAR_MIN_WORK` now that the rayon pool exists.
6. **Recordings outside the frozen panel**, after the decisions are frozen.
7. **Not run:** `cargo check --target x86_64-unknown-linux-gnu -p
   turbospark-audio` (`onig_sys` needs a cross gcc on the dev host), the app
   and release gates, and a `CHANGELOG.md` entry (`docs/RELEASE.md`).
8. **Unrelated and pre-existing:**
   `whisper::parity_tests::whisper_log_mel_matches_old_pipeline_bitwise`
   fails by 1 ULP in release builds only.

Only F32 weights load.
