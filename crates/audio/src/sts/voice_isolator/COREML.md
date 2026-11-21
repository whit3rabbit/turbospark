# Voice Isolator Core ML streaming step

`crates/audio/tools/export_voice_isolator_coreml.py` exports the realtime
checkpoint (`denoiser_realtime.safetensors`) as one stateful Core ML program
that runs one block of the plain stream (`streaming.Stream`, no windowed
lanes, cache = the 1200-frame attention window), the same semantics as
`stream.rs`.

## Contract

One call = F = 2 new frames (20 ms at 48 kHz, hop 480), look-ahead embedding
fixed at position 2. Minimum deployment target macOS 15 (states).

| Kind | Name | Type, shape | Meaning |
| --- | --- | --- | --- |
| input | `spectrum` | f32 [1, 2, 1025, 2] | (re, im) of rfft(hann_periodic(2048) * frame) for the 2 new frames, as `streaming._analyze` |
| input | `rope_cos`, `rope_sin` | f32 [2, 32] | cos/sin of t * 10000^(-d/32), t = absolute frame index (host computes in f64) |
| input | `mask` | f32 [2, 1201] | additive, 0 visible, -1e4 hidden, over [1199 cached slots oldest first, 2 new frames] |
| state | `k0..k3`, `v0..v3` | f16 [46, 8, 1199, 64] | per attention block: last 1199 frames' rotated keys and values |
| state | `conv0..conv2` | f16 [46, 1024, 3] | per Mamba block: time Mamba's last 3 conv inputs |
| state | `ssm0..ssm2` | f16 [46, 1024, 16] | per Mamba block: time Mamba scan state |
| output | `raw` | f32 [2, 1025, 4] | decoder output of the 2 new frames, before read-ahead |

Each call shifts every k/v state left by 2 and appends the 2 new keys
(rotated at their absolute positions) and values. Zero states plus a mask
that hides empty slots is a fresh stream. Query i (frame t + i) sees cached
slot j when frame t - 1199 + j exists and is at least t + i - 1199, and new
slot j' when j' <= i. When a stream ends on an odd frame, the host feeds a
second frame anyway (its window zero-padded like `_analyze`) and drops its
output; the mask keeps it from the real frame. The host applies the
look-ahead read (frame k uses raw of min(k + 2, last)), `correction`, irfft,
the window and the overlap-add with the squared-window envelope.

Inside the program: the 46 band encoders and decoder heads (packed as
batched matmuls), all 7 blocks, the bidirectional freq Mamba over the bands
(an associative Hillis-Steele scan, 6 steps instead of 46), the band
attention. The encoder LayerNorm normalizes x / 16 with eps / 256, exact in
fp32 and free of fp16 overflow for any input in [-1, 1].

## Command

```sh
uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \
    --with torch==2.7.0 --with safetensors==0.5.3 \
    python -I -u crates/audio/tools/export_voice_isolator_coreml.py \
    --reference /path/to/voice-isolator-c73b3fab81a870410cbac50d2f05f25edc4d10aa \
    --checkpoint ~/models/voice-isolator/denoiser_realtime.safetensors \
    --clip ~/models/voice-isolator/real_clip.npy \
    --out ~/models/voice-isolator-coreml
```

Writes `VoiceIsolatorStep.mlpackage` and `.mlmodelc` (fp16 compute, the
contract file), `VoiceIsolatorStep-fp32.*`, `torch_stream.npy` and
`report.json`. The torch pass and the reference stream take about 10 min
each on the CPU; `--reuse-torch` reuses them to iterate on conversion.

## Measured (M4 Max, macOS 26.5.2, AC power, other sessions active)

Torch StreamStep (f32 states) vs reference `streaming.Stream` (block_frames
2, pushes of 960 samples) over `real_clip.npy` (10 s), final audio after the
replicated host synthesis: max abs diff 4.8e-7, SNR 127.5 dB. With f16
states in torch: max abs 2.9e-4, SNR 70.4 dB, so the f16 states bound the
quality of every Core ML variant.

Core ML stream vs the f32 torch stream over the clip, and per-call latency
(coremltools `predict` with Python overhead, 300 calls after 30 warmup):

| Precision | Units | Max abs | SNR dB | Correction SNR dB | Median ms | p95 ms | Under 20 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | CPU_AND_GPU | 3.7e-4 | 66.8 | 66.4 | 17.9 | 29.1 | median yes, p95 no |
| fp16 | CPU_AND_NE | ANE compile fails (ANECCompile, load error -14) | | | | | no |
| fp16 | ALL | same ANE compile failure, no fallback | | | | | no |
| fp32 | CPU_AND_GPU | 2.8e-4 | 70.6 | 70.2 | 43.0 | 44.8 | no |
| fp32 | CPU_AND_NE | 2.8e-4 | 70.7 | 70.3 | 369.8 | 403.6 | no |
| fp32 | ALL | 2.8e-4 | 70.6 | 70.2 | 42.7 | 43.8 | no |

An earlier fp16 CPU_AND_GPU run (50 calls) gave median 16.3 ms, p95 18.6 ms;
p95 depends on host load. "Correction SNR" compares output minus input,
the part the program computes.

The shift-append costs about 6 ms per call: the same program with a 2-slot
write instead of the shift (a ring buffer, not this contract) measured
median 10.6 ms, p95 11.0 ms on CPU_AND_GPU fp16. Softmax over a masked key
set ignores slot order and the keys carry absolute rotations, so a ring
needs only a write-slot input and a mask in slot order.
