# Audio jobs for `tslab --audio-job`

Each file is one `AudioJob` (`crates/lab/src/workload.rs`), run once warm
and once measured by an audio arm. The same job and seed on both arms of an
A/B make the output digest a function of the arithmetic alone.

- `music-short.json`: MiniMax Music 3, 4 s, 4 flow steps, seed 7. Short
  enough to A/B in minutes; the AR stage dominates at this length.
- `speak-short.json`: one sentence for a Kokoro install, seed 7. A MegaTTS3
  install needs its own job with `reference_wav` and `consent: true` (voice
  cloning needs consent for that exact reference), and takes `steps`.

A transcription job names a 16 kHz mono WAV (the lab does not resample on
the runtime's behalf) and is not checked in, because no clip with clear
redistribution terms is in the tree:

```json
{"task": "transcribe", "wav": "/path/to/clip-16k-mono.wav", "language": "en"}
```
