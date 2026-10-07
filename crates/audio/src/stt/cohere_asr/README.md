# Cohere Transcribe (refused family)

This directory documents the negative finding for
`mlx_audio/stt/models/cohere_asr/` (cohere_asr.py, config.py, tokenizer.py,
audio.py, vad.py) at mlx-audio 0.5.7, source commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Status

Not implemented. The module detects the family's `model_type "cohere_asr"`
distribution and refuses to load it with the documented reason. A wrong or
missing refusal is the failure mode this preserves: without a verified
checkpoint there is no way to validate a Rust decode, and shipping an
unvalidated architecture would produce fluent but incorrect transcripts.

## Investigation record

Both candidate profiles were loaded with unmodified mlx-audio 0.5.7 on the
shared 16 kHz smoke clip during the pinned-inventory investigation:

- Official: `CohereLabs/cohere-transcribe-03-2026` @
  `b1eacc2686a3d08ceaae5f24a88b1d519620bc09`. The Hub returned 403 because
  the verification environment lacks gated access; no weights were ever
  obtained. The revision is recorded for future re-verification.
- Public mirror: `mlx-community/cohere-transcribe-03-2026-mlx-8bit` @
  `a0acb7f93cd32d82c4fbf801b6d8fb39d20c509f`. The `mlx-int8/` subfolder
  loads through the reference loader but produced incoherent text on the
  reference clip, so the conversion itself is rejected as a runnable
  profile.

Because neither candidate can produce a trusted reference transcription,
there is no fixture source, no parity gate, and no pinned profile. The
upstream architecture (Conv2d subsampling encoder, decoder with KV cache,
CohereAsrTokenizer, batching up to 32 clips) is understood but unverified.

## What a future implementation requires

1. Gated access to the official distribution, or a public conversion that
   transcribes the smoke clip correctly through unmodified mlx-audio 0.5.7.
2. A fresh pinned revision with per-file SHA-256 digests recorded here.
3. The full port and verification chain the other families document:
   fixture parity against golden reference tensors, a checkpoint-gated
   transcript test, and the runtime/catalog/FFI/Swift gates.
