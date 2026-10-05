# Granite Speech 5.0 TurboCTC

Encoder-only English CTC recognition, ported from `mlx_audio/stt/models/granite_speech5_ctc/` at mlx-audio 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Pinned profile

| Profile | Hugging Face repository | Revision | Format |
|---|---|---|---|
| 470M TurboCTC | `ibm-granite/granite-speech-5.0-470m-turboctc` | `947f59af40db9791170a0628cf0f3f4812d720f1` | BF16 safetensors, tokenizer JSON |

The pinned checkpoint loaded with mlx-audio 0.5.7 and returned `the quick brown fox jumps over the lazy dog` on the local 16 kHz smoke WAV. Rust loaded the same revision and returned the exact same transcript. The 946,180,704-byte checkpoint directory was removed after both runs. This single clip does not establish recognition quality or product readiness.

## Inference contract

- Input: mono 16 kHz PCM samples as `f32`.
- Frontend: centered 512-point STFT, 400-sample periodic Hann window, HTK 80-bin power-mel, global 8 dB floor, log normalization, replicate-padded deltas, pair stacking.
- Encoder: 16 block-local Conformer layers, subsampling in layers 0 and 1, midpoint CTC self-conditioning, tied CTC output head.
- Decode: framewise argmax, CTC repeat collapse and blank removal, then the checkpoint's tokenizer.
- The `-nc` profile is not included. Its license and tokenizer differ and need a separate pinned reference run.

## Reference source

The module follows `granite_speech5.py`, `config.py`, and `README.md` from the pinned mlx-audio commit. The Python reference computes the precise mel filterbank in f64, casts its weights to f32, and uses HTK scaling without Slaney area normalization. Rust keeps that family-specific contract local rather than changing the shared audio crate.

## Verification

The mlx-audio and Rust checkpoint smokes used revision `947f59af40db9791170a0628cf0f3f4812d720f1` and the same generated 16 kHz speech WAV; both returned `the quick brown fox jumps over the lazy dog`. The Rust frontend matches the pinned Python `compute_features` output on a deterministic 1,280-sample input. Regenerate that small fixture with `uv run --project ../mlx-audio --extra stt --locked python crates/audio/scripts/generate_granite5_features_fixture.py`; SHA-256 is `19e67ca427466486ac6eb2dfb0c0d5ba47a28f943fa8e2f98b9ec31ac740cb0c`. The checkpoint download was removed after inference. Encoder-stage comparisons, broader task quality, runtime/catalog, FFI, Swift, performance, and memory gates remain open.

The midpoint self-conditioning and final CTC projection now process eight
rows at a time. The midpoint still computes the full-vocabulary softmax and
conditioning projection for each row, while the final projection collapses
to token IDs without retaining a full logits matrix. Synthetic tests match
both previous full-matrix calculations across batch boundaries. Checkpoint
parity for these changes remains open.
