# underdog_woof

Underdog Woof, a small English text-to-speech model with two fixed voices
(Juno, Mira). **First slice only**: checkpoint reader, dequantized speech-stage
weights and the frozen tensor-hash gate. No forward pass yet. The plan, gates
and open work are in [docs/UNDERDOG_WOOF.md](../../../../../docs/UNDERDOG_WOOF.md).

## License

Underdog Non-Commercial License 1.0. Non-commercial use only, credit "Voice by
Underdog Woof", no training or distillation on the model or its audio. Commercial
use needs a license from Underdog (license@conway.tech). Do not ship this family
before the maintainer has asked Conway whether a Rust reimplementation is
acceptable.

## URLs

- Model: <https://huggingface.co/ConwayResearch/Underdog-Woof>, revision
  `be702960539777fc3f698e54335276a8391b00f9`.
- Reference code: that repo's `underdog_woof/runtime.py` (NumPy only). There is
  no mlx-audio implementation, so this is the normative reference for the family.
- Frozen hashes and MLX/Core ML qualification evidence:
  `cowWhySo/Underdog-Woof-MLX`.

## Architecture

Speech stage: 8 layers, width 448, 7 heads, feed-forward 1216, flow head width
320 and depth 4, 32 latent channels at 25 frames per second, 22,747,584 weights
in 45 matrices. Decoder: 2-layer windowed transformer plus a causal SEANet
adapted from Kyutai's Mimi. 24 kHz mono output. Weights ship as int8 with a
per-row f32 scale and are dequantized to f32 at load.

## Verified stages

| Gate | State |
| --- | --- |
| `.npz` reader, stored members only | Unit tests against synthetic archives |
| f16 to f32 | Exact against the formula for all 65,536 bit patterns |
| `lin_w`, `deq` | Hand-checked cases computed with NumPy, plus refusals |
| File SHA-256 and 60 dequantized tensor hashes | Passes on the real file (witness-gated) |
| Hash gate has teeth | A 1-ulp scale error fails the 42 int8 tensors, one corrupted weight fails one |
| RNG, speech stage, decoder, normalizer | Not started |

The gate needs `woof_int8.npz`, found through `TURBOSPARK_UNDERDOG_WOOF_NPZ`
(fallbacks `~/models/underdog-woof/woof_int8.npz`, `~/models/woof_int8.npz`) and
skips with a message when it is absent. Tests do not download anything.

## Regeneration

`testdata/tts_underdog_woof/tensor_hash_reference.json` is copied from
`cowWhySo/Underdog-Woof-MLX` (`tensor_hash_reference.json`), produced by
`verify_source_integrity` in its `mac_validate.py`. Re-freezing it means rerunning
that script against a new upstream revision and bumping `SOURCE_REVISION` and
`SOURCE_WEIGHTS_SHA256` together.

## Limits

The reader does not verify per-member CRC-32. Integrity rests on the whole-file
SHA-256 and the tensor hashes. Free-running waveforms are not sample-exact against
the reference by design of the model, so no gate may require that.
