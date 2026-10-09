# Whistle

Native Rust CPU speech recognition for Cactus Compute's Whistle checkpoint.
This audio-crate API loads the original CQ `.cact` file and returns greedy
transcripts, language codes, signed token IDs, and acoustic embeddings.

## Quickstart

Run these commands from the repository root. Download the immutable checkpoint
once; tests and inference perform no network requests.

```sh
mkdir -p /tmp/turbospark-whistle-reference
curl -fL https://huggingface.co/Cactus-Compute/whistle/resolve/b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact \
  -o /tmp/turbospark-whistle-reference/whistle.cact
shasum -a 256 /tmp/turbospark-whistle-reference/whistle.cact
# b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb

cargo run --release -p turbospark-audio --example whistle_transcribe -- \
  /tmp/turbospark-whistle-reference/whistle.cact /path/to/mono-16000.wav en
```

Omit `en` to detect one of `en`, `de`, `fr`, `es`, `it`, `nl`, or `pl`.
Audio must contain 160 to 480,000 finite mono samples at 16 kHz, normalized
to `[-1,1]`. The caller owns resampling and clip segmentation. The example
checks the WAV format and reports exhaustion of the 318-token context budget.

```rust,no_run
use std::path::Path;
use turbospark_audio::models::stt::whistle::Whistle;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let wav = turbospark_audio::read_wav_f32(Path::new("mono-16000.wav"))?;
assert_eq!((wav.sample_rate, wav.channels), (16_000, 1));
let model = Whistle::load(Path::new("whistle.cact"))?;
let result = model.transcribe(&wav.samples, Some("en"), 100)?;
println!("{}", result.text);
assert!(!result.truncated);
let embeddings = model.encode(&wav.samples)?;
assert_eq!(embeddings.width, 512);
# Ok(()) }
```

`None` detects language from the initial decoder logits. Language markers are
resolved from the checkpoint tokenizer. Exact digital silence returns an empty
transcript; its language is empty unless supplied. This is not a general VAD.
Each call owns fresh decoder state, including causal filters, KV, and engrams.
`tokenize` and `decode_tokens` expose the checkpoint's BPE and byte fallback.

## Architecture and references

The [model card](https://huggingface.co/Cactus-Compute/whistle/tree/b358ddadd89b7a713b5aa131f23032d3cca1b251)
and [release article](https://cactuscompute.com/blog/whistle) describe the
upstream model. This port has no Python, C++, Needle, or Metal runtime dependency.

| Component | Supported deployment geometry |
|---|---|
| Container | Needle 3, tag `0x05e12a84`, 681 tensors, embedded tokenizer |
| Exported weights | 55,151,433 tensor elements, excluding mel filters, permutations, manifest, and tokenizer; checkpoint 16,919,407 bytes |
| Quantization | CQ2/CQ4, 128-element normalized Hadamard groups; A8 activation rounding |
| Frontend | 16 kHz, 80 mel bins, FFT 512, symmetric Hann 400, hop 160, checkpoint mel bank |
| Stem | Three 3x3 stride-2 convolutions, 128 channels, CQ pointwise projections; time and frequency reduced by eight |
| Encoder | Eight bidirectional Conformer blocks, width 512, four mHC residual lanes |
| Decoder | Eight causal text blocks, width 512, four lanes, audio cross-attention, engrams in layers 3 and 7 |
| Attention | Eight heads, query/key width 48, value width 64; two self-attention KV heads |
| Output | 8,199 tokenizer pieces, seven language markers, text vocabulary 8,192; tied CQ head |
| Embeddings | Width 512, approximately one row per 80 ms, final frame padded by the stem |

The loader refuses other geometry, tensor shapes, unsupported quantization,
overlapping or out-of-bounds payloads, non-finite weights, and malformed
tokenizers. Model outputs must be finite. CQ weights are rotated codebook
indices with FP16 group norms, so the shared MLX affine decoder cannot read them.

Whistle is absent from the project's pinned mlx-audio 0.5.7 reference.

The pinned [Needle JAX model](https://github.com/cactus-compute/needle/blob/af2654dffa4987f7e2cbbacdf59c2f854fd18302/needle/model/architecture.py),
[exporter](https://github.com/cactus-compute/needle/blob/af2654dffa4987f7e2cbbacdf59c2f854fd18302/needle/model/export.py),
and [quantizer](https://github.com/cactus-compute/needle/blob/af2654dffa4987f7e2cbbacdf59c2f854fd18302/needle/model/quantize.py)
define the text/CQ contracts. They omit the speech frontend and encoder.

The pinned released CPU library supplies the independent speech operator
oracle, replacing the usual pinned Python reference for this family.
See [upstream issue 169](https://github.com/cactus-compute/needle/issues/169).

## Verification

Default checks need no checkpoint or external library:

```sh
cargo test -p turbospark-audio stt::whistle --lib
cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio
```

Checked-in fixtures cover CQ row reconstruction, CQ projection, a Hadamard
feed-forward block, the log-mel frontend, and the real tokenizer. Absolute
tolerances are `1e-6`, `3e-6`, `4e-6`, and `2e-4`, respectively. Additional
tests cover corrupt archives, signed IDs, PCM bounds, and causal filter order.

The checkpoint test compares two English transcripts with the released oracle,
with both forced and detected language. It also checks token truncation and
silent-audio behavior. On macOS, prepare its two WAVs:

```sh
say -v Samantha -r 145 -o /tmp/turbospark-whistle-reference/speech.aiff \
  'The quick brown fox jumps over the lazy dog. This is a test of speech recognition.'
afconvert /tmp/turbospark-whistle-reference/speech.aiff \
  /tmp/turbospark-whistle-reference/speech.wav -f WAVE -d LEI16@16000 -c 1
curl -fL https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/tests/jfk.flac \
  -o /tmp/turbospark-whistle-reference/jfk.flac
shasum -a 256 /tmp/turbospark-whistle-reference/jfk.flac
# 63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715
afconvert /tmp/turbospark-whistle-reference/jfk.flac \
  /tmp/turbospark-whistle-reference/jfk.wav -f WAVE -d LEI16@16000 -c 1
TURBOSPARK_WHISTLE_REFERENCE=/tmp/turbospark-whistle-reference \
  cargo test -p turbospark-audio stt::whistle --lib --release -- --include-ignored
```

These runs establish two English checkpoint witnesses. They do not establish
corpus WER, other-language quality, or parity of every intermediate tensor.
Deep CQ/A8 stages have rounding drift against the optimized release oracle.
No throughput, peak-memory, Metal, or release qualification is claimed.

The implementation supports offline greedy transcription and encoding only.
Streaming, word timestamps, keyword biasing, beam search, and the upstream
spectral silence gate are not implemented. Runtime sessions, model catalog,
CLI install routing, FFI, and Swift exposure remain separate integration gates.

## Regenerate operator fixtures

The research script uses private symbols from one exact macOS arm64 library.
It verifies both input SHA-256 digests before any ABI calls, builds a temporary
shared library, and removes it afterward. Its platform requirement applies to
fixture generation only. Install LLVM's `llvm-nm` and Xcode command-line tools.

```sh
curl -fL https://huggingface.co/Cactus-Compute/needle3/resolve/2ae11323dc000f5e70c49f7403efa6af12ba9e67/macos-arm64/libneedle.a \
  -o /tmp/turbospark-whistle-reference/libneedle.a
python3 crates/audio/src/stt/whistle/generate_reference.py \
  /tmp/turbospark-whistle-reference/whistle.cact \
  /tmp/turbospark-whistle-reference/libneedle.a \
  crates/audio/src/stt/whistle/fixtures
```

[Fixture manifest](fixtures/manifest.json) records checkpoint/runtime pins and
file digests. `ops.cact` is a modified subset of the checkpoint: eight encoder
Q rows, one Hadamard MLP, two permutations, mel filters, and the tokenizer.
It is deliberately refused by the full model loader. Goldens are little-endian
FP32 values generated by the independent runtime, never by this Rust port.

## License

Rust code uses the repository's [MIT license](../../../../../LICENSE).
The checkpoint-derived fixtures retain Cactus Compute attribution and the
[Apache-2.0 license](fixtures/LICENSE-APACHE); see [fixture notices](fixtures/NOTICE).
