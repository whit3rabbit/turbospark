# Music Generation Models

Music generation models in TurboSpark produce polyphonic musical compositions and songs from text prompts, tags, and lyrics.

## Implemented Model Families

### MiniMax Music 3 (`minimax_music3`)

- **In-Crate Directory**: [`crates/audio/src/music/minimax_music3/`](../../crates/audio/src/music/minimax_music3/)
- **Documentation**: [`crates/audio/src/music/minimax_music3/README.md`](../../crates/audio/src/music/minimax_music3/README.md)
- **Upstream URLs**:
  - Reference: [`mlx_audio/music/models/minimax_music3/`](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/music/models/minimax_music3)
  - Hugging Face Repositories: `MiniMaxAI/MiniMax-Music3`, `mlx-community/MiniMax-Music3-8bit`, `mlx-community/MiniMax-Music3-4bit`
  - Upstream Attribution: [mikolaj92/minimax-music3-mlx](https://github.com/mikolaj92/minimax-music3-mlx) (Apache-2.0)
- **Architecture**:
  - **Autoregressive (AR) Backbone**: Qwen3-based language model with time-major KV cache generating hierarchical semantic tokens.
  - **Residual Depth Decoder (RVQ)**: Expands semantic tokens into multi-codebook latent acoustic frames.
  - **Condition Encoder**: Fuses text prompts and timing tokens into DiT condition vectors.
  - **Diffusion Transformer (DiT)**: Continuous-time flow-matching diffusion network operating on latent chunks with Euler ODE stepping.
  - **Vocoder**: DAC-style multi-scale convolutional vocoder generating 44.1 kHz stereo audio.
- **Benchmarks & Timing**:
  - Apple M4 Max (CPU f32 software path):
    - Tiny AR (9000 frames): 64.3 s
    - Tiny Flow per chunk: 169.1 ms
    - Full-size converted models use `runtime::Music3Runner` on macOS with packed resident Metal weights. Real-time performance remains unqualified.
  - Apple M4 Max, Metal, cached 4-bit profile, indicative (host not quiet, not a frozen row): 1 s of audio at 2 flow steps generates in about 12 s (118 s before the kernel work, byte-identical output); 8 s at 1 flow step takes 75 s (RTF 9.4) with peak footprint 11.7 GiB, which stays at 11.8 GiB through 16 s. The AR stage runs at about 0.3 s per frame, and a default 30-step DiT chunk projects to about 250 s, so the default request is far from real time. Method, instruments, and rejected experiments are in the [family README](../../crates/audio/src/music/minimax_music3/README.md#metal-performance).
- **Numeric Contracts**:
  - Bit-exact RNG reproduction of MLX keys, splits, and uniforms.
  - Audio output: 44.1 kHz, stereo (2 channels).
