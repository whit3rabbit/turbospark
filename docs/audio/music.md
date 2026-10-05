# Music Generation Models

Music generation models in TurboSpark produce polyphonic musical compositions and songs from text prompts, tags, and lyrics.

## Implemented Model Families

### MiniMax Music 0.5 (`minimax_music3`)

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
    - Full-scale official model (~11B parameters) requires Metal GPU offload for real-time streaming.
- **Numeric Contracts**:
  - Bit-exact RNG reproduction of MLX keys, splits, and uniforms.
  - Audio output: 44.1 kHz, stereo (2 channels).
