# Embedding model inventory

The family list comes from the mlxrs `embeddings` module
([findit-studio/mlxrs](https://github.com/findit-studio/mlxrs/tree/main/mlxrs/src/embeddings)):
EmbeddingGemma, SigLIP2 NaFlex, and CLAP, plus the shared pooling,
normalization, similarity, and ColVision scoring helpers. The Snowflake
arctic-embed models (BERT, XLM-RoBERTa, GTE, and Nomic BERT encoders)
were added afterwards, starting from their mlx-community conversions. Each MLX
conversion found on Hugging Face is paired with the upstream checkpoint it
came from, and every repository is pinned to a full commit. The
machine-readable table is `src/pins.rs`, and this file must agree with it.

Verification status per checkpoint lives in each family README. A pin
here means "this is the exact artifact", not "this passed a gate".

Pins resolved 2026-10-13 from the Hugging Face API.

## EmbeddingGemma (`gemma3_text`, feature `embeddinggemma`)

| Repository | Revision | Weights | Upstream |
|---|---|---|---|
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m/tree/57c266a740f537b4dc058e1b0cda161fd15afa75) | `57c266a740f537b4dc058e1b0cda161fd15afa75` | f32, gated | - |
| [google/embeddinggemma-300m-qat-q4_0-unquantized](https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized/tree/7740c61c3ad98a9deb8423a20388ab6c3c5c45f2) | `7740c61c3ad98a9deb8423a20388ab6c3c5c45f2` | QAT, unquantized, gated | - |
| [google/embeddinggemma-300m-qat-q8_0-unquantized](https://huggingface.co/google/embeddinggemma-300m-qat-q8_0-unquantized/tree/7b5b24595322ab0ea4d08827066860a6df8cb0aa) | `7b5b24595322ab0ea4d08827066860a6df8cb0aa` | QAT, unquantized, gated | - |
| [mlx-community/embeddinggemma-300m-bf16](https://huggingface.co/mlx-community/embeddinggemma-300m-bf16/tree/2f420efb007317a1ef2aa1608f87ef24226e6807) | `2f420efb007317a1ef2aa1608f87ef24226e6807` | bf16 | google/embeddinggemma-300m |
| [mlx-community/embeddinggemma-300m-8bit](https://huggingface.co/mlx-community/embeddinggemma-300m-8bit/tree/c8316c9e35cf8830541b04d315398c67faa6e497) | `c8316c9e35cf8830541b04d315398c67faa6e497` | MLX affine 8-bit, g64 | google/embeddinggemma-300m-qat-q8_0-unquantized |
| [mlx-community/embeddinggemma-300m-6bit](https://huggingface.co/mlx-community/embeddinggemma-300m-6bit/tree/e03111914aba981cacdd4a4c3a920d5064d9e33a) | `e03111914aba981cacdd4a4c3a920d5064d9e33a` | MLX affine 6-bit, g64 | google/embeddinggemma-300m-qat-q8_0-unquantized |
| [mlx-community/embeddinggemma-300m-5bit](https://huggingface.co/mlx-community/embeddinggemma-300m-5bit/tree/717843cb884a1ee6ca617ee6b84ab26546106583) | `717843cb884a1ee6ca617ee6b84ab26546106583` | MLX affine 5-bit, g64 | google/embeddinggemma-300m-qat-q8_0-unquantized |
| [mlx-community/embeddinggemma-300m-4bit](https://huggingface.co/mlx-community/embeddinggemma-300m-4bit/tree/5d9ef074df3957afc5c77127f208fddbc3c54187) | `5d9ef074df3957afc5c77127f208fddbc3c54187` | MLX affine 4-bit, g64 | google/embeddinggemma-300m-qat-q4_0-unquantized |
| [mlx-community/embeddinggemma-300m-qat-q4_0-unquantized-bf16](https://huggingface.co/mlx-community/embeddinggemma-300m-qat-q4_0-unquantized-bf16/tree/41b4b86661761a34f2bc275661ba67319dd83409) | `41b4b86661761a34f2bc275661ba67319dd83409` | bf16 | google/embeddinggemma-300m-qat-q4_0-unquantized |
| [mlx-community/embeddinggemma-300m-qat-q8_0-unquantized-bf16](https://huggingface.co/mlx-community/embeddinggemma-300m-qat-q8_0-unquantized-bf16/tree/2aeb9554ff157247ed06a41e61e8339e7cae9c22) | `2aeb9554ff157247ed06a41e61e8339e7cae9c22` | bf16 | google/embeddinggemma-300m-qat-q8_0-unquantized |

The Upstream column is what each model card says the conversion came
from. The quantized MLX repos were made from the QAT checkpoints, not the
base model, so compare them against QAT goldens. Against base-model
goldens they score cosine 0.976 (8-bit) and 0.917 (4-bit), which measures
the QAT weight change plus quantization, not the port.

Against their QAT sources they score min cosine 0.99965 (8-bit) and 0.947 (4-bit). Stored
codes sit a mean 0.25 quantization steps from the QAT weights, against up
to 2.35 from the base model, which confirms the model cards. The "bf16"
MLX repo stores F16.

The 5-bit and 6-bit conversions are pinned but the loader refuses them
until they get the same golden and test as 4-bit and 8-bit.

The google repositories are gated: accept the Gemma license on Hugging Face
before downloading. The MLX conversions store the sentence-transformers
Dense head inside the main safetensors (`dense.0`, `dense.1`) instead of
`2_Dense/` and `3_Dense/`, and quantize `embed_tokens` as well as the
linears.

## SigLIP2 NaFlex (`siglip2`, feature `siglip2`)

| Repository | Revision | Weights |
|---|---|---|
| [google/siglip2-base-patch16-naflex](https://huggingface.co/google/siglip2-base-patch16-naflex/tree/b53b807d3a2d5e2b3911292f2d69e5341cdc064c) | `b53b807d3a2d5e2b3911292f2d69e5341cdc064c` | f32 |

No MLX conversion of a NaFlex checkpoint exists on Hugging Face (searched
`mlx-community` and the `mlx` library tag). mlx-embeddings loads the
upstream repository directly.

## CLAP HTSAT-unfused (`clap`, feature `clap`)

| Repository | Revision | Weights |
|---|---|---|
| [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused/tree/79b58ed25fc00386262a2bea4b19fd21dc4310a0) | `79b58ed25fc00386262a2bea4b19fd21dc4310a0` (`refs/pr/3`) | f32 safetensors |

Upstream `main` (`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`) ships only
`pytorch_model.bin`, a pickle. The pin is the safetensors conversion opened
by the Hugging Face SFconvertbot as PR 3. Every other file has the same
blob id as `main` (checked through the API), and the PR adds only
`model.safetensors`.

Whether those tensors equal the pickle's was not
checked, since that needs unpickling. All CLAP fixtures come from the
safetensors, so the port matches the pinned artifact either way. Download
with `revision="refs/pr/3"` or the commit above. No MLX conversion exists.

## Snowflake arctic-embed (features `bert`, `gte`, `nomic-bert`)

| Repository | Revision | Architecture | Weights | Upstream |
|---|---|---|---|---|
| [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0/tree/ac6544c8a46e00af67e330e85a9028c66b8cfd9a) | `ac6544c8a46e00af67e330e85a9028c66b8cfd9a` | `xlm-roberta` | f32 | - |
| [mlx-community/snowflake-arctic-embed-l-v2.0-bf16](https://huggingface.co/mlx-community/snowflake-arctic-embed-l-v2.0-bf16/tree/3da10e38e7875d6501adc14665c95b6708646121) | `3da10e38e7875d6501adc14665c95b6708646121` | `xlm-roberta` | bf16 | Snowflake/snowflake-arctic-embed-l-v2.0 |
| [mlx-community/snowflake-arctic-embed-l-v2.0-8bit](https://huggingface.co/mlx-community/snowflake-arctic-embed-l-v2.0-8bit/tree/98033f618b7503c73aa857bc0656f763a1587e73) | `98033f618b7503c73aa857bc0656f763a1587e73` | `xlm-roberta` | MLX affine 8-bit, g64 | Snowflake/snowflake-arctic-embed-l-v2.0 |
| [mlx-community/snowflake-arctic-embed-l-v2.0-6bit](https://huggingface.co/mlx-community/snowflake-arctic-embed-l-v2.0-6bit/tree/c8125c01b2142f0065f2494c123a7df46f9ef08e) | `c8125c01b2142f0065f2494c123a7df46f9ef08e` | `xlm-roberta` | MLX affine 6-bit, g64 | Snowflake/snowflake-arctic-embed-l-v2.0 |
| [mlx-community/snowflake-arctic-embed-l-v2.0-4bit](https://huggingface.co/mlx-community/snowflake-arctic-embed-l-v2.0-4bit/tree/430b52bb3dcb9ad2404b1ac2b60f2fdc00f26033) | `430b52bb3dcb9ad2404b1ac2b60f2fdc00f26033` | `xlm-roberta` | MLX affine 4-bit, g64 | Snowflake/snowflake-arctic-embed-l-v2.0 |
| [Snowflake/snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l/tree/d8fb21ca8d905d2832ee8b96c894d3298964346b) | `d8fb21ca8d905d2832ee8b96c894d3298964346b` | `bert` | f32 | - |
| [Snowflake/snowflake-arctic-embed-m-v1.5](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5/tree/e58a8f756156a1293d763f17e3aae643474e9b8a) | `e58a8f756156a1293d763f17e3aae643474e9b8a` | `bert` | f32 | - |
| [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m/tree/fc74610d18462d218e312aa986ec5c8a75a98152) | `fc74610d18462d218e312aa986ec5c8a75a98152` | `bert` | f32 | - |
| [Snowflake/snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s/tree/e596f507467533e48a2e17c007f0e1dacc837b33) | `e596f507467533e48a2e17c007f0e1dacc837b33` | `bert` | f32 | - |
| [Snowflake/snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/tree/d8c86521100d3556476a063fc2342036d45c106f) | `d8c86521100d3556476a063fc2342036d45c106f` | `bert` | f32 | - |
| [Snowflake/snowflake-arctic-embed-m-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/tree/95c2741480856aa9666782eb4afe11959938017f) | `95c2741480856aa9666782eb4afe11959938017f` | `gte` (repo code) | f32 | - |
| [Snowflake/snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long/tree/92d97331f1f4b6a366c1f161354b9f3390cc219f) | `92d97331f1f4b6a366c1f161354b9f3390cc219f` | `nomic_bert` (repo code) | f32 | - |

None of the Snowflake repositories are gated. Only l-v2.0 has MLX
conversions, and their model cards name it as the source. Their stored
codes sit a mean 0.25 quantization steps from the l-v2.0 weights, which
confirms it. They quantize the word, position, and token-type tables too,
and the "bf16" repo really is BF16.

The MLX repos ship `modules.json` without `1_Pooling/`. The loader takes
CLS pooling from l-v2.0 only when `config.json` and the prompt table match
it exactly, and refuses otherwise (`BertEmbedder::load_with_pooling` takes
an explicit config). Against upstream they score min cosine 0.99995
(bf16), 0.99948 (8-bit), 0.99139 (6-bit), and 0.90167 (4-bit), which is
the conversion's cost. Against their own dequantized weights the port
matches to 2.6e-7.

m-v2.0 and m-long define their architectures in Python files inside the
repository (`trust_remote_code`). Fixtures were generated by running those
files at the pinned commits after an audit, never code from another
repository. The Rust ports do not execute any of it.

Two reference quirks the ports do not copy. GTE refuses to load as
configured without xformers, so its goldens set
`use_memory_efficient_attention` and `unpad_inputs` to false, which leaves
the math unchanged (measured 0.0 difference). Nomic BERT caches its
NTK-scaled rotary table, so after one input over 2048 tokens the reference
reuses the scaled table for shorter inputs (cosine 0.99775 on a short text
afterwards). The port is stateless and matches a freshly loaded model.

## MLX conversions found but not ported

Different architectures from the three mlxrs families. Recorded so a
later port starts from a pinned pair rather than a search.

| MLX repository | Revision | Upstream | Why not ported |
|---|---|---|---|
| mlx-community/embeddinggemma-2-{bf16,8bit,6bit,5bit,4bit,mxfp8,mxfp4,nvfp4} | bf16 `1a4ffddb7905d3f63486748deabe091a01fb6201`, 8bit `7505ef2f8ddef45efef6d060865f27989b3c9cec`, 4bit `8ab839e127cb8cb305edd71fb7a02c82f2d49ab3` | google/embeddinggemma-2 @ `914f7f89142e33e77833254d9c9b90c3cef7303b` | `EmbeddingGemma2Model` is a Gemma4 multimodal encoder (text, vision, and `gemma4_audio` towers), not the Gemma3 text encoder. mxfp/nvfp formats are not affine. The FluidInference Core ML export runs instead, outside this crate (`docs/COREML.md`). |
| mlx-community/siglip2-base-patch16-224-8bit | `5249fc157310584fe99dae6964707278eb6df50f` | google/siglip2-base-patch16-224 | Fixed-resolution SigLIP2 uses the `siglip` (v1) architecture with a fixed position grid, not NaFlex. |
| mlx-community/siglip2-so400m-patch16-384 | `47bb082a6eeb9789923d547a845ca2c55ae39245` | google/siglip2-so400m-patch16-384 | Same as above, so400m width. |
