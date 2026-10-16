# `fish_drama3`: pinned facts and bring-up gates

The evidence record for Fish Audio Drama 3 as an audio family. The text
backbone runs on the existing `QwenGdnMoe` flow with streamed routed experts.
The audio tower and codec are new. Every number below comes from a named
source, not from a remembered formula.

Built with Fish Audio.

## Sources

1. **The checkpoint**, `mlx-community/fish-audio-drama-3-4bit` at revision
   `cc639e27aba7edb73429d3e3806f471dcd79d316`. Safetensors headers were
   range-read on 2026-10-14. `codec.safetensors` was downloaded and matched
   its published sha256.
2. **The reference implementation**, mlx-audio `mlx_audio/tts/models/fish_drama3/`
   at merge commit `dad2087f790dfe96b07d5aac1405db4b578b411f`
   (Blaizzy/mlx-audio PR 1018). This is newer than the crate's normative
   mlx-audio commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`, which has no
   Drama 3. The codec reference stays at `e1b19b9`.
3. **The upstream card**, `fishaudio/drama-3`: 35B parameters, 3B active,
   block-FP8 routed experts, 10 codebooks decoded causally.

### Pinned files

| File | Bytes | sha256 |
| --- | ---: | --- |
| `model-00001.safetensors` | 4,354,661,374 | `de6eeebdd8c09316acd4f5fdd05482221e8596a19a6e8384d7fd129d70e894bf` |
| `model-00002.safetensors` | 4,360,386,773 | `cdd874495884ba7a1f9ae5b2baf7fc8fed85f73799556e87a6406184b4f8cf9f` |
| `model-00003.safetensors` | 4,304,379,853 | `eefa12cee1d032b511869de2a71666b4f07c82ca2fd8c94c3c5bcd1da431ef11` |
| `model-00004.safetensors` | 4,407,528,323 | `cd4960872fc5ceff7c719bd5be9d9f770f3357068eca83af065db6ba68927a8a` |
| `model-00005.safetensors` | 4,425,623,733 | `e0dbbeb7f66a2168fd744df15987b145780213d65cb6b3579a2531a78e92774e` |
| `model-00006.safetensors` | 335,374,378 | `0b3e0b1b620080e4f4db53cc3b9c697c4179964be0e2172b339b605459470a7e` |
| `codec.safetensors` | 1,572,080,341 | `6ff9d6f07920b2155e3a48ada21956004b96df9114d255ed9f0788484ba73040` |
| `tokenizer.json` | 20,791,070 | `0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6` |

The small files (`config.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `chat_template.jinja`, `LICENSE`, `NOTICE`) are
plain git objects. Their sha256 is computed at catalog pin time.

## License

Fish Audio Community License 1.0 (`LICENSE` in the repo):

- Free for research and non-commercial use.
- Commercial use only while annual revenue is under USD 500,000.
- No Hosted Service (an API or app whose main purpose is TTS for third
  parties) without a separate license.
- A distributed copy carries the license and the `NOTICE` file.
- A product that uses it shows "Built with Fish Audio".

The catalog entry installs only behind an acknowledgment of these terms. The
upstream `fishaudio/drama-3` card names a different license
(`fish-audio-research-license`). This port pins the MLX conversion and its
Community License.

## Architecture

Per decoded frame (46.4 ms of 44.1 kHz audio):

1. The backbone's final RMSNorm output `h` goes through `semantic_head`
   (4097 rows: semantic tokens 248099..252194, then EOS 248046). Sampling
   picks the frame's main token. EOS ends the chunk.
2. The tower samples codebooks 1..9 from `h`, one block and one head per
   codebook, each block holding a KV cache across frames.
3. The next backbone input is `(embed[main] + codebook_emb[c1 + 4096]) *
   2^-0.5` for an audio frame (`vq_backbone_codebooks = 1`). Text rows use
   `embed[token]` alone.
4. Codes `c0..c9` go to the Fish S1 DAC, which emits 2048 samples per frame.

| Part | Facts |
| --- | --- |
| Backbone | `text_config.model_type: fish_omni_text`. 40 layers, `layer_types` three `linear_attention` then one `full_attention`, hidden 2048, 16 q heads and 2 kv heads at head_dim 256, attention output gate, qk norm, partial rotary 0.25 (NeoX, not interleaved), rope theta 1e7. GDN: conv kernel 4, 16 key heads and 32 value heads at 128. MoE in every layer: 256 experts, top 8, moe_intermediate 512, shared expert 512 with a sigmoid gate, `norm_topk_prob`. `rms_norm_eps` 1e-6. Zero-centered norm gammas. Vocab 252,288. This is the `QwenGdnMoe` baseline field for field. |
| Tower | `fish_qwen3_audio_decoder`, `decoder_type: codebook_mtp_shared`. 4 enrichment layers plus 9 residual layers. Each: embedding, `enorm`/`hnorm`, `eh_proj` (4096 to 2048), one block (dim 2048, 16 q and 2 kv heads at 256, gated attention with `wqkv` rows `[Q | gates | K | V]`, interleaved rope on a quarter of each head, theta 1e7, SwiGLU 6144), then `final_layernorm`. 9 codebook heads of 4096. One shared codebook table `[40960, 2048]`. All blocks run at the same temporal position. |
| Codec | Fish S1 DAC `build_ae()`: 44.1 kHz, hop 512, 2048 samples per frame, one semantic book of 4096 and nine acoustic books of 1024. |

### Prompt (`speech_template_version: reference-bracket-v1`)

Rows are `[token, c0..c9]`. Text rows carry zeros in the code columns. Audio
rows are `[c0 + 248099, c0..c9]`. Each string segment is tokenized on its
own, because tokenizing the joined prompt changes ids at the seams.

```text
<|im_start|>system\n
convert the provided text to speech reference to the following:
\n\n[Reference 1]\nText: <|speaker:0|> <transcript>\nSpeech: <codes>
<|im_end|>\n                          <- end of the cacheable system block
<|im_start|>user\n<|speaker:0|> <history text><|im_end|>\n
<|im_start|>assistant\n<|voice|><history codes><|im_end|>\n
<|im_start|>user\n<|speaker:0|> <target text><|im_end|>\n
<|im_start|>assistant\n<|voice|>      <- generation starts here
```

Without a reference the system line is `convert the provided text to speech`.
Long text splits into chunks of about 300 UTF-8 bytes with at most three
speakers each. Every chunk sees the first and the previous chunk as history.

### Sampling recipe

Temperature 0.7, top-p 0.7, top-k 30, at most 1024 frames per chunk. The
temperature applies before top-p. The cumulative mass runs over the full
vocabulary. The token that crosses top-p is dropped unless it ranks first.
Repetition-aware sampling: when a semantic main token repeats one of the last
16 main tokens, it is replaced by a second draw at temperature 1.0, top-k 30,
top-p 0.95.

**Text-scaled frame cap (not in the reference).** A chunk also stops at
`64 + 24 * tokens + 22 * speaker_tags + 108 * cues` frames when that is under
1024 (`TextCap::MEASURED` in `generate.rs`). Tokens are the backbone
tokenizer's, counted over the chunk's words with special tokens, `[cues]` and
paired-tag markup removed (`chunking::spoken_text`). A paired tag counts as a
cue. Below about 40 tokens the scaled cap is the tighter one, and longer
chunks keep the 1024 bound.

`ChunkEnd` reports `TextCap` apart from `Eos` and `MaxFrames`, and
`FishDrama3Runner` logs every chunk that ends without EOS. `text_cap: None`
(`text-cap=off` in `fish_drama3_say`) restores the reference's behavior.

Measured 2026-10-15 with the cap off: `fish_drama3_say` on the pinned 4-bit
install (Metal trunk, tower and codec, 16 streamed slots) ran the
quant-evaluation sentences (20 English, seeds 0 and 1) plus 8 extras at
seeds 0 and 1: two Chinese, one Japanese, one German, a three-turn dialogue,
`[laughs] ... [pause]`, numbers, and a 290-byte chunk. That gave 56 clips,
pooled with the 80 mlx-audio clips of the 3-bit comparison (same recipe and
seeds, 4-bit and 3-bit).

The host was on AC power but not quiet (1-minute load 9 to 14 from other
sessions' builds). A frame count is fixed by the seed, so the load affects
timing only, and no timing is quoted here. A failed clip is one whose
Parakeet transcript misses most of the text. In this port, every failed clip
also ends in 458 or more silent frames (RMS under -45 dBFS), and every other
clip in at most 8.

| Clips | Count | Frames per token |
| --- | ---: | --- |
| Normal, no cue or tag | 121 | median 6.75, 90th percentile 8.74, slowest 13.6 (German, 163 frames on 12 tokens) |
| Normal, `[sighs] Fine. If we miss it, it's on you.` | 4 | 101 to 124 frames on 12 tokens, so the sigh adds at most about 50 |
| Failed, this port | 6 of 56 | s18 seed 1 and both dialogue and both cue clips ran to 1024. s08 seed 0 ended by EOS at 495 after 37 voiced frames |
| Failed, mlx-audio | 5 of 80 | 1024, 1024, 854, 566 (`[sighs]`), 244 (`[sighs]`, no words) |

Every failure in this port speaks part of the text ("Nobody expected the old
radio to work.", or the first dialogue turn) and then emits silence. The
dialogue and cue texts failed at both seeds. The mlx-audio checkpoints are no
longer on the evaluation drive, so the reference was not run on those two
texts. The cause is narrowed below (dialogue and cue failures).

The constants keep every normal clip under half its cap. The closest are the
290-byte chunk (498 frames against the 1024 bound) and the German clip (163
against 352). The cap would stop 10 of the 11 failed clips 47 to 672 frames
(2 to 31 s) sooner. The eleventh (244 frames of sighing) is shorter than any
cap that leaves room for a sigh.

Speaker tags had no normal measurement when the constants were set, because
both dialogue clips failed. Their 22 frames (1 s) is a turn gap with margin.
Three later dialogue seeds completed (below). Their gaps ran 15 to 78 frames,
longer than 22, but the per-token margin covers that: the slowest used 240 of
its 658 frames. The cap only shortens the
silence after a failure. It does not bring back the words that were never
spoken.

With the cap on, the same install and seeds:

| Clip | Cap off | Cap on |
| --- | --- | --- |
| s07 seed 0 (21 tokens, cap 568) | 201 frames, EOS | 201 frames, EOS, byte-identical WAV |
| s16 seed 0 (`[sighs]`, cap 460) | 118 frames, EOS | 118 frames, EOS, byte-identical WAV |
| s18 seed 1 (22 tokens, cap 592) | 1024, max frames | 592, `TextCap`, the same 55 voiced frames |
| Dialogue seed 0 (cap 658) | 1024, max frames | 658, `TextCap` |

**Dialogue and cue failures (2026-10-16).** The prompt is ruled out. For
both texts the port builds the same rows as mlx-audio: `prompt_real.json`
cases 4 and 5 are the reference's `split_text` then `build_frames` on the
real tokenizer, and `real_tokenizer_frames_match_the_reference` passes on
them. They also pin a reference quirk the port shares. A short turn body is
never stripped, so every tag is followed by two spaces (token 220, then
` Are`) and turns are joined by ` \n` (token 695).

Ten seeds per text, same install and recipe, cap off:

| Text | Complete | Failed |
| --- | --- | --- |
| Three-turn dialogue (cap 658) | 3 of 10, EOS at 108, 117 and 240 | 7: six run to 1024 after the first turn, one ends by EOS at 587 after two turns and 465 silent frames |
| `[laughs] ... [pause] Seriously.` (cap 664) | 2 of 10, EOS at 171 and 201 | 8: the first sentence, then silence where `[pause] Seriously.` belongs (one EOS at 843, seven at 1024) |

Every failure stops at a gap the text asks for: the turn change or the
`[pause]`. Completed dialogue clips leave 15 to 78 silent frames between
turns and then speak again. Failed ones never come back from the first gap.

The reference runs on these texts are still owed. The re-downloaded mlx-audio
checkpoint (cc639e27, all 16 files verified) holds 24 GB of GPU buffers.
Beside the host's other work it swapped and ran under one frame per minute,
so the runs were stopped. Until they run, only the prompt is cleared. The
rest of the loop (trunk numerics, tower, sampling) has no teacher-forced
parity against mlx-audio yet (see Gates), so the 70 to 80% failure rate is
not yet attributed to the port or to the model.

## Tensor inventory (from the headers)

1,841 tensors in six shards. Names: `backbone.model.{embed_tokens,layers.N,norm}`,
`semantic_head`, `tower.*`. No `lm_head`. No MTP tensors.

| Group | Storage |
| --- | --- |
| Routed experts `mlp.switch_mlp.{gate,up,down}_proj` | U32 4-bit, group 64, stacked `[256, out, packed_in]`, BF16 scales and biases |
| Attention, linear-attention and shared-expert projections | 8-bit, group 64 |
| `embed_tokens`, router `mlp.gate`, `shared_expert_gate`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, `conv1d`, `linear_attn.norm` | BF16, unquantized |
| Layer norms, q/k norms, final norm, tower norms | F32, zero-centered gammas stored as `1 + w` |
| Tower projections and codebook heads, `semantic_head` | 8-bit, group 64 |
| Tower embeddings and `codebook_embeddings` | BF16 |

The codec is raw PyTorch layout: conv weight norm under
`P.conv.parametrizations.weight.original0/1`, bias at `P.conv.bias`, the
quantizer projections as `weight_g/weight_v`, plus `freqs_cis` tables. The
Rust loader maps these exactly as the reference `DAC.sanitize` does
(`crates/audio/src/codec/fish_s1_dac/mod.rs`, `wn_names`).

## Memory accounting, from the headers

| Group | Size |
| --- | ---: |
| Routed experts (40 layers x 256) | 17,280.0 MiB |
| One expert (gate, up, down with scales and biases) | 1.6875 MiB |
| Trunk dense projections and norms | 1,473.9 MiB |
| `embed_tokens` (BF16, row-gathered) | 985.5 MiB |
| Tower projections and heads | 1,043.9 MiB |
| Tower embeddings (BF16, row-gathered) | 368.0 MiB |
| `semantic_head` | 8.5 MiB |
| Codec (F32) | 1,499.2 MiB |

Expert slot cache across 40 layers: 540 MiB at 8 slots, 1,080 MiB at 16,
2,160 MiB at 32, 4,320 MiB at 64. The streamed working set at 16 slots is
about 1.5 GiB of trunk projections, 1.05 GiB of slots, 1.0 GiB of tower and
1.5 GiB of codec, plus KV, GDN state and snapshots (about 61 MiB of GDN state
each). That is a computed estimate, not a measured peak. mlx-audio peaks at
about 26 GB with every expert resident.

## Compute placement

| Part | Placement | Reason |
| --- | --- | --- |
| Input row | Host gather | One table row plus one codebook row. No compute to move. |
| Backbone | Metal, streamed experts | Top-8 of 256 routing per token and expert streaming do not fit Core ML's static, resident graphs. |
| Tower | Metal | Single-token decode reads about 1 GiB of 8-bit weights per frame, so it is bandwidth-bound. Nine host sample points per frame. |
| Codec | Metal through the `ComputeBackend` conv path (the WavVAE `ConvExec` pattern) | The CPU path is slower than real time (measured below). |

No Core ML or native MLX export of the Fish S1 DAC exists. A Neural Engine
codec stays an experiment, taken up only if the Metal codec measurably
contends with the backbone.

The next backbone step needs only `main` and `c1`. So backbone step n+1 can
start while the tower samples `c2..c9` of frame n (8 of its 13 blocks). The
reference does not overlap these.

## Measured at Phase 0

**CPU codec, portable f32** (`crates/audio/examples/fish_s1_dac_timing.rs`),
M4 Max, AC power, 2026-10-14. `quiet_host` false: the 1-minute load was
44.0 before and 42.8 after, while another session was building. This is a
contaminated reading, recorded as an upper bound, not a benchmark row.

| Direction | Audio | Wall | Real-time factor |
| --- | ---: | ---: | ---: |
| Encode | 9.98 s | 10.02 s | 0.997x |
| Decode | 9.98 s | 14.37 s | 0.695x (66.8 ms per frame) |

The decode needs 46.4 ms per frame to keep up. The CPU path does not, and it
would share cores with the backbone's expert reads.

**First real generation, 2026-10-14.** The pinned checkpoint, streamed into a
local install by `crates/repack/examples/fish_drama3_intake.rs` (22 GB, 1,624 s,
the trunk's norm-shift check reading 0.984 on the real gammas), then
`crates/runtime/examples/fish_drama3_say.rs` on the model card's sentence
`[sighs] Fine. If we miss it, it's on you.` with 16 streamed expert slots and
seed 0. The trunk is the Metal `HiddenTrunk`. The tower and the codec are the
portable f32 paths (the tower's large matvecs on scoped threads, the codec
streamed in 7-frame pieces). M4 Max, AC power, 1-minute load 3.4 at start.

| Measure | Value |
| --- | ---: |
| Frames | 97, ended by EOS (4.50 s of audio) |
| Parakeet Redux transcript | "Fine. If we miss it, it's on you." (exact; the cue is not spoken) |
| Peak `phys_footprint` | 5.17 GB |
| Backbone prefill | 107 ms per row (38 rows, no reference) |
| Backbone decode | 81.7 ms per frame |
| Tower and sampling (CPU) | 131.6 ms per frame |
| Codec, streamed (CPU) | 150.7 ms per frame |

The footprint lands in the estimate above and is a fifth of mlx-audio's 26 GB.
Speed is not real time yet: the tower and codec need their Metal paths, and the
backbone runs at about 2.5 times the frozen text decode cost.

Later runs were contaminated (load 10 to 41 from a parallel Swift build) and read 76 to 92 ms
per frame for the backbone. 32 slots read the same as 16 (77.6 ms), which
points at page-cache misses rather than slot misses: the 17 GB of expert files
do not stay cached beside other work on this 36 GB host (0.4 GB free, 13 GB of
file-backed pages in total during the sweep). Mapped residency on a cold page
cache ran over 8 minutes on the same sentence and was stopped. A clean warm
measurement is still owed.

## Phase 4: device paths

**Metal tower** (`crates/runtime/src/fish_drama3_tower.rs`). The tower runs
on the qwen36 flow's own kernels: the INT8 affine GEMV on the checkpoint's
8-bit rows, the centered RMSNorm kernels (each F32 gamma stored as BF16
`gamma - 1`), split-KV decode attention over a KV pair per block, the sigmoid
gate, SiLU-mul and residual add.

- **Rope by permutation.** The tower rotates interleaved pairs over the first
  quarter of each head. The flow's `rope_neox_subdim` rotates pairs
  `(i, i + rotary / 2)` at the same frequencies. Permuting each query and key
  head's first 64 rows (and the q/k norm gammas) to evens then odds turns one
  into the other. Scores are dot products over identically permuted q and k,
  so they are unchanged, and gates and values stay as stored.
- **Layout.** `wqkv` is split into three row ranges, so K and V land straight
  in this position's cache row. One walk over the weights only measures. A
  second writes each piece straight into the Metal buffer, so the host never
  holds more than one matrix.
- **Passes.** A decode frame is ten command buffers: the semantic head, the
  enrichment layers with residual layer 0, then one per remaining residual
  layer. Each ends in a host sample.

Gates, on the tiny fixture against the pinned mlx-audio goldens
(`crates/runtime/tests/fish_drama3_tower.rs`):

| Check | Result |
| --- | ---: |
| Head logits vs the bf16 reference | 1.5e-2 worst relative (bound 3e-2) |
| Head and semantic logits vs the f32 CPU tower | 2.5e-3 worst relative (bound 1e-2) |
| Greedy codes vs reference and CPU | identical |
| Split prefill and truncate replay | bit for bit |

Mutations caught: no rope permutation (2.0e-1), gates permuted with the
queries, q_norm left unpermuted (4.7e-2).

**Metal codec** (`audio::codec::fish_s1_dac::DacExec`,
`crates/runtime/src/fish_drama3_codec.rs`). Convolutions, transposed
convolutions, the ConvNeXt pointwise linears and Snake run on
`gpu::Music3Device` in F32, the route MegaTTS3's WavVAE decoder takes. Code
lookups, the transformers, norms and GELU stay on the CPU.

The exec lives apart from `FishS1Dac`, so the codec stays `Send` and `Sync`
for Echo TTS and Fish Omni, which share it. The device is tied to one thread.
So `CodecWorker` owns the codec and its uploaded weights for the runner's
lifetime, and it decodes on its own thread while generation runs.

On the tiny codec the Metal decode is 2.5e-7 from the host, and encoded codes
are identical. On the real codec the worker's streamed decode is 1.0e-5 from
the host's whole decode. A fake device running the host kernels checks the
routing bit for bit: every conv kind, the linears and Snake are sent out, and
each weight is uploaded once.

**Exact streaming.** The first streamed decoder re-decoded 16 frames of
context before every 7-frame piece. That is 3.3 times the work of the frames
themselves.

Every conv in the decode path is causal. So each stride-1 conv now carries its last `kernel_eff - 1` input columns between pieces, zeros at
the start like the batch path's causal padding. Each transposed conv carries
the `ceil(kernel / stride) - 1` input columns its next outputs still read. A
piece costs only its own frames. The stream matches the whole decode bit for
bit on the host at every chunking tried.

**Chunked prefill with per-row hidden** (`HiddenTrunk::prefill`). The qwen
MoE chunked driver takes host embedding rows (written over their `scratch.x`
rows before the first commit, as the sequential `host_embed` is) and, in
place of the head, runs the sequential flow's final norm once per row into
`moe_x`, which is micro-batch wide and idle once the last routed pass
retires. One router readback per layer per 16 rows instead of per row. On the
synthetic install, 37 mixed token and host rows (two micro-batch edges) give
hidden states byte-identical to stepping, and so does the step after.
`set_chunked_prefill(false)` keeps the per-row loop as the reference.

**Warm runs, 2026-10-14.** These ran `fish_drama3_say` with the Metal tower
and the Metal codec, seed 0, on AC power. The sentence is 34 words, which
decode to 201 frames (9.33 s of audio) ending in EOS. The expert files were
read first (17 GB in 3 s, already in the page cache).

The script waited for a 1-minute load under 3, and the runs started at 4.0
to 4.6, so this is not a quiet host. Each row is one run after a discarded
warmup. The mapped run collapsed (below) and a disk scanner then started,
so the second interleaved round and the runner example were not run. The runner
example puts the codec beside generation.

| Expert slots, residency | Prefill per row (44 rows) | Backbone per frame | Tower and sampling | Codec, streamed | First frame | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16, streamed | 48.8 ms | 48.4 ms | 9.9 ms | 26.3 ms | 2.63 s | 5.57 GB |
| 32, streamed | 55.6 ms | 49.9 ms | 9.9 ms | 27.2 ms | 2.92 s | 6.73 GB |
| 16, mapped, under memory pressure | 545 ms (24 rows) | 8.6 s | 242 ms | | | |
| 16, mapped, calm host (2026-10-15) | 478 ms (29 rows) | 29.5 ms | 25.1 ms | | | |

- Parakeet Redux transcribes the output word for word.
- Chunked prefill reads 48.8 ms per row against 91.7 to 107 ms per row when
  stepping in earlier runs. Those runs were not interleaved with this one.
- 32 slots are no faster than 16, as in the contaminated sweep, and cost
  1.2 GB more.
- Mapped residency is fast or 180 times slower depending on how much wired
  memory the rest of the machine leaves for Drama's table. The 22-minute
  run was in decode, not prefill (the example prints prefill only at the
  end), and ran beside parallel compiles at load 16 to 17. On a calm host
  the backbone decodes mapped at 29.5 ms per frame. Both mapped rows are
  single short runs. The fit guard now refuses mapped residency for Drama
  on this machine (below).
- The backbone and the tower together take 58 ms per frame against the
  46.4 ms budget, so generation runs at about 1.25 times real time before
  any overlap. The codec (26 ms per frame) runs on its own thread and is not
  on that path, but it shares the GPU. Its contention is not measured yet.
- The backbone reads 48 ms per frame against 25 to 31 ms per text token on
  the same flow. Why TTS frames cost more (expert reuse across frames, slot
  misses) is the next question. The other lever is starting backbone step
  n+1 once `c1` is sampled, which hides the tower's last 8 blocks.

**Mapped residency and memory pressure, 2026-10-15.**
`crates/bench/tests/mapped_residency_cycle_probe.rs` maps every layer file
of the Drama trunk as one no-copy buffer each. It times one INT4 GEMV per
submission while walking the first n layers in order, as a decode does.

| Held during the walk | Last fast n (GiB mapped) | First slow n (GiB mapped) | Slow pass |
| --- | ---: | ---: | ---: |
| nothing, calm host | 40 (16.88) | none | |
| nothing, parallel compiles at load 16 to 17 | 35 (14.77) | 40 (16.88) | 70 to 179 ms |
| 16 GiB of incompressible memory | 25 (10.55) | 30 (12.66) | 156 to 160 ms |

A fast pass is 0.11 to 0.15 ms. Mapping only part of each layer file keeps
all 40 buffers fast at 8.4 and 12.7 GiB, so the buffer count is not the
limit. 14 GiB of mostly-zero pages changed nothing, because the memory
compressor absorbs them. During a slow Drama run, total wired memory stayed
between 10.5 and 12.3 GB, so the mappings are not held wired in full once the
driver is under pressure. When they fit they are: on a calm host, wired
memory rose from 5.4 to 22.7 GiB within two seconds of the probe binding all
40 layers, and fell back to 5.5 GiB when it exited. A second run reproduced
it: 4.7 GiB at idle, 20.4 GiB as the walk reached 40 layers, 5.3 GiB after
exit.

The reading that fits all of it: the GPU wires every bound layer mapping in
full. Once other wired memory pushes the total near what the driver keeps,
it unwires a layer, and the next pass that binds it re-wires the whole
layer (about 2.6 GB/s). A decode walks every layer in order, so every pass
misses. A Gemma 4 table (12.3 GB) leaves more room on this host than
Drama's 16.9 GiB. `docs/EXPERT_RESIDENCY.md` has the full write-up. MiniCPM-V 4.7's trunk (16 GB of experts) shows the same collapse in
text decode: 0.155 tok/s mapped against 11.2 streamed, at load 6 to 7.

`auto` resolves to mapped only when the streamed cache does not fit, which
is exactly when memory is short, so on a small machine it would pick this
cliff. `model_io::mapped_residency_fits` now guards both the open and the
admission resolver. It budgets the resident weights plus the expert table,
and 10 GiB for the rest of the machine, against Metal's recommended working
set (28.08 GiB here). Drama 3 (19.26 GiB) and MiniCPM-V 4.7 (20.48 GiB) do
not fit. Mapped asked for by name is refused with the reason, and `auto`
resolves down to streamed. That gives up the calm-host 29.5 ms row above to
rule out the 180 times collapse under ordinary load
(`docs/EXPERT_RESIDENCY.md`). Decided 2026-10-15: past the budget, mapped
asked for by name stays refused rather than allowed with a warning.

**Codec profile.** A `sample` of the real codec's 60-frame whole decode on
Metal puts 84% of the decode thread in the decoder stack. Most of that waits
on the F32 tiled conv kernel (about 530 GFLOPS effective). A host scan of every
device output for non-finite values (`gpu::music3::finite_output`) is
another 20%, and the post transformer on the CPU 15%. These are the codec's
remaining levers.

## Speed work, 2026-10-16

All Drama tests pass on `main` at d89053b8, rebuilt in an export with its own
target directory. That includes the qwen MoE batched prefill another session
landed, which the hidden-row prefill shares.

**Why a frame costs more than a text token.**
`crates/runtime/examples/fish_drama3_step_cost.rs` steps one trunk two ways
and diffs its phase counters per step. Text means a paragraph's own tokens, and
frames means a real generation's decode rows. Same trunk, slots and code path,
on AC power at load 5, with expert reads counted on the SSD.

| Slots | Text: hit, ms per step | Frames: hit, ms per step | Frames: expert I/O, share from SSD |
| ---: | ---: | ---: | ---: |
| 16 | 63%, 32.8 | 34%, 39.9 | 20.3 ms, 17% |
| 32 | 77%, 31.2 | 39%, 40.5 | 20.4 ms, 18% |
| 64 | 89%, 32.8 | 53%, 40.7 | 19.9 ms, 24% |

- Audio frames reuse experts about half as often as text, so the gap is
  expert I/O. GPU work per step is about the same in both.
- More slots raise the hit rate but not the speed. The bigger slot cache pins
  memory, and more of the remaining reads come off the SSD.
- Host state outweighs all of this. The same binary read 89 to 91 ms per
  frame step an hour earlier, during another session's work. The 48 ms
  backbone row from 2026-10-14 was taken under load too.

**Overlapping the tower tail** (`GenerateOptions::overlap_tower_tail`, on by
default). The next backbone row needs only `main` and `c1`, so `TowerModel`
splits a frame into `begin_residual_codes` (through head 0) and
`finish_residual_codes`. The backbone step runs while the tower finishes on a
second thread. The draws keep their order, so the codes match the serial loop.

| Per frame, 16 slots, `fish_drama3_say`, interleaved | Serial | Overlapped |
| --- | ---: | ---: |
| Backbone | 32.3 to 35.1 ms | 32.3 to 33.4 ms |
| Tower and sampling not hidden | 8.8 to 8.9 ms | 3.9 to 4.0 ms |
| Prefill per row (44 rows, warm) | 34.9 ms | 34.1 to 34.7 ms |

Serial and overlapped runs write byte-identical WAVs.
`overlap_and_serial_frames_agree` holds that, and three mutations fail it or
the tower parity tests: a wrong code in the overlapped row, and a wrong code
index in either tower's finish.

**End to end.** `FishDrama3Runner` (codec decoding at the same time) on the
same sentence, warm. Real-time factor 1.14 to 1.16, first samples after 1.9
to 2.1 s. After the first samples it produced 9.33 s of audio in 8.8 s, so
streamed playback keeps up once it starts. The codec costs about 7 ms per
frame of GPU contention. Parakeet transcribes the output word for word.

**Where a frame's expert I/O goes.** `fish_drama3_step_cost` at 16 slots with
`TURBOSPARK_EXPERT_DISK_IO=1`, two rounds, expert files read first. AC power,
1-minute load 3 to 5, so the host was not calm.

| Per step | Text | Frames |
| --- | ---: | ---: |
| Wall | 33.2 to 35.1 ms | 41.7 to 42.2 ms |
| Expert I/O | 10.7 to 12.2 ms | 18.0 to 18.1 ms |
| Expert bytes read | 200 to 204 MiB | 348 to 355 MiB |
| Of those, from the SSD | 9.5 to 13.4% | 12.1 to 13.2% |

- A frame reads 1.7 times the expert bytes of a text token and spends about
  1.6 times as long reading them. The whole-step rate is the same for both,
  16 to 19 GiB/s.
- The SSD share costs more than its size. Comparing runs gives 0.16 to 0.20 ms
  per MiB off the SSD (5 to 6 GiB/s) against 0.03 to 0.04 ms per MiB from the
  page cache. At 12% that is about 8 of the 18 ms. This is an estimate from
  two run pairs, not a gate.
- The page cache cannot hold the table. A one-off `mincore` probe right after
  a run found 9.4 of 16.9 GiB of expert files resident, from 77% of layer 0
  down to 28% of layer 36. Reads skew toward the resident experts, so only 12%
  of the bytes come from the SSD.
- The trunk weights give nothing back. `model_weights.bin` is the Metal buffer
  itself (one no-copy mapping), not a second copy in the page cache.

**Cache policy cannot fix the miss rate.** A `TURBOSPARK_ROUTER_TRACE=1`
capture of one seed-0 generation (52 prompt rows, 224 frames), replayed with
`scripts/expert_policy_replay.py --skip 52` and `scripts/pilot_ceiling.py`.
The replay is deterministic, so host load does not touch it. Hit rate:

| Slots per layer | Engine LFU | LRU | Decayed LFU (h=64) | Belady optimum |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 36.2% | 37.3% | 37.4% | 51.9% |
| 32 | 45.9% | 48.2% | 48.1% | 67.2% |
| 64 | 60.7% | 63.7% | 63.3% | 80.9% |
| 128 | 80.9% | 82.3% | 82.2% | 91.1% |

- Online policies gain 1 to 3 points. The optimum gains 10 to 21, but it needs
  to know the future routes.
- At 16 slots 90% of the misses are evictions, not first touches. The previous
  token's routing names 29% of a layer's experts and none of the misses.
- Frame routing is diffuse. Half the table per layer (128 slots) still hits
  only 81%.

**What is left.**

- More slots do not help (table above). Mapped residency is refused past the
  wired budget.
- Router lookahead prefetch is the one open lever. `docs/EXPERT_ROUTING.md`
  closed it on Gemma at an 85% hit rate, and named what would reopen it: a
  high miss rate and a disk-bound read. Frames have the first (64% misses at
  16 slots) and a small part of the second (12% of the bytes). The read is
  still mostly a memcpy that competes with the GPU for bandwidth.
- Nobody has measured a lookahead predictor's recall on this trunk. The probe
  (`TURBOSPARK_PILOT_PROBE`) is wired only into Gemma. The next step is that
  probe for `QwenGdnMoe`, before any prefetcher.
- Free memory sets the SSD share. That is read from the residency probe and
  was not varied here, so these timings should move with the host.

## Refusals and descope for v1

- Mapped expert residency where the fit guard rejects it, which includes
  Drama 3 on a 36 GiB machine. Asked for by name it is refused, and `auto`
  resolves to streamed (decided 2026-10-15, above).
- No `speed` control and no `instruct` text. Delivery cues go inline, such as
  `[whispers]`.
- One reference voice per request. Multiple references, one per speaker tag,
  come later.
- Caller text must not contain structural tokens (`<|im_start|>`,
  `<|im_end|>`, `<|voice|>`, `<|text|>`, `<|interleave|>`, `<|semantic:`).
  Speaker tags run `<|speaker:0|>` to `<|speaker:15|>`.
- Voice cloning requires `consent: true` at every entry point.
- Only the 4-bit expert, 8-bit dense conversion. The 3-bit and bf16
  conversions are refused until each has its own gates. Decided 2026-10-15
  to stay on 4-bit and build no 3-bit support, after an mlx-audio comparison
  of `mlx-community/fish-audio-drama-3-3bit` (7120e8b1, 3-bit routed experts
  only) against the pinned 4-bit (20 sentences, 2 seeds, 40 clips each):
  - On clips that complete, Parakeet WER is 1.46% (3-bit) against 1.43%,
    with the same errors in both.
  - Catastrophic clips: 3 of 40 against 2 of 40, not separable at this size.
    One clip per quant ran to the 1,024-frame cap. The others are a
    near-runaway and a `[sighs]` cue that turned into long sighing.
  - Teacher-forced top-1 agreement is 77 to 80% at a KL of about 0.07 nats
    per step. 3-bit's own samples are less likely under both models, and it
    made 4.5% more audio.
  - There is no higher-precision reference: the 8-bit checkpoint does not
    run on a 36 GB machine.
- No prebuilt TurboSpark install on Hugging Face yet. Installs convert locally
  from the pinned `mlx-community` revision. The repacked install is a
  Derivative Work under the Community License, so a published copy needs the
  `LICENSE`, a `NOTICE` that says what changed and how, and "Built with Fish
  Audio" on its card. Publishing waits until the Phase 6 gates pass and the
  install format is settled, since only TurboSpark can load it and any format
  change would mean republishing.
- The intake refuses `rms_norm_eps` other than 1e-6 and a layer pattern other
  than three linear layers per full one. It also refuses norm tensors whose
  mean is not near 1, since an unshifted zero-centered export would run wrong.

## Gates (to be frozen from the first measurement)

- Trunk: hidden states from a token id and from its embedding row are
  bit-identical. Chunked prefill with per-row hidden output equals the
  sequential path byte for byte.
- Teacher-forced parity against mlx-audio on the same bytes: per-row hidden
  cosine, semantic top-1 and codebook-head agreement. The reference reports
  0.995 hidden cosine and 86% semantic top-1 against PyTorch. Thresholds are
  set from our first measurement, not copied.
- Router INT8 versus BF16 top-8 agreement, and the effect of narrowing the
  F32 norms to BF16.
- End to end: STT word error rate on a fixed clip set and a listening check.
- Memory oracle at 16 slots on a 30 s generation, holding snapshots.
- Speed: ms per frame for backbone, tower and codec separately, at 8, 16, 32
  and 64 slots. Time to first audio with a cached voice. The gap between
  long-text chunks. Mapped residency stays out of these gates while the fit
  guard refuses it here.
