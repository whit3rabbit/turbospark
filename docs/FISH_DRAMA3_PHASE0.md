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

The reference recipe (mlx-audio, `SamplingParams::REFERENCE`) is temperature
0.7, top-p 0.7, top-k 30, at most 1024 frames per chunk. The temperature
applies before top-p. The cumulative mass runs over the full vocabulary. The
token that crosses top-p is dropped unless it ranks first. Repetition-aware
sampling: when a semantic main token repeats one of the last 16 main tokens,
it is replaced by a second draw at temperature 1.0, top-k 30, top-p 0.95.

**Product default since 2026-10-16: Fish Audio's serving recipe**
(`SamplingParams::FISH_SERVING`): temperature 1.0, top-p 0.9, top-k 30, the
same RAS and top-p rule. It is the default of Fish Audio's own runtime
(SGLang-Omni `fishaudio_drama3/speech_request.py` at 28c3e35a). The measured
reason is under "Pause trap and the recipe" below. Every measurement above
that heading ran the reference recipe.

**Text-scaled frame cap (not in the reference).** A chunk also stops at
`64 + 24 * tokens + 22 * speaker_tags + 108 * cues` frames when that is under
1024 (`TextCap::MEASURED` in `generate.rs`). Tokens are the backbone
tokenizer's, counted over the chunk's words with special tokens, `[cues]` and
paired-tag markup removed (`chunking::spoken_text`). A paired tag counts as a
cue. Below about 40 tokens the scaled cap is the tighter one, and longer
chunks keep the 1024 bound.

`ChunkEnd` reports `TextCap` apart from `Eos` and `MaxFrames`, with the
chunk's `limit`. `FishDrama3Runner` logs every chunk that ends without EOS
and returns a `Drama3Take` whose `stopped_early()` says the take likely
stopped speaking early (below). `text_cap: None` (`text-cap=off` in
`fish_drama3_say`) restores the reference's behavior.

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

**Pause trap and the recipe (2026-10-16).** A per-frame trace of the main
draw (`fish_drama3_say` with a throwaway probe) shows the mechanism on
`Welcome back. This is a short voiceover test with no reference voice.` In
the gap after a sentence the semantic head spreads its mass over a few
silence tokens (top-1 0.2 to 0.36) with EOS near 0, so the model is pausing,
not ending. Under top-p 0.7 the first draw keeps one to three of those
tokens and can never pick speech. The only way back is the RAS retry (T 1.0,
top-p 0.95), which fires on almost every silent frame. Seed 1 leaves a
13-frame gap through a retry. Seed 0 never does: 12 voiced frames, then 411
silent ones to the 424-frame cap. Fish Audio's own runtime samples the main
draw at T 1.0 and top-p 0.9 with the same RAS, so speech tokens survive the
first draw too.

Same install and probe, text cap on. A take fails when it ends without EOS or
holds hundreds of silent frames:

| Text | Reference recipe | Fish Audio's recipe |
| --- | --- | --- |
| The voiceover script, seeds 0 to 11 | 4 of 12 | 0 of 12 |
| Three-turn dialogue, seeds 0 to 5 | 4 of 6 | 1 of 6 |
| `[laughs] ... [pause] Seriously.`, seeds 0 to 5 | 4 of 6 | 3 of 6 |
| Total | 12 of 24 | 4 of 24 (Fisher exact, two-sided, about 0.03) |

An explicit `[pause]` still traps half the takes under either recipe, so the
recipe is not the whole fix there.

Corpus check through `FishDrama3Runner` (`fish_drama3_wer`,
`crates/audio/testdata/fish_drama3/eval_corpus.txt`: 20 English sentences,
the voiceover script, two dialogues, two cue lines, numbers, German and a
249-byte chunk; seeds 0 and 1; Parakeet TDT on the CPU). The Phase 0 corpus
was kept off the repository and is gone, so this one replaces it:

| Recipe, seed | Reported stopped early | Failed (harness rule) | Mean WER | WER of passing clips | Widest EOS chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference, 0 | 4 | 1 | 8.1% | 2.6% | 33% of its limit |
| Reference, 1 | 4 | 2 | 9.1% | 2.0% | 42% |
| Fish Audio, 0 | 1 | 0 | 1.9% | 1.6% | 39% |
| Fish Audio, 1 | 0 | 0 | 0.8% | 0.8% | 42% |

Under the reference recipe even a plain sentence (`The train to the coast
leaves at seven, ...`) and the 249-byte chunk (to `max_frames`) were
trapped. The one Fish Audio take reported stopped early is the `[pause]`
line, which never spoke `Seriously.` The harness's two-halves rule marks
only three of the nine trapped takes as failed. Every completed chunk used
at most 42% of its limit, so the `TextCap::MEASURED` constants still hold
under the new recipe. Fish Audio's recipe became the product default on this
evidence. The host was loaded by other sessions' builds, which changes
timing only; a seed fixes the frame counts.

**Stopped early.** `Drama3Take::stopped_early` is true when a chunk ends
without EOS, or the emitted audio holds a silent run of
`STUCK_SILENCE_FRAMES` (160 frames, 7.4 s, at -45 dBFS per 2048-sample
frame). Completed takes pause at most 78 frames, and one trapped take reached
EOS only after 359 silent frames, which a chunk-stop check alone would miss.
It reaches the app as `AudioResult.stoppedEarly`, and the app retakes such a
take with a fresh seed at most twice when the seed was left to it.
`crates/bench/tests/fish_drama3_pause_trap_gate.rs` holds the 24 takes above,
through the runner, to a frozen count. It measured 4 of 24, the same takes
and frame counts as the probe, and the longest silence in a completed take
was 50 frames. With the default reverted to the reference recipe it measured
12 of 24 and failed.

## Pause guard (2026-10-17, prototype)

Under Fish Audio's recipe an explicit `[pause]` still traps 3 of 6 takes
(seeds 0, 2 and 3 of `[laughs] ... [pause] Seriously.`). The guard below
resamples a pause that outlives every healthy one, inside the generation
loop, before any of it reaches the codec. It is built
(`GenerateOptions::pause_guard`, `PauseGuard::MEASURED`). The design came
first and the prototype measured it: the constants as designed recover too
few traps, so the default differs from the design in `hold_after` (0, not 64)
and `max_retries` (6, not 3). "Prototype results" below has the evidence and
what the change costs. The sections before it are the design as written.

### Measurements behind it

`crates/runtime/examples/fish_drama3_silence_probe.rs` speaks
`eval_corpus.txt` at seeds 0 and 1 and the gate's 24 takes on the runner's
path (default options, 8192 rows, 16 streamed slots, Metal tower and codec,
the codec streamed in 4- then 7-frame pieces). It records each draw's main
code `c0`, the decoded RMS of its frame, and the semantic head's EOS
probability and top 30 at temperature 1.0. The run gave 80 takes and 11,722
frames, 4,689 of them silent (RMS under -45 dBFS). It reproduced the gate's
4 of 24 and the corpus's one trapped clip (the `[pause]` line at seed 0, the
same take as gate seed 0).

The host was at load 9 to 20 from other work, which moves timing only. The
JSONL and the analysis scripts are kept outside the repository with the
session notes.

- **Silence is many codes.** The takes drew 2,670 distinct codes. A frame's
  own code predicts its RMS better than a neighbour's does (token-majority
  purity 0.966, against 0.932 and 0.951 one frame off). But the 94 codes
  drawn at least 5 times and silent at least 90% of the time cover only 80%
  of silent frames. In the traps the RMS silence starts about 140 frames
  before the codes settle on one deep-silence code (3174, drawn 1,233 times,
  all of them in trapped takes). A counter over a small fixed silent set
  starts that late.
- **Voiced codes are the learnable side.** Count a code as voiced when most
  of its frames were voiced, and every other code (silent-majority or never
  drawn) as silent. Held out by text (the gate texts are corpus lines 20, 21
  and 23, so a take's table excludes every take of its text), this calls
  91.6% of silent frames silent and 21% of voiced frames silent, mostly codes
  the held-out table never saw. 2,324 of 4,096 codes are voiced-majority over
  all 80 takes.
- **A pause counter that tolerates strays.** Add one per silent-classified
  frame. A voiced frame keeps the count going unless it is the third voiced
  frame in a row, which resets it to zero. Held out by text, its largest value
  on healthy takes is 55 (the 249-byte chunk at seed 1). The decoded RMS
  agrees: the longest healthy silence that speech followed is 50 frames (the
  `[pause]` itself, gate seed 4). Every trap passes 128: the count starts
  within 6 frames of the RMS onset in four traps and 27 frames late in the
  fifth (gate seed 2, whose tail stays diverse), and reaches 64 at frames 138
  to 199 and 128 at frames 202 to 263.
- **A trapped pause is not decided at its onset.** Speech mass is the head's
  top-30 probability on voiced-majority codes (a lower bound: unseen codes
  count as silent). At offsets 0, 24 and 48 frames into the RMS silence it
  averages 0.24, 0.08 and 0.05 in the five traps, against 0.18, 0.07 and 0.17
  in the 41 healthy pauses of 12 frames or more. The trap shows later. In
  three of the five, the distinct codes per 40 frames fall from 24 to 31 at
  the onset to 1 by 160 frames in. EOS stays under 0.01 until about 128
  frames in.
- **Restore points are cheap.** On the real trunk at row 232,
  `HiddenTrunk::snapshot` (a pooled blit of the 61 MiB recurrent state, then
  a wait for idle) took 0.548 ms median and 17.0 ms worst over 40 calls, and
  `restore` 0.540 ms median and 0.80 ms worst. The tower's `truncate` moves a
  cursor (Metal) or truncates host vectors (CPU), and its split-prefill and
  truncate replay is already gated bit for bit.

### The mechanism

In `generate.rs`, behind `GenerateOptions::pause_guard: Option<PauseGuard>`,
on by default. `None` restores today's loop for parity work and A/B runs.

| Constant | Designed (shipped differs where noted) | Why |
| --- | ---: | --- |
| `resume_frames` | 3 | Voiced frames in a row that end a pause. 2 misses one trap held out by text. |
| `hold_after` (H) | 64 frames (3.0 s). Shipped: 0 | Above every healthy count (55 token, 50 RMS). A pause shorter than this never touches the guard. |
| `retry_after` (K) | 128 frames (5.9 s) | Twice H. Healthy pauses under the old recipe reached 78 (dialogue, 2026-10-16), still well below. |
| `max_retries` (N) | 3 per chunk. Shipped: 6 | Bounds the extra work to N x (K - H) frames. Set from the prototype's recovery rate. |
| voiced table | 4,096-bit const | Voiced-majority codes from a training run disjoint from `eval_corpus.txt` (below). |

Per chunk:

1. Classify each drawn main token with the table and run the counter.
2. When the counter is about to pass H, take a restore point before the
   frame is fed: `backbone.snapshot()`, `tower.position()`, the current
   `hidden`, the previous `frame`, the RAS history, the chunk's frame count
   and the counter. The draw that triggered it is kept and fed as usual.
3. Hold every frame after the restore point: no `FrameEvent::Frame` for it.
   Each held frame sends a new `FrameEvent::Held { chunk }`, so a consumer
   still polls its stop flag once per frame.
4. If the pause ends (3 voiced frames in a row), EOS arrives, or the chunk
   reaches its cap, release the held frames in order and drop the restore
   point. That output is what the loop without the guard would have emitted.
5. If the counter reaches K, roll back: `backbone.restore`,
   `tower.truncate`, restore the copied state, drop the held frames, and
   draw again. The key sequence just continues, so the redraw differs and a
   seed still fixes the whole take.
6. After N rollbacks, release and keep going as today. The chunk then runs to
   its cap and the take reports stopped early, so the app's retake remains
   the backstop.

The cap counts frames on the kept path, so a rollback gives frames back.
`ChunkEnd` and `ChunkReport` gain `retries`. A take where the guard never
fires draws the same keys, feeds the same rows and emits the same frames as
the loop without it. The only differences are the counter and, past H, a
snapshot.

### Settled questions

- **Classifying silence without decoding.** Use the voiced table and the
  counter above, not a small silent set and not the decoded RMS. The decoded
  RMS is exact, but the codec decodes 7-frame pieces on its own thread. The
  restore point would then have to be taken before silence is known. The
  codec worker would also need a rewind protocol. Its stream state is host
  memory (a 128-frame attention window and conv carries), so that could be
  done. The table keeps the guard in the portable
  loop, and the RMS rule stays the judge of the result (`stopped_early`, the
  gate).
- **Where frames are held.** In the loop, not the runner as first proposed.
  The loop is the only producer of `FrameEvent::Frame`, so a frame it never
  sends never reaches a codec. `fish_drama3_say`, the runner and
  `/v1/audio/speech` all get the guard without changes. Holding in the runner
  would need the loop to tell it which frames to drop.
- **Snapshot memory.** A restore point exists only after a pause passes H,
  which no healthy take in the probe did. `GdnStateManager` pools snapshot
  buffers. The trunk then keeps at most two mirrors (the system block's and
  the guard's), 61 MiB above today's peak and only during a long pause.
  `open_bytes` already budgets two snapshots. The memory oracle's text has no
  such pause, so it would not see the second mirror. The prototype adds one
  trapped request (the `[pause]` line, seed 0) to the oracle and re-freezes
  the ceiling if the peak rises.
- **Streaming latency.** First audio is unchanged. A pause under H streams as
  today. Past H the stream stalls until the pause ends or rolls back. That is
  at most K - H = 64 frames (3 s at about real time) per attempt. A listener on a
  live stream hears that stall as silence, about what the trap would have
  played. A saved take, the app's (it collects the whole take before playing
  or retaking it) and a non-streamed `/v1/audio/speech` response drop the
  abandoned silence. `/v1/audio/speech` has no retake of its own, so for the
  server the guard is the only recovery. As shipped (H=0) every pause is held
  until it ends, so a streamed response delays each pause's frames by the
  pause's length (56 frames, 2.6 s, at most on the healthy takes measured),
  and an unrecovered attempt holds up to K = 128 frames (5.9 s).

### Open, and what the prototype must show

- **Whether a redraw from H recovers.** (Answered in "Prototype results":
  it rarely does at H=64, and the onset fallback is the shipped default.)
  Speech mass 48 frames in is similar
  in healthy and trapped pauses. That only says a redraw has a chance. The
  prototype measures recoveries per rollback on the trapped takes. If redraws
  from H rarely recover, the fallback is a restore point at each pause's
  onset. That costs a snapshot per pause (median 4 per take, about 2 ms) and
  holds every pause from its onset, which stalls live streams at every long
  pause. It needs its own sign-off.
- **The voiced table's reach.** It comes from voices the model picks with no
  reference, in English and one German line. A cloned voice or another
  language draws codes the table has not seen, which count as silent. The
  3-in-a-row reset tolerates many of them, and a false trigger only redraws
  from H. The training run adds Chinese, Japanese and one cloned voice. Its
  corpus is disjoint from `eval_corpus.txt`, so the gate and the WER check
  stay held out.
- **Done means:**
  - A fake-backbone unit test where the model stays silent until a redraw
    and the guard recovers. The same seed traps with the guard off.
  - A lower frozen count in `fish_drama3_pause_trap_gate.rs`.
  - `fish_drama3_wer` at seeds 0 and 1 with no WER change on passing clips.
    They should be byte-identical, since the guard should not fire on them.
  - The memory oracle with a trapped request.

### Prototype results (2026-10-17)

Built as designed: the loop holds frames from `hold_after`, rolls back to a
restore point at `retry_after`, and sends held frames only when the pause
ends. `crates/runtime/examples/fish_drama3_guard_sweep.rs` speaks sets of
takes at a list of settings (`hold:retry:retries`) through the runner.

**Mechanism checks.** Six unit tests in `generate.rs` drive a scripted
semantic head through the fixture tower and a fake backbone:

- A trapped pause recovers after a redraw, the same seed traps with the guard
  off, and nothing from an abandoned attempt is sent. The backbone and the
  tower are back at the restore point on every redraw, with the point's hidden
  state and previous frame.
- A pause that ends by itself goes out unchanged though its tail was held.
- Out of retries, the chunk runs on as without the guard.
- A stop while frames are held ends the call.
- Bad settings are refused.
- The shipped table matches its generated count.

Seven mutations were checked by hand and each fails a test: no tower
truncate, no backbone restore, no hidden restore, no frame-count truncate, no
previous-frame restore, no RAS-history restore, and no holding.

**Recovery.** The table comes from `voiced_table_corpus.txt`: 126 takes
(42 lines at seeds 0 to 2, 8 of them in a cloned voice; English, Chinese,
Japanese, German and Spanish), 14,452 frames, 2,763 of 4,096 codes voiced.
Applied to the earlier 80 eval takes, which it never saw, it calls 93.0% of
silent frames silent and 12.5% of voiced ones, and the counter peaks at 55 on
healthy takes. A trap's counter reaches 64 at frame 138 to 179 and 128 at 202
to 243.

Eight takes that trap with the guard off: the gate's four (held out of the
table) and four from the training run (in sample for it). The settings were
picked on these, so read them as an upper bound.

| hold:retry:retries | Recovered of 8 | Retries |
| --- | ---: | ---: |
| 64:128:3 (the design) | 2 | 22 |
| 64:128:6 | 4 | 40 |
| 32:128:3 | 5 | 17 |
| 32:128:6 | 7 | 25 |
| 16:128:3 | 6 | 12 |
| 0:128:3 | 6 | 12 |
| 0:128:6 | 8 | 16 |

Thirty fresh takes no setting was chosen on: the `[pause]` line at seeds 6 to
17, the dialogue at seeds 6 to 11, and two new texts at seeds 0 to 5. 11 of
them trap with the guard off.

| Setting | Recovered of the 11 | Healthy takes turned bad |
| --- | ---: | ---: |
| 64:128:3 (the design) | 2 | 0 |
| 32:128:6 | 6 | 0 |
| 0:128:6 (shipped) | 8 of the first 9 it reached | 0 |

0:128:6 stopped at its 26th take, which is one of the 11. That take spent all
six retries without recovering and then hit the error below. Two more of the
11 were not reached.

A redraw from deep in a pause rarely recovers because the head has settled
into silence: its speech mass averages 0.24 at the onset and about 0.05 from
48 frames in. Earlier restore points redraw while the choice is still open.
The design's fallback (a restore point at the pause's onset) is the better
setting, so H=64 is the wrong default.

**What an early restore point costs.** Only streaming latency: a held frame
reaches the codec when its pause ends. On the 75 healthy eval takes, with the
shipped table:

| hold_after | Healthy takes that hold anything | Restore points per take | Held frames per take | Longest single hold |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0 of 75 | 0 | 0 | 0 |
| 32 | 8 | 0.2 | 1.7 | 24 frames (1.1 s) |
| 16 | 26 | 0.5 | 6.5 | 40 frames (1.9 s) |
| 0 | 75 | 9.2 | 61.3 | 56 frames (2.6 s) |

A snapshot takes 0.55 ms and a restore 0.54 ms on the real trunk. The app
collects a whole take before playing or retaking it, so it feels none of the
hold. A streamed `/v1/audio/speech` response (`stream=true`) would hear a
healthy pause arrive late by up to the longest hold.

**Gate.** `fish_drama3_pause_trap_gate` with the shipped table: 4 of 24
stopped early with the guard off. At the design's settings (64:128:3) it is
3 of 24. The dialogue take (seed 3) recovers after 3 retries, and the target
line still traps seeds 0, 2 and 3 (seed 3 reaches EOS at frame 561 after 478
silent frames, which the silence rule still counts). At the shipped default
(0:128:6) it is 0 of 24, and the target line's seeds 0, 2 and 3 end by EOS at
222, 176 and 131 frames after 1, 4 and 1 retries, with longest silences of 76,
51 and 27 frames. The gate is frozen at 0.

**WER.** `fish_drama3_wer` on the 28-line corpus, guard at 0:128:6 against
guard off, same host and seeds. This setting holds every pause, so it is the
strictest check that holding changes nothing.

| Seed | Guard off | Guard 0:128:6 |
| ---: | --- | --- |
| 0 | 1 stopped early, mean WER 1.9%, passing clips 1.6% | 0 stopped early, mean WER 1.6%, passing clips 1.6% |
| 1 | 0 stopped early, mean WER 0.8% | identical: all 28 clips match in frames, WER and transcript |

At seed 0, 27 of 28 clips match in frames and transcript. The `[pause]` line
goes from 664 frames to the text cap, never speaking "Seriously.", to 222
frames ending in EOS after one retry, and Parakeet hears the whole line
(WER 0.0).

**Memory.** The memory oracle gains a request the guard rolls back (the
`[pause]` line at seed 0). Session peak against the 6,411 MiB ceiling:

| Guard | After the cloned run | After the paused run | Session peak | Retries on the paused request |
| --- | ---: | ---: | ---: | ---: |
| 64:128:3 (the design) | 6,291 MiB | 6,361 MiB | 6,364 MiB | 3 |
| 0:128:6 (shipped) | 6,357 MiB | 6,357 MiB | 6,357 MiB | 1 |

At H=64 only the paused request holds a second recurrent-state mirror, and it
adds about 70 MiB (6,291 to 6,361). At H=0 every take with a pause holds one,
so the warm-up and the cloned run already carry it (+66 MiB over the design's
6,291 after the cloned run). The pool reuses the mirror: a take makes about 9
restore points, and the warm replay grew the footprint by 0.00 MiB. The
ceiling holds with 54 MiB spare (0.8%, down from the 5% margin it was frozen
with). It is unchanged here, and re-freezing it at the 6,357 MiB peak plus 5%
(6,675 MiB) is a decision for review.

**A hazard the guard may raise.** The tower's nine residual heads have 4,096
entries but the codec's acoustic books have 1,024, and nothing masks the
heads. A residual code of 1,024 or more fails the whole request (`code 3403
outside the codebook of 1024`, raised by the codec). A scan of generated
frames found 0 of 6,722 with the guard off and 1 of 3,830 at 0:128:6: the
`Ready?` dialogue at seed 1, frame 403, after all six retries were spent.
That is the error that stopped the sweep. One frame cannot say whether the
guard raises the rate (through the extra silent frames it draws, or through
state after a rollback) or the rate is small either way. Sampling the
residual heads over the codec's 1,024 codes would remove it. That change is
separate from the guard and is not made here.

### Alternatives not taken

- **Bias the main draw after a long pause** (raise voiced codes or EOS). No
  snapshot and no hold, but every long pause then samples from a distribution
  the model did not produce. The redraw keeps every kept frame a draw from
  the model's own head.
- **Roll back without holding.** Simpler, but every rollback leaves K - H
  frames (3 s) of abandoned silence in the saved audio.
- **End the chunk after the last retry.** Saves the trap's tail, but a false
  trigger would then drop words. Running to the cap as today loses nothing.

### Gate by recipe and guard (2026-10-18)

`crates/bench/tests/fish_drama3_pause_trap_gate.rs` on origin/main
(9a3bf8db) plus the recipe change and the guard, pinned 4-bit install, 16
streamed slots, M4 Max. It counts takes `Drama3Take::stopped_early` reports,
selecting the recipe with `TURBOSPARK_DRAMA3_GATE_RECIPE` and the guard with
`TURBOSPARK_DRAMA3_GATE_GUARD`. Each count is frozen per configuration. No
request failed in the 96 takes.

| Recipe, guard | Stopped early | Voiceover (12) | Dialogue (6) | Pause cue (6) |
| --- | ---: | ---: | ---: | ---: |
| Reference, off (the loop before) | 12 of 24 | 4 | 4 | 4 |
| Fish Audio, off | 4 of 24 | 0 | 1 | 3 |
| Reference, on | 2 of 24 | 1 | 0 | 1 |
| Fish Audio, on (the default) | 0 of 24 | 0 | 0 | 0 |

With both, the guard retried four takes (dialogue seed 3 once, the pause cue
at seeds 0 and 3 once and at seed 2 four times), and each then ended by EOS.
The masking of the residual heads to the codec's 1,024 codes
(`81d7bced`, claude/wizardly-jennings-0c4e66) is not in this tree. No take
here drew a code past the book, so these counts do not depend on it.

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

**Router lookahead, measured.** `TURBOSPARK_PILOT_PROBE` now runs in the
`QwenGdnMoe` sequential decode path. Layer L's post-attention residual goes
through layer L+1's own `post_attention_layernorm` and router, and the guess is
filed beside the real selection. Qwen's router norm has a weight, so unlike
Gemma the probe owes its own norm encode. It is off by default and allocates
and dispatches nothing when off. The prompt rows run chunked prefill, which has
no probe, so they carry no guess and `--skip 52` drops them. One seed-0 clip
(52 prompt rows, 224 frames), `scripts/pilot_ceiling.py --skip 52`. The recall
is deterministic, so host load does not touch it.

- `TURBOSPARK_PILOT_PROBE=self` aims the probe at its own layer. Recall reads
  100.0% on all 40 layers, as it must. Two mutations of the layer bookkeeping
  fail it. Filing the guess one row off reads 3.2% (random is 3.1%). Forcing
  the next-layer target reads 54.3% (it is the real probe). The same two
  mutations fail the `router_hist` unit tests.
- The WAV is byte-identical with the probe off, on, and `self`, and equal to
  the pre-change build's.

Recall of the guess, as the share of the real top-8 named by the first k picks:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall | 11.5% | 21.6% | 30.0% | 37.0% | 42.7% | 47.2% | 51.0% | 54.3% |
| Pick in the real top-8 | 92% | 86% | 80% | 74% | 68% | 63% | 58% | 54% |

- The head of the ranking is good and the tail is not. The first pick is right
  92% of the time. All eight are right 54%. Gemma's full-width recall was 70.6%
  at a hit rate of 85%.
- Depth helps a little. Layers 1 to 9 read 55%, 10 to 19 read 46%, 20 to 29
  read 53%, 30 to 39 read 63%. The deep layers are the ones with the most SSD
  reads, so the SSD part is not predicted worse than the rest. First-touch
  misses are named 45% of the time at full width against 50% for evictions.

Priced at 16 slots, replaying the trace through the LFU cache with the guess
installed ahead of each pass. A prefetch takes a slot, so a wrong guess evicts
something. The plain sweep in the script ignores that. The last two columns do
not.

| k | Misses named | Reads per hit | Demand misses left | Bytes vs baseline | Hidden, ceiling |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.6% | 1.13 | 91.3% | 1.02x | 1.6 ms |
| 2 | 18.5% | 1.24 | 83.6% | 1.07x | 3.0 ms |
| 4 | 32.3% | 1.52 | 72.4% | 1.21x | 5.0 ms |
| 8 | 49.6% | 2.24 | 60.9% | 1.70x | 7.0 ms |

- Hidden is 18 ms times the demand misses removed. At 32 slots it is the same
  within 1 point (k=4 leaves 73.2% at 1.21x, k=8 leaves 60.4% at 1.69x).
- The SSD part, about 8 of the 18 ms, hides at the same fraction. That is
  0.7 ms at k=1, 2.2 ms at k=4 and 3.1 ms at k=8. This assumes the SSD misses
  are as predictable as the rest. The cold and deep-layer numbers above say
  they are, but the trace cannot say which reads hit the SSD.
- Full width does lose, as it did on Gemma. Eight wide reads 1.70x the bytes to
  remove 39% of the demand misses. It names half the misses and the wrong
  guesses push out experts the pass needed. The model also checks itself: a
  perfect guess at full width leaves 0.0% demand at 1.00x.
- Narrow does not lose. Nothing here is below 1.00x bytes, but k=2 to 4 adds
  7 to 21% bytes for 3 to 5 ms. On Gemma the best case was 0.66 ms of 24 ms,
  or 2.7%. Here k=4 is 5.0 of 42 ms, or 12%.

Those hidden figures are ceilings. They assume the prefetch overlaps fully and
costs no bandwidth. That is the weak point. The expert read is mostly a page
cache memcpy at 16 to 19 GiB/s, and the GPU competes for the same memory. The
window is also short. A layer takes about 1.1 ms of the 43 ms step, and a k=4
prefetch is about 7 MiB. Not measured: the contention, a prefetch thread and
its sync, and the cost of a wrong eviction in the real cache (the model assumes
the LFU victim). The probe's own dispatch cost is a small part of this. Its
cb1 GPU time reads 13.4 to 18.5 ms with the probe off and 14.1 to 14.9 ms with
it on, and best wall is 39.9 ms off against 40.0 ms on. The host ran at load 4 to 6, so
that bounds it under about 1 ms and no tighter.

**Recommendation: build, narrow, and gate it on an A/B.** The Gemma verdict
does not carry over. A 5 ms ceiling at 21% extra bytes is worth one prototype
and full width is not. Build a prefetch at k=2 to 4 behind
an env gate. Reads for layer L+1 run on a worker while layer L computes. Each
layer owns its streamer and slot buffers, so the worker and the main thread do
not share a cache. Keep it only if an interleaved A/B on a calm host shows 2 ms
per frame or more at 16 slots. If it shows less, delete it and close the lever
here too. Free memory sets the SSD share, so run the A/B at the host's real
free memory and again after a purge. This was one clip and one sentence.

**Prefetch prototype, not kept.** `TURBOSPARK_QWEN_PREFETCH=<k>` was built
to the recommendation. A worker takes layer L+1's streamer after layer L's
probe, reads the first k guessed experts that are not resident into slots the
next plan can still evict, and hands the streamer back before that layer
plans. The WAV is byte-identical at k=2, 4 and 8. The cache does what the
replay said. The hit rate goes from 34.3% to 44.7% (k=2) and 51.0% (k=4), and
demand reads fall from 355 to 299 and 265 MiB per step. The worker keeps up
with the window at k=2 and 4 (main thread waited 5.7 and 177 ms over a whole
clip) and not at k=8 (2,035 ms).

The 2 ms stop rule was not met. Five interleaved rounds of
`fish_drama3_step_cost` at 16 slots (frames, 160 steps) ran at host load 6 to
20, because other sessions were compiling. That is not the calm host the rule
needs, and the off arm alone ranged 47.6 to 77.7 ms.

| Faster than off, ms | Round 1 | 2 | 3 | 4 | 5 | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k=2 | 0.1 | -4.7 | 1.7 | 10.0 | 6.7 | 1.7 |
| k=4 | 7.2 | -2.3 | 21.4 | -2.9 | 0.3 | 0.3 |

- The demand expert I/O falls in all 10 matched cases. But cb1 GPU time rises
  in 6 of the 10 (round 2: 16.9 ms off, 22.1 and 21.0 ms with k=2 and 4).
  That looks like the bandwidth contention the ceiling left out, and it eats
  the gain. The runs cannot prove it at this load.
- The least loaded round (2, load 6 to 9) has prefetch slower by 4.7 and
  2.3 ms.
- So the ceiling of 3 to 5 ms did not show up in a loaded host. It is not
  ruled out on a calm one. Nothing here beats the 2 ms bar, so the code is
  not committed. The patch is kept in the shared notes folder
  (`qwen-prefetch-prototype.patch` and the new `prefetch.rs`).

**What is left.**

- More slots do not help (table above). Mapped residency is refused past the
  wired budget.
- A calm-host rerun of the prefetch A/B (host load under 2, one round per
  arm interleaved, 5 rounds) from the saved patch. If k=2 to 4 gains 2 ms
  there, land it. If not, close the lever.
- The probe works on Qwen 3.6 text decode as well. Text hits 63% at 16 slots
  with 11 ms of expert I/O, so its recall and price have not been taken and
  are not covered by this clip.
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
