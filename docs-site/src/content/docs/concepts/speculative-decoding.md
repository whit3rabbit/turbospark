---
title: Speculative Decoding: MTP and DFlash2
description: Why speculation is per-family on this engine, how the MTP head and the DFlash2 block drafter work, how the policy resolves, and which families it pays on.
---

Decoding costs one forward pass per token, and a forward pass costs nearly
the same whether it produces the logits for one token or for a block of
them. Speculative decoding exploits that asymmetry: a cheap drafter
proposes several tokens, the target model verifies all of them in one
batched pass, and the longest matching prefix is committed, plus one free
bonus token from the row after the last accepted proposal. When the
drafter is right, several tokens leave per target pass. When it is wrong,
the engine pays a rollback: restore state, replay the accepted prefix.

On this engine speculation is a property of a family, not of the engine.
The drafters are checkpoint components, tensors shipped beside the trunk,
not generic heads the engine can bolt onto any model. Both drafters this
port runs live on the dense `qwen3_5` line and are wired in
`crates/runtime/src/families/qwen/`: the checkpoint's own
multi-token-prediction (MTP) head and the DFlash2 block drafter. Whether
an install can speculate is decided by what its bytes contain and by an
architecture check, and the answer is refused with a reason rather than
silently degraded.

## The MTP head: step-wise drafting from the checkpoint's own module

The MTP head is the checkpoint's own drafter. It ships in the official
checkpoint's last shard; the common mlx-community conversions drop the
`mtp.*` tensors, so an install carries one only if that shard was
streamed. Detection is a presence check, `mtp.fc.weight` in the resident
index, with no manifest field that could disagree with the bytes
(`families/qwen/mtp_state.rs`, `install_has_mtp_head`).

The head is one full-attention block whose tensors are a trunk
full-attention layer's shape field for field, so the draft step runs the
trunk's own encoders under `mtp.layers.0.*` names. No new Metal kernel
and no new dispatch shape (`families/qwen/mtp.rs`). What it adds beyond
the block is `fc`, a plain GEMV over a `[2 * hidden]` buffer, and three
RMS norms:

```text
x   = fc([ norm_e(embed(next)), norm_h(h_t) ])   // 2H -> H
x   = x + attn(input_layernorm(x))               // the head's OWN KV
x   = x + ffn(post_attention_layernorm(x))
out = lm_head(mtp.norm(x))                       // the TRUNK's lm_head
```

The head has neither an embedding table nor an output head; both are
shared with the trunk, which is what makes it 849 MB rather than several
GB, about 1.5% of a forward pass once quantized to INT4.

Two conventions carried over from the checkpoint matter more than they
look. The head's seven norms (five whole-vector ones plus `q_norm` and
`k_norm`) are centered, `x * (1 + w)`, where the trunk's tensors of the
same names are plain `x * w`. And the hidden half of `fc`'s input is the
trunk's POST-final-norm state, not its residual stream: `model.norm`
carries a learned per-channel weight, so feeding the residual instead is
a different direction, not a scale error the next norm absorbs. Getting
either wrong still produces finite, plausible logits; the head just stops
agreeing with the trunk, and the only symptom is a collapsed accept
length.

The head drafts one token per step. Depth beyond one is the caller's
loop: the drafted token is fed back as `next_token`, where the head's own
residual stream stands in for the trunk hidden state it cannot know yet.
That chaining approximation is inherent to drafting more than one token
from a single module and is one of the two things accept length measures.
Before decoding, the head is primed over the prompt with
`mtp_prime_step`, a step taken for its KV row alone with the full-vocab
head skipped; the head's attention span comes from the position argument,
never from a cursor, so an unprimed head attends over rows nobody wrote
and quietly costs accept length with no error anywhere.

The head's state (`MtpState`) is deliberately small: its own one-layer
`KvCacheManager` built from a cloned `ArchConfig` (one full-attention
layer, about 4 KiB per token on this architecture), the `[2 * hidden]`
concatenation buffer `fc` reads, and the M-row scratch the batched verify
runs on. Widening the trunk's cache was not an option: its sizing is what
every family's memory-oracle peak is frozen against. A round of block B
takes `B + 1` head steps for `B` proposals; the extra step is taken for
its KV row alone, because a round where every proposal is accepted needs
a row the proposal-producing steps do not write, and the fully-accepted
case is the one a good drafter hits most often.

## The DFlash2 block drafter: one pass proposes the whole block

DFlash2 is a separate, independently published block-diffusion drafter
for the same dense architecture (`incoai/Qwen3.8-27B-DFlash2`, streamed
beside the trunk; `families/qwen/dflash_state.rs`). Where the MTP head
drafts a token at a time, DFlash2 proposes a whole block in ONE forward
pass and the target verifies the block in one batched pass.

Three facts about the drafter organize its implementation
(`families/qwen/dflash.rs`):

1. **Its KV cache holds TARGET-derived rows.** Every round, the trunk
   states the previous verify pass captured at layers `[5, 19, 33, 47,
   61]` are fused by `fc`, normed, and projected by the drafter's own
   `k_proj`/`v_proj` into its cache. The drafter never runs its layers
   over the context; a draft pass is always and only the `block + 1`
   query rows: the bonus row carrying the anchor token's embedding, and
   the proposal rows embedded as the mask token (`DFLASH_MASK_TOKEN`,
   248070).
2. **The capture buffer is the `fc` input, already laid out.** Five aux
   states per row, concatenated `[row][aux][hidden]`, is exactly the
   `[rows, 5 * hidden]` matrix `fc` multiplies, so nothing rearranges a
   captured row between the trunk pass and the context write.
3. **Candidate selection runs on the host.** A round is one command
   buffer for the context KV, one for the block forward (embeddings, the
   drafter's layers, final norm, two head GEMMs: the trunk's `lm_head`
   for the selector's candidates and unary scores, the drafter's
   `hidden_projection` for its edge scores), then a host phase over two
   readbacks: a top-16 scan over the vocabulary, codebook row gathers,
   and one bilinear score per step (`dflash_draft/`, `DFLASH_TOP_K`).
   Microseconds beside a pass that reads a gigabyte of weights. A
   drafter's arithmetic is a throughput axis, never a correctness one;
   the target verifies every proposal.

The drafter's residual stream is held divided by
`DFLASH_RESIDUAL_SCALE` (8.0), with the norms reading it taking an eps
divided by 64, because its true residual peaks at 113,920 against FP16's
finite ceiling of 65,504. RMS norm is scale-invariant, so the division
cancels everywhere except that eps.

Because the drafter's cursor advances only at the next round's context
write, the loop's rewind request lands AHEAD of the cursor in the normal
case, where the step-wise MTP head is genuinely walked back. That is the
structural difference between a block drafter and a step-wise one: for
DFlash2, rewinding is a cursor move, and rows past the cursor are stale
by construction and always rewritten before anything reads them again.

## How the policy resolves

The flags are `--speculative off|auto|N` (a block size 1-15) and
`--speculative-drafter auto|mtp|dflash` (`crates/invocation/src/options.rs`).
The server takes the same pair; the C ABI takes the drafter choice as an
option key.

The policy lives in the runtime crate (`speculation_policy.rs`) rather
than in either binary, because both front ends need the same three
decisions in the same order:

1. `resolve_drafter` reads the install's resident index (kilobytes, not
   the weights) and says which drafter the install carries, recording
   both presence answers even when the caller named a drafter
   explicitly.
2. `draft_policies` turns the choice plus the request into the
   `DraftPolicies` to open the runner with. The drafter the flag named
   owns the request and the other is pinned off, so a dflash-carrying
   install never silently opens both drafters' state.
3. `resolve_speculation` decides whether this run may draft, from the
   request, the drafter, the engine's own blocker, and whether the run
   is deterministic.

All of this resolves once at open, because drafter state is allocated
there and there is one runner per process. At the CLI the resolved plan
applies to every turn of the session (`crates/cli/src/generate/session.rs`).
On the server, the install half is fixed at open and the per-request
half is applied per call (`crates/server/src/real_model.rs`).

The two request spellings fail differently, on purpose. A named block is
a promise the caller made to themselves, typically for a measurement, so
failing to keep it is a hard error carrying the reason. `auto` asked for
"speculate if you can", so the same condition is a warning and the run
continues sequentially. Relatedly, an unparseable env value on the
engine seam (`TURBOSPARK_MTP_DRAFT`, `TURBOSPARK_DFLASH_DRAFT`) resolves
as `auto` rather than off: a typo should not silently disable a feature
the install can serve. And with no drafter asked for, nothing is
allocated at all, so the off path is identical in bytes and footprint to
an engine that never had the module.

**Detection is not enablement.** `auto` resolves to the MTP head even on
an install whose only drafter is DFlash2, and returns a note naming the
flag that would run it. The reason is measured, not stylistic: through
the shipped loop over 600-token generations, DFlash2 at block 2 reads
1.33x on code and 1.47x on math against 0.90x on prose, and its energy
A/B reads +17.4% J/token on the protocol's prose case. A default that
switched it on would make one common workload slower and hungrier
without being asked. The MTP head is the opposite case (1.44-1.66x) and
keeps its `auto`. Resolving to Mtp also means open allocates no DFlash2
state, which is 213 MiB of peak footprint on the real 27B install.

## Why acceptance is exact only at temperature 0

Acceptance in the loop is `target == proposal`, an argmax agreement
(`speculative.rs`). That is exact speculative decoding at temperature 0
and wrong at any other temperature: a sampled run that only ever accepts
the target's argmax is biased toward the mode, silently narrowing the
distribution the caller asked for. The correct algorithm for the sampled
case is rejection sampling with residual correction (Leviathan et al.,
arXiv 2211.17192; Chen et al., arXiv 2302.01318), which needs both
models' full distributions rather than their argmaxes and is not
implemented here.

So the sampled case is refused, not approximated. A quiet fallback to
greedy would change what the model writes while reporting success, and a
quiet fallback to the sequential loop would report a speculative run that
never speculated. The loop itself returns an error
(`RuntimeError::SpeculationUnavailable`); the policy layer decides what
that error means, which is where the two front ends part company:

- At the CLI, temperature is a property of the single run, so the whole
  plan is resolved once at open and a sampled run simply never
  speculates.
- On the server, determinism is a property of the REQUEST. The install
  half was resolved at open; per request, a deterministic one takes the
  speculative loop and a sampled one falls back to the sequential (or
  chunked) loop silently.

The server consequence is worth stating plainly: most clients send a
non-zero temperature, so a server started with speculation speculates on
a minority of its traffic.

## Tradeoffs: the per-family measured verdicts

**Block 2 is the optimum for both drafters, and that is a property of
this engine.** `DEFAULT_SPECULATION_BLOCK` is 2 (MTP) and
`DFLASH_SERVING_BLOCK` is 2 (DFlash2), independently measured. Verify
cost scales close to linearly in the block, the accept chain decays, and
the probability a round must roll back compounds: 10% at block 2 rising
to 98% at 15 for the MTP head, and on DFlash2 prose 52% at block 2
rising to 96% at 8 even though per-position acceptance on code and math
runs 0.84-0.98. The mechanism is the rollback term: this family's
recurrent gated-DeltaNet state cannot be rewound incrementally the way a
KV cursor can, so a rejected batched round restores a whole state
snapshot and replays the accepted prefix as a second batched pass. Two
drafters, two architectures, one answer.

**Dense pays.** The MTP head measures 1.44x at block 2 through the
accept-length probe, and 1.66x once the head's norm conventions were
corrected. DFlash2 accepts about 7 of 8 proposals per round at its
trained block, and at the serving block of 2 measures 1.33x on code,
1.47x on math, and 0.90x on prose: which is exactly why `auto` leaves it
off and it is opt-in.

**MoE does not pay, twice over.** First as arithmetic: measured on the
real Qwen 3.6 35B-A3B install, speculation is worth about 1.1x at best,
because the composite verify cost is dominated by a 19% un-amortizable
per-token floor no kernel moves and by the routed expert pair; the batched
kernel fix that halved the largest single term moved the answer about
two points, because the two terms that dominate were untouched
(`docs/SPECULATIVE_DECODING.md`). Second as policy: the speculation
blockers refuse a MoE install because no published MoE conversion of
this architecture ships a drafter this port can ingest (every mlx
conversion drops `mtp.*`, and the published DFlash2 drafter targets the
dense half). The batched routed verify itself runs, so the blocker
reports a checkpoint gap, not a missing kernel.

Two more verdicts that shape the surface:

- **The verify is INT4-only.** The batched GEMM has one arm; the 1-bit
  and 2-bit checkpoints of this architecture decode normally but cannot
  speculate, and the blocker says so rather than letting the first verify
  fail mid-generation.
- **A sequential verify never overshoots; only a batched pass can.**
  That asymmetry is what makes block 2 pay and block 8 lose: the batched
  pass computes rows for proposals it is about to reject, and pays the
  restore on top.

One cost worth knowing that is not in the round: the speculative loop
runs every prompt token through the full `produce` rather than the
head-skipping prefill path, because the drafter reads the trunk's hidden
state for each position. Every number above was measured in that
configuration.

## Limits

- **`qwen4_exp` is not wired.** The drafters and the policy's imports
  live under `families/qwen`; the `families/qwen4` flow has no MTP or
  DFlash2 modules, and nothing in the policy reaches it.
- **The sampled case is out of scope by design** until rejection
  sampling with residual correction is implemented; see above.
- **Cancellation granularity is the committed token.** The loop polls
  once per committed token through the same sink a sequential decode
  uses, but there is no poll during a verify pass, which is one batched
  forward and not interruptible.
- Acceptance is observable through `TURBOSPARK_SPEC_STATS=1`, which
  prints rounds, accepted per round, per-position acceptance, and
  rollback counts; it exists because both probes hand-roll their own
  round and a loop-level acceptance gap is invisible without it.

## Where the code lives

- `crates/runtime/src/speculative.rs`: `run_raw_completion_speculative`
  and its cancellable sibling, the sampling gate, the round loop
  (draft, batched verify, accept, rollback, bonus token).
- `crates/runtime/src/speculation_policy.rs`: `Speculation`,
  `SpeculativeDrafter`, `DrafterChoice`, `SpeculationPlan`, and the
  three resolvers.
- `crates/runtime/src/families/qwen/mtp.rs` and `mtp_state.rs`: the
  head's draft, prime and rewind steps, its state, presence detection
  and the MTP speculation blocker.
- `crates/runtime/src/families/qwen/dflash.rs`, `dflash_state.rs` and
  `dflash_draft/`: the DFlash2 state, constants, and the
  context-write / block-forward / host-selector round.
- Front ends: `crates/cli/src/generate/session.rs` and
  `crates/server/src/real_model.rs` call the three policy functions.
- The measurement records live in the repo as `docs/MTP.md`,
  `docs/MTP_SPECULATIVE.md`, `docs/DFLASH2.md` and
  `docs/SPECULATIVE_DECODING.md`.
