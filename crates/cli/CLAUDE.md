# turbospark-cli

TWO process entry points. `turbospark-check` parses `argv` using
`turbospark-invocation`, applies exit status and output stream routing, and
drives GPU token generation (`RealForwardRunner`) on macOS.
`turbospark-model` is the catalog and download surface, backed by
`turbospark-catalog`.

## Directory & File Structure

```
crates/cli/
+-- Cargo.toml              # Crate manifest, declaring BOTH binaries
+-- src/
|   +-- main.rs             # turbospark-check process entry point
|   +-- generate/           # Non-interactive text & chat template generation driver
|   |   +-- mod.rs          # Generation loop coordination and session management
|   |   +-- session.rs      # Session lifecycle; maps invocation enums onto runtime's
|   |   \-- format.rs       # Prompt/footer rendering and channel splitting
|   +-- chat.rs             # Interactive REPL session runner using window-fit
|   \-- bin/
|       +-- model.rs        # turbospark-model: argv, subcommand parse, exit codes
|       \-- model_cmd/
|           +-- mod.rs      # The seven subcommands
|           \-- render.rs   # Printing. No decisions.
\-- tests/
    +-- mference_check.rs   # CLI flag parse & exit status integration tests
    +-- real_generation.rs  # End-to-end real generation integration tests
    \-- model_cli.rs        # turbospark-model argument surface & exit codes
```

## Key Modules

- `main.rs`: Reads command-line arguments, delegates parsing to `turbospark-invocation`, prints resolved requests, and routes execution to generation routines.
- `generate/`: Coordinates tokenizer loading, chat template rendering, prefill chunking, and GPU decode generation loops. `open_session` resolves `--model` through `catalog::resolve_model_arg` first (see Gotcha 5) and maps the parser's enums onto `runtime`'s. **The speculation POLICY is no longer here**: it moved to `runtime::speculation_policy` when `turbospark-server` needed the same three decisions (see Gotcha 10).
- `chat.rs`: Interactive REPL loop maintaining user/assistant turn history and applying `fit_conversation_window` to manage context window bounds.
- `bin/model.rs`: `turbospark-model`'s argv parse and exit-code mapping. **A second binary rather than subcommands on `turbospark-check`, and that is a decision**: `turbospark-invocation` is a pure, flat option parser whose contract is "`--model` is required and exactly one mode flag is set", with a five-place rule for every new flag and a hardcoded option-count assertion. A subcommand grammar does not belong in it, and bending it into one would put a required `--model` in front of a command whose entire job is that there is no model yet. Two exit codes, and a script doing `probe X && pull X` depends on the difference: 2 for a malformed invocation, 1 for a run that was asked for correctly and did not work.
- `bin/model_cmd/`: the seven subcommands (`list`, `info`, `probe`, `recommend`, `pull`, `path`, `rm`). **Nothing here decides anything** -- `turbospark-catalog` resolves rows, reaches verdicts and runs the walk; this module chooses column widths. Same split `main.rs` has with `invocation`, and it is what lets the verdict logic be tested without a terminal.

## Development & Test Commands

```sh
# Run tests for turbospark-cli
cargo test -p turbospark-cli

# Run CLI against a model with prompt string
cargo run -p turbospark-cli --bin turbospark-check -- --model /path/to/model --prompt "Hello"

# Interactive chat mode
cargo run -p turbospark-cli --bin turbospark-check -- --model /path/to/model --chat

# The catalog and download surface (docs/MODELS.md).
cargo run -p turbospark-cli --bin turbospark-model -- list
cargo run -p turbospark-cli --bin turbospark-model -- probe owner/name
cargo run --release -p turbospark-cli --bin turbospark-model -- pull tinyllama

# Then, with no path anywhere:
cargo run --release -p turbospark-cli --bin turbospark-check -- \
  --model tinyllama --messages-file /tmp/p.json
```

## Real-Model Smoke Tests (Run Before Handoff)

Always run BOTH greedy and sampled smoke commands whenever altering decode, KV cache, output head, or Metal encode logic:

```sh
cargo build --release -p turbospark-cli
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json

# 1. Greedy generation (catches math bugs)
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 1 --temperature 0.0001 --top-k 1

# 2. Sampled generation (catches distribution bugs greedy cannot see)
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 20260721
```

## Crate Gotchas

1. **Greedy is Not a Full Smoke Test**: `argmax` is invariant under monotone probability transformations. A broken logit distribution will often output identical greedy tokens while failing catastrophically under sampling. Always verify sampled output coherence.
2. **The power profile is resolved ONCE, in `open_session`.** That call is the only place this process asks the OS about Low Power Mode, and it happens before the first turn so an interactive `--chat` session cannot change pace mid-conversation because the machine was plugged in. `Session.rate` then feeds both `GenerationConfig` literals (`run_prompt` and `stream_turn`). `--max-tokens-per-sec` overrides whatever cap the profile carries, in both directions, so `--power-profile performance --max-tokens-per-sec 8` paces at 8 without enabling thermal stepping. `invocation::PowerProfile` and `runtime::PowerProfile` are separate enums (the parser crate is pure and depends only on `foundation`); `map_power_profile` is the single place they meet.
3. **Chat Template Necessity**: Running `--prompt` on an instruction-tuned model yields babble because special chat markup (`<|turn>`) is absent. Use `--messages-file` or `--chat` to ensure chat templates are rendered correctly.
4. **STDOUT is the answer and STDERR is the reasoning -- always on `gpt-oss`, and on ChatML or Gemma whenever `--reasoning` asked for a level.** Harmony puts the model's reasoning in an `analysis` channel before its answer, so `ChannelSplit` runs that dialect's output through `StructuredAssistantDecoder` and routes the two streams apart; redirecting stdout therefore captures the answer alone. ChatML and Gemma join it only when a level was requested, because their thought channels are unreachable otherwise. With no `--reasoning` and no gpt-oss install, no decoder is built at all and the printing path is byte-identical to what it was. **Skipping this is not cosmetic**: measured on the real Gemma 4 install, a `--reasoning low` run with no decoder printed a bare `thought`, then the scratch work, then the answer, as one run of content. **Only the ANSWER accumulates into the returned reply**, which is what `chat.rs` appends to history: that is a correctness point rather than cosmetics, because Harmony's own convention drops the analysis channel from prior turns and feeding it back sends the model framing it was never trained to read. Note the consequence for any test asserting stderr is empty: on a gpt-oss install the reasoning is expected there. `tests/real_generation.rs` no longer makes that assertion in any mode -- all three now print the shared `[stop=...]` footer to stderr (`--prompt` used to print its own summary to stdout instead, and with it skipped the withheld `Tail`).

   **TOOL CALLS ARE OFF HERE BY CONSTRUCTION, not by omission.** `ChannelSplit` builds its decoder with an EMPTY allowlist, and a Harmony call is parsed only when the caller offered that tool by name (tokenizer crate Gotcha 5), so a `commentary to=functions.x` body stays reasoning and prints to stderr with the rest. `turbospark-check` has no way to run a tool and no shape to render one in; the server is where that lives.

   **DO NOT SKIP AN EMPTY DELTA BEFORE THE SPLIT.** `stream_turn` used to return early on empty text, which is harmless for five families and total for this one: the detokenizer skips special tokens, so EVERY Harmony frame token (`<|channel|>`, `<|message|>`, `<|end|>`, `<|start|>`) arrives as `(id, "")`. Skipping those means the state machine never sees a single transition and the whole turn prints as one run of content, markup words and all, with no error anywhere. Measured on the real install: the first end-to-end run after wiring the splitter printed `analysisThe user asks...assistantfinalThe sky appears blue...` to stdout, which looks exactly like a decoder that was never built. Every transition arrives as an empty delta; the emptiness check belongs AFTER `ChannelSplit::push`, on its output.

6. **`--expert-cache-slots` DEFAULTS TO `auto`, so this binary's tok/s is a property of the machine and its startup line is the only thing that says which.** `open_session` maps `invocation::ExpertCacheSlots` onto `runtime::ExpertCacheSlots` and calls `open_with_slot_policy`, which resolves against physical memory and the install's expert stride; `runner.expert_cache_slots()` is then echoed to stderr, because under `auto` the REQUEST carries no number and a line echoing it would describe nothing. On this machine the 13 GB Gemma 4 install resolves to 32 slots, which is ~51 tok/s against ~44 at 16 for ~1.6 GB more peak (`docs/DECODE_BUDGET.md`).

   Three consequences for anyone working here. **A tok/s footer from this binary is not comparable to a `docs/BENCHMARKS.md` row** unless `--expert-cache-slots 16` was passed -- every harness pins that constant and this binary no longer does. **A throughput A/B must pin the flag on both arms**, or a machine that resolved differently between two runs (a big download finished, another model opened) silently varies the wrong thing. And **the smoke md5s are taken over stdout, which includes the resolved-request block**, so `expert_cache_slots: Auto` vs `Fixed(16)` changes the hash while the generated text does not: compare `sed -n '/^generating/,$p'` output, or pin 16 and patch that one line back before comparing against a historic hash. Both standing references (`67a23bb5...` greedy, `ebfba17a...` sampled) were re-derived that way when the default changed and are unmoved.

   The mapping is the same shape as Gotcha 2's `map_power_profile`: two enums, one in the pure parser crate and one in `runtime`, meeting in exactly one place. The parser may not look at the machine or the install, and sizing needs both.

5. **`--model` takes a path OR a catalog alias, and the PATH always wins.**
   Resolution lives in `generate.rs` via `catalog::resolve_model_arg`, not in
   `turbospark-invocation`, which is pure and whose contract keeps the value an
   opaque string. The order is load-bearing rather than a tie-break: a bare
   name that silently preferred an alias would run a DIFFERENT model than the
   one on the command line, fluently, with no error and with a perfectly
   plausible tok/s footer. An unresolvable name is passed through unchanged, so
   a machine with no `HOME` reports the same "no such install" it always did.
   `crates/catalog/tests/store.rs` pins both directions.

7. **`--prefill-chunk` IS WIRED AS OF 2026-08-26, and NOT by consuming
   `request.prefill_chunk` unconditionally.** `stream_turn` calls a
   `resolve_chunk_tokens` helper: `MFERENCE_PREFILL_CHUNK` still wins first
   (unchanged env-seam contract, see below), and otherwise the flag's value
   (`Fixed(n).resolved()` or `Auto -> DEFAULT_CHUNK_SIZE`, via
   `invocation::PrefillChunk::resolved`) is used ONLY when
   `session.runner.supports_chunked_prefill()` says this install's family can
   serve it -- else `None`, the sequential path, with NO error. That
   asymmetry is deliberate: the flag defaults to `Fixed(128)` on every
   invocation whether or not the caller typed it, so an install the chunked
   driver doesn't serve must fall back silently rather than error on a
   caller who never asked for anything. `supports_chunked_prefill()` is the
   SAME predicate `ChunkedPrefillRunner::prefill_chunk`'s own refusal uses
   (`crates/runtime/src/real_forward_api.rs`), so the two can't disagree.
   Two families serve it today: Gemma 4, and the DENSE half of `llama`
   (Mistral, Llama 2/3.x, `families/llama/prefill.rs`; ROADMAP.md's PF-02
   section has the full list of what's still unserved and why).

   Verified end to end on real installs (not just the synthetic parity
   suite): greedy and sampled stdout are md5-IDENTICAL between a
   pre-wiring binary (sequential by default) and the current one (chunked
   by default) on both `~/models/gemma4.gturbo` and
   `~/.turbospark/models/mistral7b.gturbo`. `MFERENCE_PREFILL_CHUNK=64` on
   an unsupported family (checked against `gptoss-20b.gturbo`) still hits
   the named hard refusal; the same install with no env var and the default
   flag generates normally with no error at all.

   Two consequences carried over from when this was a bare seam. **The env
   var is STILL an A/B seam whose two arms must produce identical tokens**,
   like `MFERENCE_SHARED_CB` next door: verified on the real install at
   chunk spans 32, 128 and 512 against the frozen greedy and sampled
   digests. And **the resolved-request block already printed
   `prefill_chunk` before this landed**, so nothing about wiring the flag
   moved a printed field -- unlike the `expert_cache_slots` case in
   Gotcha 6, the standing stdout digests did not need re-deriving for this
   change (confirmed by the same md5-identical comparison above).

8. **`--max-context` DEFAULTS TO `auto`, and the resolved number lives on
   `Session`, never on the request.** `open_session` resolves the window
   before opening (the failure it catches is an allocation), stores
   `plan.resolved` on the session, and every downstream consumer -- the
   admission check in `clamp_max_new`, `run_raw_completion`'s bound,
   `chat.rs`'s window fitting -- reads THAT. Under `auto` the request carries
   no number at all, so a consumer reading `request.max_context` would be
   fitting a conversation against a window the KV cache was not allocated at.

   Two consequences for reading a run. **The stdout md5 moves and the
   generated text does not**, exactly as it did when the slot default changed
   (Gotcha 6): the resolved-request block now prints `max_context: Auto`
   instead of `4096`, so `sed 's/max_context: Auto/max_context: 4096/;
   s/Fixed(16)/16/'` over the full stdout is what reproduces the standing
   references. Both were re-derived that way when this landed and are
   UNMOVED (`67a23bb5...` greedy, `ebfba17a...` sampled). And **an install
   that declares no trained context resolves to 4,096**, which is every
   install written before that field existed -- so nothing already on disk
   changed footprint, and a `context:` line reading 4,096 on a machine with
   room for far more is the install's silence, not a cap.

   The startup line reports the resolved window, the checkpoint's own trained
   context, the KV bytes and what `auto` would have chosen, for the reason the
   expert-cache line reports the resolved slot count: the window is most of
   the KV footprint, so no peak or prompt refusal is comparable across runs
   without it.

9. **This crate has TWO binaries and NO lib target, so `tests/*.rs` cannot
   reach anything in `src/`.** An integration test may only drive the built
   binaries as processes (`mference_check.rs`, `model_cli.rs` do). Logic worth
   unit-testing -- `resolve_speculation`, `map_power_profile` -- takes a
   `#[cfg(test)] mod` inside its own module instead. Adding a lib target to
   avoid that would put every private helper on a public surface.

10. **`--speculative` decides ONCE, in `open_session`, and the hard-fail/warn
   split is the whole contract.** Both of its inputs (does the install carry a
   usable drafter, is this run deterministic) are fixed for the process, and a
   `--chat` session that started speculating must not stop silently three turns
   in. `Speculation::Block(n)` is an ERROR when it cannot be served -- a caller
   who named a block is measuring, and a run that quietly did not speculate is
   the number that ends up in a table -- while `auto` WARNS on stderr and
   decodes sequentially, because most installs carry no head and a hard error
   would make the common case a failure. `off` is silent, deliberately: a
   warning there would train people to ignore the one that matters.

   **THE POLICY LIVES IN `runtime::speculation_policy`, NOT HERE.** It was
   `generate/speculation.rs` until 2026-08-21, when `turbospark-server` needed
   the same three decisions in the same order and could not reach a line of it
   -- this crate has two binaries and no lib target (Gotcha 9). `open_session`
   now maps `invocation::Speculation` / `SpeculativeDrafter` onto the runtime's
   through `map_speculation` / `map_drafter`, the same two-enums-one-mapping
   shape Gotchas 2 and 6 describe, and calls `resolve_drafter`,
   `draft_policies` and `resolve_speculation`. The nine policy tests moved with
   it and gained a tenth covering `draft_policies`, which had none while it was
   an inline `match` reachable only with a 14 GB install in hand.

   **`auto` DETECTS BOTH DRAFTERS AND ENABLES ONLY ONE.** `resolve_drafter`
   reads the resident index (kilobytes, before the open) and returns a
   `DrafterChoice`: an MTP head resolves to `Mtp` and is switched on, while a
   DFlash2-only install ALSO resolves to `Mtp` -- so `open` allocates no
   DFlash2 state -- carrying a `note` that names `--speculative-drafter
   dflash`. The asymmetry is measured, not stylistic: through the shipped
   loop the head pays 1.44-1.66x while DFlash2 at block 2 reads 1.33x on code
   and 1.47x on math against 0.90x on PROSE, with its power arm at +17.4%
   J/token, so enabling it by default makes the common workload slower and
   hungrier without being asked. Resolving to
   `Mtp` also saves 213 MiB of peak footprint, measured as the gap between
   the two arms of one protocol case on the real 27B.

   The note OUTRANKS the engine's own blocker as the disabled reason. Both
   are true of such an install -- it has no MTP head, and its DFlash2 drafter
   was deliberately left off -- and only one names something the caller can
   act on. Under a NAMED block the same note becomes the hard error, which is
   right: `--speculative 2` does not say which drafter, and the message says
   which flag would. An install carrying BOTH gets no note, because the head
   is being used and there is nothing to explain; that case is the only input
   on which the rule differs from the weaker "does it have dflash", which is
   why `crates/repack` grew a both-drafters fixture to pin it.

   **A NAMED BLOCK ON AN INSTALL WITH NO DRAFTER OPENS ANYWAY, so the reason
   comes from the blocker rather than from the open.** `draft_policies` maps
   `Block(n)` onto `Off` when `resolve_drafter` established the install has no
   drafter of the named kind, for BOTH drafters (the MTP guard landed
   2026-08-21, the DFlash2 one 2026-08-22). Without it the open fails first
   and names the wrong obstacle: on the MoE `ornith35b`,
   `--speculative-drafter dflash --speculative 2` used to say "stream it
   beside the trunk", advice no artifact can satisfy because the published
   DFlash2 drafter targets the DENSE half of that architecture. The hard fail
   is unchanged either way; only the sentence improves.

   **VERIFYING ANY OF THESE REFUSALS END TO END NEEDS `--temperature 0`.** The
   SAMPLED refusal is resolved ahead of the drafter's and this binary defaults
   to 0.2, so a run meant to exercise a drafter message reports "acceptance is
   exact only at temperature 0" instead, which reads like the case passing.

   Two things not to re-derive. The reason string comes from
   `RealForwardRunner::speculation_blocker()` and is never rebuilt here -- the
   engine owns the conditions it refuses on, and a second copy in the CLI would
   name the wrong cause the first time they disagree. And the SAMPLED case is
   refused rather than downgraded to greedy: acceptance is
   `argmax(target) == proposal`, exact only at temperature 0, so approximating
   it would change what the model writes while reporting success. That makes
   `--speculative` unreachable at this binary's own defaults (T=0.2) until
   rejection sampling lands.
