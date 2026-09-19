# SKILL.state: Bounded-State Agent Runtime (Measured Positive)

**In English: an agent doing 50 tool steps finishes instead of dying at step
30-35 with an HTTP 400. It costs 3-5x fewer tokens, and every step costs the
same as the first. The compounding row is the one that matters: the
append-only loop costs 1.5x more at ten steps, 5.1x more at forty. Short tasks
lose nothing by staying on the old loop. It saves no memory and moves no
benchmark row.**

Shipped as an opt-in per-project toggle, default off (see "What shipped").

The question this page answers: can SKILL.state (arXiv 2608.26263, "SKILL.state:
Scalable Long-Horizon Agent Skills", Badhe/Tiwari/Chung, Google + Purdue,
Aug 2026) be applied to this engine and the local models it runs, and what
would it take?

The answer, measured 2026-08-29/30 on three real installs on this machine, is
YES, and by a wider margin than the paper's own open-weight arm predicted.
All three installs drove a 50-step bounded-state loop at a **1.00 valid-patch
rate on the first try**, with a **perfect final state and zero divergence at
any step**, against the paper's 0.42 for Gemma-4-31B. The append-only baseline
on the identical event stream did not merely score worse: it ran out of
context and stopped, at step 30 to 35 of 50. So the go/no-go condition this
page set for itself ("valid-patch rate with retry at or above roughly 90%") is
met with room to spare, and the feature is NOT blocked on the
grammar-constrained decoding this engine lacks.

Two things that measurement changed about the assessment below, both worth
reading before quoting the paper here. **Grammar-constrained decoding was the
wrong thing to worry about**: not one syntax failure occurred in any state-arm
run across 200 steps. What DID vary, hugely, is whether the model wraps its
JSON in markdown fences (gemma4 did on 92% of replies, gptoss on none), and
that is fixed by the rescue this repo's guardrails already do, not by a
grammar. And **the paper's 68%-premature-overwrite failure class did not
appear at all**: 14 of 14 no-op events per run produced the empty patch.

The scale caveat is stated up front rather than in a footnote: this task is
smaller than the paper's (8 shelves and 14 items against their 500-shelf
inventory), so 1.00 here does not refute their 0.42. It says the failure is a
function of task difficulty, and that a realistic-but-modest schema is well
within what these installs do reliably. See "What the measurement does not
show".

## What it actually saves, in plain English

Three sentences, then the numbers behind them.

**An agent that runs 50 tool steps finishes.** Today's loop does not: it stops
between step 30 and step 35 because the prompt outgrew the context window, and
the failure is a hard HTTP 400 rather than a degraded answer. That is the whole
result in one line. Everything else is secondary.

**It costs 3 to 5x fewer tokens for the same work, and the gap widens the
longer the agent runs.** Tokens here are local compute, not a bill, so this is
directly time and battery.

**Every step costs the same as the first.** Today each step is slower and
fatter than the one before it, because the prompt carries everything that
already happened.

| per 50-step agent run | today (append-only) | with SKILL.state | difference |
|---|---|---|---|
| gemma4 steps completed | 35 of 50 | **50 of 50** | finishes |
| gemma4 tokens | 60,016 (for 35 steps) | **20,253** (for 50) | 3.0x fewer, for 43% more work |
| gemma4 seconds per completed step | 25.1 | **17.8** | 1.4x faster |
| qwen38-27b steps completed | 30 of 50 | **50 of 50** | finishes |
| qwen38-27b tokens | 55,913 (for 30 steps) | **19,522** (for 50) | 2.9x fewer, for 67% more work |
| qwen38-27b seconds per completed step | 122.4 | **31.9** | 3.8x faster |
| gptoss-20b steps completed | 39 of 50 | **50 of 50** | finishes |
| gptoss-20b tokens | 117,539 (for 39 steps) | **29,931** (for 50) | 3.9x fewer, for 28% more work |
| gptoss-20b seconds per completed step | 86.1 | **16.0** | 5.4x faster |

The seconds are wall-clock from one run each on AC power, not a benchmarked
throughput figure, so read them as the shape (constant against growing) rather
than as rows in `docs/BENCHMARKS.md`. The token counts are exact, from the
server's own `usage` field.

**The saving compounds rather than being a flat discount.** Comparing the two
arms over exactly the same first N steps, on gptoss-20b, which is the one
install with per-step token records for both arms:

| steps run | 10 | 20 | 30 | 40 |
|---|---|---|---|---|
| state arm, cumulative tokens | 5,457 | 11,430 | 17,056 | 23,109 |
| append-only, cumulative tokens | 8,116 | 30,546 | 59,210 | 117,539 |
| **how much more the append-only loop costs** | 1.5x | 2.7x | 3.5x | **5.1x** |

That last row is the paper's O(T) against O(T^2) claim in one line: at ten
steps the difference barely matters, and at forty it is the difference between
working and not. A short agent task loses nothing by staying on today's loop.

**What it does NOT save: memory or model quality.** The KV cache, the expert
slot cache and the weights are untouched, so peak footprint is unchanged and
no `docs/BENCHMARKS.md` row moves. This buys horizon and tokens, nothing else.

**Who this is for.** A long-running agent doing many tool calls, which is the
Swift app's project/agent mode. Ordinary chat gets nothing from it and should
stay on `window-fit` (see the last section).

## What the paper claims

SKILL.state replaces append-only conversation history with a compact mutable
JSON execution state. At each step the model receives exactly three things:

- P: an immutable procedural specification (the task instructions),
- Sigma_t: the current execution state, a domain-specific JSON document,
- O_t: the latest environment observation.

The model emits a triple: chain-of-thought reasoning, a structured state
patch (JSON mutations, null meaning delete), and the next action. The runtime
validates the patch against a domain schema authored once per domain, merges
it (Sigma_{t+1} = Sigma_t merged with the patch), executes the action, and
PERMANENTLY DISCARDS the reasoning trace. The prompt therefore stays at
roughly |P| + |Sigma| + |O| (about 1,800 tokens in their experiments)
regardless of step count: cumulative tokens O(T) instead of the O(T^2) of an
append-only loop.

Headline numbers, all Gemini-3-Flash unless stated:

- Warehouse task at T=200 steps: 0.94 accuracy at 122k cumulative tokens,
  against ReAct's 0.74 at 2.61M (21.4x more tokens).
- InterCode CTF (100 tasks): 54.2% pass@1 at 387k tokens vs the best
  stateful baseline's 41.8% at 1.13M.
- tau-bench retail: 58.3% vs 51.7%; airline: 32.4% vs 28.1%.
- Noise robustness: distractor events degrade ReAct 0.68 -> 0.53 while
  SKILL.state holds >= 0.97, because distractors filtered out of a state
  patch never appear in any later prompt.

The control that matters most for this repo: budget-matched baselines at the
SAME ~1,800-token budget. Sliding-window truncation scores 0.18 (critical
early facts evicted) and LLMLingua compression 0.22, against SKILL.state's
0.94. The win comes from the structured representation, not from the prompt
being short. This repo's `window-fit` crate IS that 0.18 baseline: oldest-turn
dropping with the first and last turns pinned
(`crates/window-fit/src/fit.rs:25`), no summarization, no state object.

The caveat that matters most: the open-weight arm. Gemma-4-31B-it scores 0.42
on the same task the proprietary model scores 0.94, with an error taxonomy of
68% premature state overwrites or deletions, 20% schema type comprehension
failures, and 12% JSON syntax errors. The paper's suggested mitigation is
grammar-constrained decoding. Note the taxonomy's shape before treating that
as a fix: constrained decoding can eliminate only the 12% syntax class and
some of the 20% type class. The 68% overwrite class is the model choosing to
destroy state it still needs, which is a semantic failure no grammar reaches.

## Where the question came from

The paper arrived as "does this apply to the engine at all". It reads at first
like an inference technique (the arXiv listing sits among kernel and serving
papers); it is not. It requires no model changes, no fine-tuning, and no engine
primitives beyond "complete a prompt and return text". Everything it adds --
schema, validator, merge, retry -- is host-side orchestration. So the question
decomposed into three: what does this tree already have, what is missing, and
does the local-model caveat bite here specifically? The third is the one only a
measurement could answer, and it is what "Measurement" below settles.

## What exists in this tree, and what does not

Every row is checkable with the named file or a grep; none of it is inferred.

| SKILL.state needs | Status here |
|---|---|
| An agent loop to host it | `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Generation.swift:68-82` rebuilds the full history every step and appends every tool result untruncated as a `<tool_response>` system message. This is the paper's append-only baseline, and the context assembly is about 15 isolated lines. |
| Context management | `crates/window-fit` oldest-turn dropping only (`fit.rs:25`), applied by the ffi's `fit_window` and the CLI chat REPL. The paper's budget-matched control scores this policy class at 0.18. |
| JSON validation of model output | Post-hoc only: `crates/server/src/guardrails.rs:237` calls forge-guardrails' `validate_tool_arguments` against the request's own tool schemas. No JSON-Schema crate anywhere in the workspace (grep `jsonschema\|schemars` over the Cargo.tomls). The Swift app mirrors this with a shallower hand-rolled check (`ForgeGuardrailsEngine.swift`, flat string arguments, no nested objects or arrays). |
| A retry-on-invalid loop | Exists: `run_guarded` (`crates/server/src/guardrails.rs:304`), default budget 1, re-renders the whole prompt with the failed turn plus a nudge appended. Note requests carrying tools are BUFFERED rather than streamed while guardrails are on. |
| Grammar-constrained decoding | Absent, and documented as absent (docs/FORGE_GUARDRAILS.md, DEVIATIONS.md). No vocabulary masking of any kind exists: no logit_bias (the OpenAI field is swept into `extra` and never read), no allowed-token set, no grammar. |
| Structured-output API (`response_format` / `json_schema`) | Absent on both server endpoints. |
| Prefix KV reuse | Implemented (`crates/runtime/src/kv_prefix.rs`, `LogitProducer::try_reuse_prefix`, opted in per session via `set_prefix_reuse`; `--chat` is a caller). It was docs-only when this page was first written on 2026-08-29 and landed on `main` the next day, which is why the section below reads as a projection. See it for what SKILL.state does to the reusable prefix. |

The measurement below says constrained decoding is NOT needed for this, so
what follows is a located seam for a future question rather than a proposal.
If it is ever wanted, it is clean and singular:
`selection::select` (`crates/selection/src/choose.rs:56`) is called from
exactly two decode loops (`crates/runtime/src/raw_completion.rs:312` and the
speculative verify path), and a mask over the working buffer between the
finiteness check and the repetition penalty would be contained. Two traps are
already documented in that crate's AGENTS.md and apply directly: `select` runs
per token at full vocabulary (V=262144 on Gemma 4) OUTSIDE every profiler
bucket, so a mask pass is a throughput change that must be measured with
`tests/host_sampler_cost.rs`'s instruments (precedent: a full sort here cost
18.9 ms/token, invisibly); and the greedy path returns at argmax before the
whole pipeline, so a mask installed after that branch silently does not exist
at temperature 0, which is exactly the temperature an agent runtime would run
at. The paper itself runs temperature 0.0.

## Where it would live

Three candidate homes, in order of fit:

1. **The Swift app's project/agent system.** It already executes tools
   including a shell, already has a step-bounded loop
   (`maxAutonomousSteps`, default 5), and its context assembly is the exact
   thing the paper replaces. A SKILL.state mode would swap the
   rebuild-history block for [system prompt + P][serialized state][latest
   tool result], add a patch parse/validate/merge step after each turn, and
   stop appending tool outputs to the persistent chat. The chat archive
   would keep the full history for the USER's benefit (display, audit); only
   the prompt sent to the engine changes. The paper's "historical
   provenance" limitation does not bite when the host keeps history anyway
   and simply stops feeding it back.
2. **Any client of `turbospark-server`.** The server needs no changes: the
   runtime is above the completions API by design. An external agent harness
   pointing at the OpenAI or Anthropic endpoint can implement the whole loop
   today.
3. **The Rust engine itself: nothing, for a minimal version.** There is no
   engine primitive in the paper. The only engine-side items that would ever
   be justified are the constrained-decoding seam above and a
   `response_format`-style server field, and both are separate decisions
   with their own costs, not prerequisites.

## Interaction with prefix KV reuse

Prefix reuse is a longest-common-prefix mechanism keyed on fed token ids
(`crates/runtime/src/kv_prefix.rs`, `crates/runtime/AGENTS.md` Gotcha 30). It
landed on `main` in `a7274a3`, one day after this page first described it as
unimplemented; nothing below was re-measured against it, so read this section
as the arithmetic of the two mechanisms rather than as a reading of the
shipped one. A SKILL.state prompt is
[fixed P][mutating Sigma][fresh O], so the LCP ends at the first byte where
the serialized state differs from the previous step: only P would ever be
reused, which is also the paper's own stated limitation of prefix caching
under this scheme. Do not read that as a conflict. The technique's whole point
is that the prompt is bounded at ~1.8k tokens, so full prefill stays cheap and
flat, where the append-only loop's prefill grows without bound and is exactly
what prefix reuse exists to rescue. The two are alternative answers to the
same cost, and a canonical serialization of the state (stable key order, so
unchanged fields do not spuriously move) would maximize what little LCP there
is for free.

## The local-model risk, as feared and as measured

This engine exists to run open-weight models locally, and the paper's one
open-weight data point is a 0.42 against the proprietary 0.94. That was the
feasibility risk: not whether the runtime can be built (it can, cheaply), but
whether the models this engine actually serves can drive it. The measurement
above answers it per failure class, and the answer is different for each:

- **12% JSON syntax errors: did not occur** in any state-arm run (200 steps
  across four runs). The real formatting hazard turned out to be markdown
  fences, which the guardrails' existing rescue handles completely. Strict,
  unrescued validity varies from 0.08 to 1.00 by dialect, so the rescue is
  load-bearing rather than a nicety.
- **20% schema type failures: did not occur.** Note the schema here is two
  keys, well under what the paper's five-field schemas ask for, so this is
  the class most likely to reappear if the schema grows.
- **68% premature overwrites: did not occur.** 14 of 14 no-op events per run
  produced the empty patch, on every install. This was the class no grammar
  could have fixed, and it is the one this measurement most clearly clears.

Host-side defenses remain available if a larger schema brings the overwrite
class back, and they are deterministic runtime policy rather than model
behavior: refuse patches that delete or overwrite fields the current
observation did not mention, require explicit null-tombstones for deletion, or
keep an undo journal so a later contradiction can restore a clobbered field.
None is needed at this task size.

## The instrument

`scripts/skill_state_probe.py`, stdlib only, no dependency and nothing
installed. It generates a deterministic warehouse task (8 shelves, 14 items,
50 events per run) and drives it through a running `turbospark-server` in two
arms over the SAME event stream, so the only variable is what the prompt
carries:

- `state`: `[fixed spec P][current state Sigma][one event]` -> a JSON Merge
  Patch (RFC 7386), validated against the domain schema, then merged.
- `history`: `[fixed spec P][every event and reply so far]` -> the complete
  state. This is the paper's strongest "stateful" baseline and is also what
  `swift/TurboSparkApp`'s loop does today.

Both arms emit a state at every step, so the scorer is identical for both.

14 of the 50 events are NO-OPS by construction (7 `AUDIT` lines that restate
what is already true, 7 irrelevant `NOTICE` lines). Those exist to probe the
paper's dominant failure class directly: the correct patch for each is `{}`,
and anything else is a premature overwrite.

**The sanity invariant, because a harness that cannot fail cannot measure.**
`--selfcheck` runs three control agents with NO server at all and asserts the
scorer discriminates: an `oracle` that emits the correct patch must score
1.00 and be exact, while `noop` (always `{}`) and `clobber` (a schema-VALID
patch that wipes every shelf) must both score below it. It also runs 21 direct
cases over the validator, the merge and the JSON extractor. Ground-truth
conservation is asserted inside the generator on every step.

That self-check was mutation-tested, five mutations, each asserted to apply
exactly once. Four reddened only their own case. The fifth, deleting merge's
null-deletion branch, SURVIVED, which per this repo's rule is a missing test
rather than a weak one: no control agent emits a null-deletion, though the
spec offers it to the model and the validator accepts it. `MERGE_CASES` closes
that, and the mutation reddens now.

```sh
python3 scripts/skill_state_probe.py --selfcheck          # offline, seconds
./target/release/turbospark-server --model ~/models/gemma4.gturbo \
    --port 8123 --guardrails off --max-context 4096
python3 scripts/skill_state_probe.py --arm state --steps 50 --port 8123 \
    --label gemma4 --out /tmp/skill-state/gemma4-state.json
```

## Measurement

Measured 2026-08-29/30 on the M4 Max, AC power, one seed (20260829) unless
noted. `--max-context 4096` and `--guardrails off` PINNED on every install, so
the model is the only variable (AGENTS.md Gotcha 35: the harness measuring a
knob must not let it sense the machine). Requests are `temperature: 0.0`,
which is this engine's exact argmax path, so runs are reproducible.
`qwen36` from the original plan is not installed on this machine; `qwen38-27b`
(dense, ChatML) is the substitute. Raw artifacts in
`~/models/skill-state-probe/`, deliberately not `/tmp`.

| install | arm | valid 1st | strict 1st | valid w/ retry | final acc | mean acc | exact | first divergence | tokens |
|---|---|---|---|---|---|---|---|---|---|
| gemma4 | state | **1.00** | 0.08 | **1.00** | **1.00** | 1.00 | yes | never | 20,253 |
| gemma4 | history | 0.70 | 0.70 | 0.70 | 0.43 | 0.91 | no | step 35 | 60,016 |
| qwen38-27b | state | **1.00** | 0.56 | **1.00** | **1.00** | 1.00 | yes | never | 19,522 |
| qwen38-27b | history | 0.60 | 0.60 | 0.60 | 0.36 | 0.86 | no | step 30 | 55,913 |
| gptoss-20b | state | **1.00** | 1.00 | **1.00** | **1.00** | 1.00 | yes | never | 29,931 |
| gptoss-20b | history | 0.52 | 0.50 | 0.78 | 0.57 | 0.95 | no | step 31 | 117,539 |
| gptoss-20b (seed 7) | state | **1.00** | 1.00 | **1.00** | **1.00** | 1.00 | yes | never | 29,236 |

`valid 1st` is after fence-stripping rescue, matching what
`crates/server/src/guardrails.rs` already does in production; `strict 1st` is
the whole reply parsing as JSON with no rescue at all.

**The prompt growth is the paper's O(T) against O(T^2), measured.** Prompt
tokens per step on gptoss-20b:

| step | 0 | 6 | 12 | 18 | 24 | 30 | 36 | range |
|---|---|---|---|---|---|---|---|---|
| state | 302 | 339 | 358 | 366 | 380 | 389 | 400 | 302-427 (1.4x) |
| history | 212 | 562 | 993 | 1425 | 1922 | 2540 | 3227 | 212-3582 (16.9x) |

**The baseline does not degrade, it stops.** Every history arm died the same
way, with the server refusing the request by name:

```
HTTP 400: prompt (3630) + max_new (512) exceeds max_context (4096)
```

Up to that point it was accurate (gemma4 read 1.00 at step 34, the last step
it completed). So the append-only arm is not wrong, it is BOUNDED, and its
`mean acc` of 0.86 to 0.95 describes only the prefix it survived. What a real
host does past that point is drop turns, which is `crates/window-fit`, which
is the policy class the paper's budget-matched control scores at 0.18.

**Three findings from the run that were not predictable from the paper.**

1. **Markdown fences, not grammars, are the real formatting hazard**, and they
   are entirely dialect-dependent: gemma4 emitted bare JSON on 8% of replies,
   qwen38 on 56%, gptoss on 100%. Rescue made all three 1.00. A production
   implementation must strip fences; it does not need constrained decoding.
2. **The no-op events were handled perfectly.** 14 of 14 per run produced the
   empty patch on every install. The paper's 68% premature-overwrite class did
   not occur once.
3. **The append-only arm degrades its own OUTPUT, not just its context.**
   gptoss's history arm hit 15 JSON syntax failures against zero in its state
   arm, and every one was a truncation at exactly `completion=512`, this
   probe's `--max-tokens` cap. Restating a growing state every step needs
   ever-longer replies. Note this specific number is sensitive to that cap and
   a larger one would trade output tokens for validity; the context wall
   underneath it is not sensitive to the cap in any interesting way.

## What the measurement does not show

- **It does not refute the paper's 0.42.** The task here is much smaller (8
  shelves, 14 items, 50 steps; theirs is a 500-shelf inventory at T=200) and
  the schema is two keys. The honest reading is that patch-emission is
  reliable on these installs at THIS difficulty, and the paper's failure modes
  are a function of schema and horizon size, both of which this probe could be
  scaled up to find.
- **It is one seed for gemma4 and qwen38.** gptoss was run at a second seed
  and reproduced exactly (1.00 across the board), which is what argues the
  first seed was not a lucky draw, but the other two installs have one run
  each.
- **It measures patch VALIDITY and state accuracy, not task success.** There
  is no tool execution, no action selection, and no environment that pushes
  back. A real agent loop adds all three.
- **The two arms differ in a second way beyond prompt content.** The state arm
  emits a small patch and the history arm restates the whole document, so the
  history arm is also the more token-hungry OUTPUT shape. That is intrinsic to
  the baseline rather than a harness artifact, but it means "history is worse"
  bundles two causes.

## The decision

The go/no-go condition is met: 1.00 valid-patch rate on the first try on all
three installs, against a bar of roughly 0.90 with retry. **A SKILL.state mode
in the Swift agent loop is worth building, and it is not blocked on any engine
change.** Concretely, what the measurement licenses:

- Build it host-side, in `AppModel+Generation.swift`'s context assembly.
- Include fence-stripping rescue and schema validation with a one-retry
  budget. Both already exist in this repo, in Rust
  (`crates/server/src/guardrails.rs`) and in Swift
  (`ForgeGuardrailsEngine.swift`); the Swift validator is the shallower of the
  two and would need nested-object support for a state schema.
- Do NOT build constrained decoding for this. The seam and its traps stay
  documented above for whenever a harder schema justifies it, but nothing in
  this measurement asks for it.

What would reverse this: a schema materially larger than two keys, or a
horizon past a few hundred steps, showing the paper's overwrite class
appearing on these installs. Scaling the probe's shelf and item counts is the
cheap way to look, and the harness takes them as constants at the top of the
file.

## What shipped

Implemented 2026-08-30 in the Swift app, opt-in per project and OFF by
default, so the append-only path is byte-identical when the toggle is unset.

- `State/AppSkillState.swift`: the state document, an RFC 7386 merge, the
  schema, the validator, and the fence-stripping extractor. Semantics mirror
  `scripts/skill_state_probe.py` deliberately, since that script is the
  evidence.
- `State/AppModel+SkillState.swift`: the bounded prompt and the patch
  application.
- `AppModel+Generation.swift`: the one branch, at the context assembly, plus
  the patch merge on turn finish.
- `AppProject.skillStateEnabled` and `AppChat.skillState`, both with tolerant
  decoding (swift/AGENTS.md Gotcha 13), and a toggle in project settings.

**The schema is generic rather than per domain**, which is a departure from
the paper and a considered one. The paper authors a schema per domain and
names "no fixed schema known in advance" as its first limitation, which a
general coding assistant arguably is. The five fields (`goal`, `files`,
`facts`, `commands`, `next`) are close in shape to the paper's own five-field
InterCode CTF schema. `AppSkillStateSchema` is the single source for both the
prompt text and the validator, so the two cannot drift.

**The growth from two keys to five was verified rather than assumed**, because
this page names schema comprehension as the class most likely to reappear as a
schema grows. `AppSkillStateRealModelTests` drives a six-step coding run
through a real server using the SHIPPED protocol text and the SHIPPED
validator, so the app is its own oracle and no second spelling of the schema
exists to rot. Against gemma4: 6 of 6 patches valid, state accumulated
correctly, and the deliberately irrelevant observation changed nothing.

```sh
./target/release/turbospark-server --model ~/models/gemma4.gturbo \
    --port 8123 --max-context 4096
cd swift/TurboSparkApp && TURBOSPARK_SKILL_STATE_SERVER=http://127.0.0.1:8123 \
    swift test --filter AppSkillStateRealModelTests
```

**One quality wrinkle that run turned up, recorded because it is not a
validity failure and no assertion catches it.** gemma4 put everything into
`facts` and used neither `files` nor `commands`, and it left `next` holding a
step it had already done. So the state stays CORRECT and drifts toward being
one flat list, which is a weaker structure than the schema offers. Nothing
here depends on it yet. If it matters later, the lever is the field
documentation in `AppSkillStateSchema.fields` (one string each), not the
validator, which cannot see the difference between a good decomposition and
a lazy one.

## Two questions this attracts

**"Do I have to write a schema?" No.** One generic schema ships hardcoded in
`AppSkillStateSchema` and covers general coding-agent work; turning the
project toggle on is the whole setup. A PER-PROJECT authored schema is the
paper's own model and is deliberately NOT built: it would be unusable until
someone wrote one, so the default experience would be unchanged. Build it only
when a specialized agent turns up that the five generic fields do not fit, and
note that `AppSkillStateSchema` is the single source for both the prompt text
and the validator, so a second schema path means keeping that property twice.

**"Would steering vectors help here?" No, and the reason is worth stating so
it is not re-derived.** This is REASONING rather than a measurement, which is
the weaker kind of claim (Gotcha 62's warning about a derived convention that
reads plausibly and is wrong), so it is stated with its argument attached
rather than as a result.

Steering (`docs/OBLITERATION.md`) is an edit to the RESIDUAL STREAM: a
direction extracted offline as a difference of means over paired prompt sets,
applied per layer at runtime. That page is explicit that it "changes behaviour
and costs throughput" and is representation engineering. A schema is a
contract about the SHAPE OF EMITTED TEXT, enforced by prompt wording plus a
deterministic validator on the way back. Nothing about it lives in activation
space.

The deeper mismatch is the shape of the two problems. A steering vector is one
direction pushing globally on every token of a generation. Schema compliance is
discrete and positional: this brace closes here, this key is spelled `facts`,
this value is a list and not a string. That is the problem shape
grammar-constrained decoding addresses by masking the vocabulary per token, and
it is not the shape a single direction addresses. Steering would be the wrong
instrument even if it were free, and it is not: 1.72% of decode with all layers
steered, and it is wired for five of eight families.

**And there is no problem here to solve.** Compliance measured 1.00 on three
installs across 200 steps and 6 of 6 on the shipped schema, so a mechanism
aimed at it would be optimizing a number already at its ceiling.

The one place an argument for steering could be constructed is the
decomposition wrinkle recorded above: the model favours one flat `facts` list
over the structured fields, which the validator structurally cannot see. That
is a quality-of-decomposition question rather than a compliance one. The cheap
lever is the one-line field documentation in `AppSkillStateSchema.fields`,
which is what the model is actually reading; try that and measure with
`AppSkillStateRealModelTests` before reaching for anything in activation space.

## Verifying a change to this

```sh
cd swift/TurboSparkApp && swift test --filter AppSkillStateTests   # 21, offline
```

Those cases mirror the probe's `--selfcheck` and were mutation-checked five
ways, each mutation asserted to apply exactly once and each reddening only its
own cases. Two traps worth keeping if this code is touched:

- **The patch merges BEFORE tool calls are parsed.** The patch is JSON in the
  same reply as a possible tool call, so parsing calls first lets the tool
  parser see it and take it for one.
- **An invalid patch is DROPPED, never partially applied.** A bad merge
  silently corrupts every step after it; a dropped one loses one step's
  bookkeeping and surfaces on `skillStateLastError`.

## What this page is not

- Not a reproduction of the paper. The numbers in "What the paper claims" are
  theirs, on their models, harness and task scale. The numbers under
  "Measurement" are this repo's, on a smaller task, and the two are not
  directly comparable. See "What the measurement does not show".
- Not a proposal to change the engine. The measured version touches only the
  Swift host; the engine-side items (constrained decoding, response_format)
  are separate decisions this page only locates, and the measurement
  affirmatively does NOT ask for either.
- Not applicable to plain chat. SKILL.state is for long-horizon TOOL loops
  with a schematizable state; ordinary conversation has no fixed schema, which
  is the paper's own first stated limitation, so `window-fit` remains the
  right mechanism for chat.
