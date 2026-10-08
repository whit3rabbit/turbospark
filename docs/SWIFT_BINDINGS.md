# Swift bindings: driving the engine from a native app

`crates/ffi` exposes the inference engine as a C ABI, and
`swift/TurboSpark` wraps that in an idiomatic Swift package. A SwiftUI app
opens a `.gturbo` install, streams tokens, stops mid-generation, installs
models, and reads engine telemetry, all in-process.

`swift/TurboSparkApp` is a full SwiftUI chat app that exercises every one of those.

---

## Before anything else: two things that will bite

**The Swift package does not build until the Rust library exists.** A SwiftPM
target may not reach outside its own directory, so `scripts/swift-lib.sh`
builds `crates/ffi` and copies the archive plus the canonical header into
`swift/TurboSpark/Sources/CTurboSpark/`. Skip it and `swift build` fails with
a missing-header error that says nothing about the real cause.

```bash
make swift-lib
```

**Every consumer of the package has to repeat one linker flag.** A library
search path in `unsafeFlags` is resolved against the root of the package
being *built*, not the package that declared it, so `TurboSpark`'s own
`-LSources/CTurboSpark` is correct when its tests link and wrong for
everybody else. `swift/TurboSparkApp/Package.swift` shows the shape:

```swift
.executableTarget(
    name: "MyApp",
    dependencies: [.product(name: "TurboSpark", package: "TurboSpark")],
    linkerSettings: [.unsafeFlags(["-L../TurboSpark/Sources/CTurboSpark"])]
)
```

**Header and archive can come from different builds, and the package now says
so.** Both are gitignored copies staged by `make swift-lib`, so a stale one of
either is easy to end up with. The header carries `TS_ABI_VERSION`, the
library reports its own through `ts_abi_version()`, and the package compares
the two once per process before the first session or server opens, throwing a
`TurboSparkError` whose message names both numbers and the fix. Read them
yourself with `TurboSparkRuntime.headerABIVersion`, `.libraryABIVersion`, and
`TurboSparkRuntime.buildInfo()` (ABI revision, crate version, and whether it is
an unoptimized build whose timings must not be quoted). The revision is bumped
when a wire shape or an option's meaning changes in a way an older header
could misread; a purely additive export does not need one.

This is a SwiftPM limitation rather than a defect here. A package published
for outside consumption should ship an `.xcframework` binary target, which
resolves paths for its consumers properly; a two-package repository does not
need the packaging step.

---

## Should you use this, or the server?

`turbospark-server` already speaks OpenAI `/v1/chat/completions` and
Anthropic `/v1/messages` with SSE streaming. If your app can talk HTTP to a
loopback port, that path exists today, needs none of this, and is covered by
its own tests.

**You can also have both, and that is usually the answer for a GUI.**
`TurboSparkServer` runs the same router in your own process over models you
already have open, so other apps on the machine can reach them without a
second copy of the weights (see [The in-process server](#the-in-process-server)).

Reach for the bindings when you want:

- **One process.** No sidecar binary to ship, sandbox, notarize and supervise.
- **Engine telemetry.** Phase counters, resolved context window, resolved
  expert-cache slots, peak footprint. None of that crosses an HTTP boundary.
- **Model management in-app.** Browse the catalog, probe an arbitrary
  Hugging Face repository, install with byte progress.
- **Cancellation that is not a dropped connection.**

Tool calling is the one thing the server has and the bindings do not; see
[Not supported](#not-supported).

---

## Quick start

```bash
make swift-lib          # build the staticlib, stage the header
make swift-demo         # run the demo chat app
```

The whole API, in one function:

```swift
import TurboSpark

let session = try await TurboSparkSession(modelPath: "~/models/gemma4.gturbo")
print("open: \(session.info.family) at \(session.info.maxContext) context")

var options = GenerateOptions()
options.maxNewTokens = 400

for try await event in session.generate(
    [ChatMessage(role: .user, content: "Explain how coastal wetlands reduce flood damage.")],
    options: options
) {
    switch event {
    case .prefill(let done, let total):
        print("reading prompt: \(done)/\(total)")
    case .content(let text):
        print(text, terminator: "")
    case .reasoning(let text):
        print("[thinking] \(text)", terminator: "")
    case .toolCall(let call):
        print("[tool] \(call.name)\(call.argumentsJSON)")
    case .stopped(let reason, let newTokens, _):
        print("\n[stopped: \(reason) after \(newTokens) tokens]")
    case .finished(let result):
        print("\n\(result.newTokens) tokens, \(result.stopReason)")
    }
}
```

A leading `~` is expanded for you. `modelPath` also takes a
`turbospark-model` alias, and an existing directory always wins over an
alias, so a bare name cannot silently open a different model than the one
you named.

---

## Sessions

### Opening

Opening maps gigabytes and compiles Metal pipelines. **Open once and keep the
session**; it runs off the calling thread, so `await`ing it from a SwiftUI
view is fine.

```swift
var options = OpenOptions()
options.maxContext = .fixed(8192)          // or .auto, the default
options.expertCacheSlots = .fixed(16)      // or .auto: 8, 16, 24, 32
options.powerProfile = .efficiency         // or nil, see below
options.maxTokensPerSec = 30
options.loadGuard = .balanced              // or .off, .relaxed (default), .strict,
                                           // .custom(bytes)
options.minAutoContext = 8192              // 0, the default, imposes no floor
options.speculation = .auto                // or .off, .block(2)
options.speculativeDrafter = .auto         // or .mtp, .dflash
options.steering = "/path/to/vector.gguf"  // or nil (disabled)
options.steeringMode = .ablate             // or .add, .clamp, .renorm
options.steeringScale = 0.5                // default 1.0; 0.0 is identity
options.steeringLayers = "20:45"           // 0-based inclusive layer range
options.steeringTarget = 0.0               // for .clamp mode (default 0.0)
options.steeringGate = 0.0                 // activation threshold >= 0
options.kvBits = .threePointFive           // or .off (default), .two, .three, .four

let session = try await TurboSparkSession(modelPath: "gemma4", options: options)
```

Everything defaults to automatic, which is what a GUI should want. Five
defaults are worth understanding rather than accepting:

**`expertCacheSlots: .auto` climbs, never falls.** It picks the largest
allowed count whose working set fits available headroom, floored at the
shipped default of 16. On a 36 GB machine with a 13 GB install it resolves to
32, which buys roughly 16% more decode for about 1.5 GB of footprint. A
machine without the headroom gets exactly the 16 it always got, so the
feature cannot make anyone slower.

**`powerProfile: nil` asks the OS**, and Low Power Mode selects `efficiency`.
That is right for a user-facing app and wrong for anything measuring: name a
profile explicitly if you are benchmarking, or an efficiency cap will quietly
become part of your result.

**`speculation: .auto` drafts ahead only if the install can, and only on
temperature-0 turns.** The drafter's state is allocated at open, so this is a
session setting and not a per-turn one -- but acceptance is
`argmax(target) == proposal`, exact only at temperature 0, so a sampled turn
takes the sequential loop whatever the session resolved. Since
`GenerateOptions.temperature` defaults to 0.2, **an app that never sends 0
never speculates**, and that fallback is silent by design (a per-turn warning
would fire on the normal case). Read `info.speculation` for the
session-level answer, once.

`.auto` also declines a DFlash2 drafter it FINDS, and says so in
`info.speculation.reason`. That asymmetry is measured rather than stylistic:
the checkpoint's own MTP head pays 1.44-1.66x, while DFlash2 at block 2 reads
1.33x on code and 1.47x on math against 0.90x on PROSE, with its power arm at
+17.4% J/token. Ask for it with `speculativeDrafter = .dflash` if your
workload is code-shaped.
`.block(n)` is a promise rather than a preference: an install that cannot
serve it throws from `init` rather than opening quietly without it.

**`loadGuard: .relaxed` is the default and is what shipped before the option
existed.** It reserves 4 GiB for the rest of the machine and spends a quarter
of what is left. Every published footprint figure for this engine was
measured under it, so changing the tier makes your numbers incomparable with
those -- which is the point of the knob, but worth knowing before you file a
bug about a peak that moved.

`.off` reserves nothing and, more importantly, declines to REFUSE: a window
too large for the machine becomes the Metal allocation failure you asked for
rather than an error with the arithmetic in it. `.custom(bytes)` caps what
the engine ALLOCATES (slot cache plus KV), not the install's size -- a large
model streaming its experts from disk is what this engine is for, and a cap
read against the install would refuse a 13 GB model on a 16 GB machine that
runs it fine.

**If you also call `TurboSparkCatalog.recommend`, pass it the SAME guard.**
The ranking and the loader's refusal share one memory budget by construction,
which is what makes a recommendation worth showing; recommending under
`.relaxed` while opening under `.strict` promises a fit the loader then
refuses, in the one place a user cannot see the two disagree.

**`minAutoContext` constrains automatic sizing only.** It refuses to open
when `maxContext: .auto` resolves below it, and says nothing about an
explicit `.fixed(2048)` -- a caller naming a number has decided how to spend
their own machine. The error names which bound was binding (memory, the
checkpoint's trained context, or an install that declares none), because
loosening the guard, lowering the floor and picking a different checkpoint
are three different fixes and only one of them helps.

**`steering` applies a control vector at open.** The vector file is parsed
and verified against the model architecture before any memory is mapped, and
the session verifies that the architecture family supports directional
steering. Steering modifiers (`steeringMode`, `steeringScale`, `steeringLayers`,
`steeringTarget`, `steeringGate`) require a `steering` path and throw if passed
alone.

**`kvBits: .off` is the default and is what every release before this option
existed produced byte for byte.** TurboQuant KV-cache quantization
(`docs/TRUBOQUANT.md`) reduces the per-token memory a session's KV cache
costs, at `.two`/`.three`/`.threePointFive`/`.four` bits per coordinate.
Unlike speculation there is no auto-detect: an unsupported family or
`head_dim` throws from `init` by name rather than silently opening at FP16,
so a caller asking for quantization either gets it or learns why not --
never a session that quietly measures the wrong footprint.

### Reading what you actually got

```swift
let info = session.info
info.maxContext         // the RESOLVED window
info.expertCacheSlots   // the RESOLVED slot count
info.trainedContext     // the checkpoint's own, or nil
info.pastTrainedContext // true when the window exceeds it
info.family             // "gemma4", "qwen36", "llama", ...
info.vocabSize          // token count in vocabulary
info.dialect            // chat template dialect ("harmony", "qwen", ...)
info.reasoningSupport   // .level | .toggleOnly | .none -- what KIND of control
info.reasoningEfforts   // [.off, .low, .medium, .xhigh] -- what to put IN it
info.steering.active    // true when a control vector is active
info.steering.supported // whether one COULD be -- gate a control on THIS
info.steering.reason    // why `supported` is false, in the open's own words
info.steering.mode      // "ablate", "add", "clamp", "renorm" or nil
info.steering.scale     // active scale multiplier or nil
info.steering.summary   // human-readable one-line description or nil
info.toolCalling.native // does this checkpoint's own markup frame tool calls
info.toolCalling.reason // why not, naming the dialect
info.speculation.block  // the RESOLVED block, or nil when off
info.speculation.drafter// .mtp | .dflash, non-nil exactly when block is
info.speculation.reason // why it is off, when you might expect otherwise
info.specialTokens.bosId        // e.g. 1 or nil
info.specialTokens.eosId        // e.g. 2 or nil
info.specialTokens.endOfTurnId  // e.g. 151645 or nil
info.specialTokens.stopTokenIds // [151643, 151645]
info.specialTokens.thinkStartId // e.g. 151648 or nil
info.specialTokens.thinkEndId   // e.g. 151649 or nil
info.kvBits                     // "off", "2", "3", "4", or "3.5 (K3/V4)"
```


**`steering.supported` AND `steering.active` ARE DIFFERENT QUESTIONS.**
`active` says a vector is running; `supported` says one COULD be. An open is
REFUSED on a family whose decode flow does not dispatch the edit, so a UI
offering the knob there offers one whose only outcome is a failed load -- and
an unsteered session on a family that steers answers
`active: false, supported: true`, so gating on `active` would disable the
control for every model that is not already steering.

**`toolCalling.native == false` IS NOT "TOOLS DO NOT WORK", AND HIDING A TOOL
CONTROL ON IT IS BACKWARDS.** It says the checkpoint's own framing hands no
call over, which is exactly the case a guardrail rescue is for
(`docs/FORGE_GUARDRAILS.md` section 0c). Report it; do not gate on it. Three
of seven dialects answer `false`, and one of them (Muse Glimmer) frames calls
in markup this engine has no parser for, so they reach a caller as reasoning.

**Read these rather than what you asked for.** Under automatic sizing you
asked for nothing, and the KV cache has already been allocated at the
resolved window. Neither a throughput nor a footprint number is readable
without the slot count beside it.

`pastTrainedContext` is reported rather than refused on purpose: RoPE
extrapolates past the trained context rather than failing, some checkpoints
carry scaling meant to exceed it, and an install written before that field
existed declares none at all. Surface it as a quality warning.

### Generating

```swift
var options = GenerateOptions()
options.maxNewTokens = 512        // clamped to what the context leaves
options.temperature = 0.2
options.topK = 64
options.topP = 0.95
options.repetitionPenalty = 1.0
options.minP = 0.0                // [0, 1); 0 disables min-p truncation
options.presencePenalty = 0.0     // [-2, 2], once per distinct generated token
options.frequencyPenalty = 0.0    // [-2, 2], scaled by generated count
options.seed = 20260721           // nil for nondeterministic
options.stop = ["\n\n---"]
options.stopTokens = [151643, 151645] // numerical stop token IDs
options.reasoning = .off
```

`minP`, `presencePenalty` and `frequencyPenalty` default to 0, which is the
identity for each. The two penalties count the GENERATED suffix only, never
the prompt, and an out-of-range value is refused by name before decoding
starts rather than clamped.

The defaults are the CLI's, so sending nothing gives what
`turbospark-check` gives with no flags. `maxNewTokens` is clamped rather than
refused when the conversation is long, so a full context generates into
whatever room is left instead of failing.

**Two prefill optimizations are on unconditionally, and neither takes a
flag.** Every `generate` call still renders the WHOLE conversation and sends
it whole, exactly as documented above; what changed (2026-09-01) is how much
of that render actually has to be re-computed.

A session continues from the previous turn's KV wherever the new render
shares a prefix with the old one (`runtime::kv_prefix`, the same mechanism
`crates/cli`'s `--chat` REPL opts into). Enabled at `open` because a GUI
session is multi-turn by construction, the same reason `--chat` is. Falls
back to a full prefill -- never an error -- on a session's first turn, on a
render that diverged anywhere (an edited history, a different reasoning
level), or on a family this cannot help (recurrent state, a sliding-window
ring past its slack). `GenerationResult.reusedPrefixTokens` reports how much
of `promptTokens` was skipped; read it if a status panel wants to show it,
but there is nothing to configure.

Prefill also runs CHUNKED rather than sequentially whenever the open
install's family supports it (Gemma 4 and the dense half of `llama`, the
same predicate `crates/server`'s automatic dispatch checks) -- a caller
cannot tell from the API surface which prefill shape ran; both produce
byte-identical tokens. Skipped when the turn carries an image, since the
vision-capable family's chunked driver refuses an open image prompt by name
rather than composing the two on an unreachable path.

**`GenerationResult` degrades instead of failing on a newer engine.** An
unrecognised `stopReason` decodes as `.unknown` and keeps the turn's content
and timings; an unrecognised `FitVerdict` decodes as `.unknown` and one
`ServerImageEvent` kind this binding predates is skipped without dropping the
events beside it. `GenerationResult.toolCalls` carries the parsed calls of the
turn (see "Not supported" for why it is empty today).

### Cancelling

```swift
session.cancel()
```

**Safe from any thread, and it never blocks.** This is the single most
load-bearing property of the whole binding. Generation holds the engine lock
for an entire turn, so the cancel flag deliberately lives *outside* that
lock, both in the C layer and in the Swift wrapper (which is why
`TurboSparkSession` is a class with a serial queue rather than an `actor`).

Put the flag behind the lock and `cancel()` waits for the generation it is
trying to stop. That does not fail, it *hangs*, and a user experiences it as
a frozen window rather than as a bug worth reporting.

**Cancelling is not an error.** The turn finishes normally:

```swift
case .finished(let result):
    if result.stopReason == .cancelled {
        // `result.content` holds everything generated so far and is valid.
        // The KV cache describes itself honestly, so the next turn continues
        // from here.
    }
```

Cancellation is last in precedence. A run that would have stopped on its own
terms on the same token reports why it *really* stopped, so a Stop pressed as
the model finishes does not relabel a complete turn as a truncated one.

Cancelling the consuming `Task` also cancels the generation, so
`for try await` inside a SwiftUI `.task` stops the model when the view goes
away.

### Reasoning

```swift
options.reasoning = .medium
```

Reasoning arrives as its own event, already separated from the reply:

```swift
case .content(let text):    reply += text        // THIS is the assistant turn
case .reasoning(let text):  thinking += text     // display only
```

**Do not feed `.reasoning` back as conversation history.** Harmony's own
convention drops prior-turn analysis and Qwen's template drops prior-turn
`<think>` blocks, so replaying it sends the model something it was never
trained to read.

**Two more events ride the same stream.** `.stopped` arrives when the model
stops, on the stream itself and just before `.finished`, with the stop
reason spelled as `GenerationResult.stopReason` spells it plus both token
counts; a UI can finalize from it without waiting for the result decode.
`.toolCall` carries one parsed invocation (`id`, `name`, `argumentsJSON`),
and `GenerationResult.toolCalls` carries them all in the result. NEITHER
CAN FIRE YET: a tool call is parsed only when the caller offered the tool
by name, and no `GenerateOptions` field offers tools -- tool calling stays
on the server surface. The events are wired so that growing the binding to
offer tools is an options change, not a stream redesign.

**The accepted levels are the checkpoint's, not this library's.** Qwen 3.8
rejects `.high` and its top setting is `.xhigh`, while gpt-oss and Muse
Glimmer accept `.high` and have no `.xhigh`. A level a template rejects throws
an error naming it, rather than being silently dropped.

**So build the picker from `info.reasoningEfforts`, never from
`Reasoning.allCases` and never from the family.** The engine probes that set
at open by rendering a one-message conversation at each spelling, so it is the
checkpoint's own answer and a new release needs no code change here:

```swift
Picker("Thinking", selection: $level) {
    ForEach(session.info.reasoningEfforts) { Text($0.label).tag($0) }
}
.disabled(session.info.reasoningSupport == .none)
```

Levels that render the same prompt are already collapsed, so a `.toggleOnly`
checkpoint reports exactly two entries. `reasoningSupport` says what KIND of
control is meaningful:

| value | meaning | what a UI should do |
|---|---|---|
| `.level` | the template takes an effort level | list `reasoningEfforts` |
| `.toggleOnly` | thinking turns on, the level is dropped | present the one on-level as a switch, labelled "On" rather than by its spelling |
| `.none` | no reasoning knob at all | disable the control, asking throws |

Two traps. A `.toggleOnly` checkpoint's on-level is `.low` BY POSITION and is
not a label. And there is no answer at all before a session exists, because
the set is read off the template at open: gate the control on having one
rather than guessing from the model's family.

### Tool calling

Offer functions with `GenerateOptions.tools`, and the model can call them:

```swift
var options = GenerateOptions()
options.temperature = 0
options.tools = [
    try ToolSpec(name: "get_weather", description: "Current weather for a city",
                 parametersJSON: #"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#)
]
var history = [ChatMessage.user("Weather in Oslo?")]

for try await event in session.generate(history, options: options) {
    switch event {
    case .toolCall(let call): print(call.name, call.argumentsJSON)   // streamed as parsed
    case .finished(let result):
        guard result.stopReason == .toolCalls else { break }
        // Run the tools, then answer them and generate again:
        history.append(.assistant(result.content, toolCalls: result.toolCalls))
        for call in result.toolCalls {
            history.append(.tool(try run(call), toolCallId: call.id, name: call.name))
        }
    default: break
    }
}
```

What the engine does and does not do:

- **Parsing** is the checkpoint's NATIVE markup only (`info.toolCalling.native`)
  and only for a function offered in `tools`. A call to anything else is plain
  text, so a model that invents a function does not get it reported.
- **`stopReason` is `.toolCalls` for any turn that produced a parsed call,**
  whichever token closed it. ChatML and DeepSeek end such a turn in
  `endOfTurn`; `crates/server` applies the same rule, and a host's tool loop
  keys on this one value. `.cancelled` stays authoritative: a Stop press never
  reads as "run the tool".
- **Ids are `toolu_<n>` in emission order.** Keep them: the Gemma template
  resolves a tool turn's function name by matching `toolCallId` against the
  preceding assistant message's calls, and renders "unknown" otherwise.
- **Rendering** goes through the checkpoint's own `chat_template.jinja`, the
  only renderer that can express tools, so a checkpoint that ships none
  refuses the turn by name. A misspelled offer (an empty or repeated name, a
  `parameters` that is not a JSON object) is refused before any decoding.
- **Tool definitions count toward the prompt**, and `countTokens` and
  `fitWindow` do not see them. Leave extra room when budgeting.
- **The binding runs nothing.** Executing a tool, and the loop around it, is
  the host's job. `TurboSparkApp` still uses its own parser and guardrails
  (`Tools/Core/*`, `Tools/Guardrails/*`) because they also cover checkpoints
  whose markup is not native, and moving the app's agent loop onto this
  surface is a behaviour change that needs the real-model gates in
  `.claude/docs/model-gates.md`. This surface is covered end to end on a
  scripted model (`crates/ffi/tests/c_surface.rs`), not on a real checkpoint.

### Estimating tokens

```swift
// Count tokens for a full conversation through the chat template:
let count = try await session.countTokens(
    [ChatMessage.system("You are a helpful assistant."), ChatMessage.user("Hello world")],
    reasoning: .off
)
print("prompt uses \(count) / \(session.info.maxContext) tokens")

// Or count raw tokens in arbitrary text without chat formatting:
let draftTokens = try await session.countTokens(in: "Draft user input...")
```

Renders the conversation through the checkpoint's chat template and counts the
exact tokens without allocating KV cache or executing forward passes. Use this
in a composer to update context meter gauges and warn users when a draft
approaches the window limit.

### Prompt rendering & template inspection

```swift
// Inspect the exact formatted prompt string passed to the model:
let rawPrompt = try await session.renderPrompt(
    [ChatMessage.system("You are a helpful assistant."), ChatMessage.user("Hello world")],
    reasoning: .medium
)
print(rawPrompt)
```

Formats the conversation through the model's Jinja chat template, formatting system
instructions, reasoning triggers, and role markers (e.g. `<|im_start|>`, `[INST]`,
`<|start|>user<|message|>`). Use this in a chat app to preview formatted prompts, debug
system prompts, and inspect dialect framing.

**A SYSTEM PROMPT IS A MESSAGE AND NOTHING ELSE AT THIS BOUNDARY.** There is
no `system_prompt` parameter anywhere in the ABI: it is `ChatMessage.system`
at index 0 of the conversation, and only at index 0 -- three of the five
fallback renderers refuse a system message at any other position, and
`fitWindow` protects only a LEADING system or developer turn. So exactly one,
first, is the shape every caller has to produce.

Where a prompt comes from is therefore the HOST's question, not this
binding's. `TurboSparkApp` resolves one from its own settings and per-chat
state and prepends it; `turbospark-server` resolves one from `--system` and
injects it when a request carries none. Neither is visible here, and a third
host would make its own arrangement.

### Tokenization & detokenization

```swift
// Encode raw text to integer token IDs:
let tokenIDs = try await session.tokenize("Hello, world!", addSpecialTokens: false)
print("Token IDs: \(tokenIDs)")

// Decode token IDs back to text:
let reconstructed = try await session.detokenize(tokenIDs, skipSpecialTokens: false)
assert(reconstructed == "Hello, world!")

// Inspect tokenizer special token IDs:
let special = session.info.specialTokens
print("BOS: \(String(describing: special.bosId)), EOS: \(String(describing: special.eosId))")
```

Exposes direct access to the model's tokenizer for token visualizers, token chip
highlighters, token-level editing, and span calculations in chat interfaces.

### Counting tokens before a model is loaded

`TurboSparkSession` maps gigabytes and compiles pipelines at open. A context
meter or a prompt-budget preview that runs before any model is loaded should
not pay that, and `TurboSparkTokenizer` reads only the tokenizer files:

```swift
let tokenizer = try TurboSparkTokenizer(modelPath: "~/models/gemma4.gturbo")  // or an alias
let n = try tokenizer.count("Draft user input...")
let ids = try tokenizer.tokenize("Hello, world!")
let text = try tokenizer.detokenize(ids)
tokenizer.close()   // also on deinit; use after close throws instead of crashing
```

Calls are synchronous and cheap, and safe from any thread. It does text-level
work only: rendering a conversation (`renderPrompt`) and fitting a window
(`fitWindow`) need the family, dialect and reasoning table the engine resolves
at open, so they stay on the session rather than growing a second, partial
copy here.

### Conversation window fitting & context budgeting

As chat conversations grow over many turns, transcripts easily exceed the
model's context window. `fitWindow` uses `turbospark-window-fit` and the model's
chat template to iteratively prune older eligible turns (retaining leading
system/developer instructions and the newest user turn) to fit into a target
token budget:

```swift
let outcome = try await session.fitWindow(
    conversation,
    maxTokens: 4096,  // nil defaults to session.info.maxContext
    reasoning: .off
)

if outcome.removedTurnCount > 0 {
    print("Pruned \(outcome.removedTurnCount) older turns to fit context window")
}

// Generate safely using the fitted transcript:
let stream = session.generate(outcome.retained, options: options)
```

`outcome.hasRoomForGeneration` is true when the fitted prompt leaves headroom
for generated response tokens.

### Embeddings & vector similarity

```swift
// Generate normalized vector embeddings for text snippets:
let embeddings = try await TurboSparkEmbedding.encode(
    texts: ["What is quantum computing?", "Subatomic particles in superposition."],
    modelPath: "snowflake-arctic-embed-m"
)

// Calculate cosine similarity between two vectors:
let score = TurboSparkEmbedding.cosineSimilarity(embeddings[0], embeddings[1])
print("Semantic similarity: \(score)")
// Rank documents by semantic similarity to a query:
let ranked = try await TurboSparkEmbedding.rank(
    query: "What is quantum entanglement?",
    documents: [
        "Photosynthesis converts light to chemical energy.",
        "Entangled particle states exhibit correlated measurements.",
        "Ancient roman aqueducts carried water into cities."
    ],
    modelPath: "snowflake-arctic-embed-m"
)
for (index, text, score) in ranked {
    print(String(format: "[%.3f] #%d: %@", score, index, text))
}

// Single text encoding:
let vector = try await TurboSparkEmbedding.encode(
    text: "What is quantum computing?",
    modelPath: "snowflake-arctic-embed-m"
)

// Retrieve top-k most similar documents:
let top2 = try await TurboSparkEmbedding.topK(
    query: "quantum physics",
    documents: [
        "Photosynthesis converts light to chemical energy.",
        "Entangled particle states exhibit correlated measurements.",
    ],
    k: 2,
    modelPath: "snowflake-arctic-embed-m"
)
```

`TurboSparkEmbedding.encode` runs BERT and XLM-RoBERTa encoder models in `.safetensors`
format with pooled, normalized output vectors. Use this for semantic search,
retrieval-augmented generation (RAG), and document ranking directly in-process.

### Background daemon inspection & control

```swift
// Inspect managed background daemon status (`turbospark start/stop/status`):
let daemon = try TurboSparkDaemon.status()
if daemon.running {
    print("Daemon running on port \(daemon.port!) (PID \(daemon.pid!))")
    print("Endpoint: \(daemon.endpoint!)")
}

// Start or restart the background daemon:
try TurboSparkDaemon.start(args: ["--port", "8080", "--model", "gemma4"])
try TurboSparkDaemon.restart()

// Stop the daemon cleanly:
try TurboSparkDaemon.stop()

// Connect external terminal coding agents:
let cmd = TurboSparkAgent.launchCommand(for: "claude", host: "127.0.0.1", port: 8080)
print("Run in terminal: \(cmd)")
```

### Asking the engine what a family or install can do

The app used to keep its own copy of two engine rules: which families dispatch
steering, and whether an install would accept `kvBits`. A copy is right until
the next family lands, and the engine refuses by name at open when they
disagree. Ask instead:

```swift
TurboSparkCapabilities.family("gemma4").steeringSupported            // false for an unknown family
TurboSparkCapabilities.kvQuantSupported(                              // manifest.json "arch" facts
    fullHeadDim: 128, layerMask: [1, 0, 1, 1], numLayers: 4)
```

Both are pure, work before any model is open, and answer "no" for anything
they cannot read. `family` never throws.

### Seeing what the engine writes to stderr

The engine reports some decisions only by `eprintln!` (a vision
auto-resolution, a speculation fallback, an oversize-image clamp), from crates
shared with the command-line tools, so they cannot take a host-specific
callback. They land on file descriptor 2 of your process:

```swift
try StderrCapture.shared.start { line in log.append(line) }   // one call per complete line
// ...
StderrCapture.shared.stop()                                    // restores the original stderr
```

It tees: each chunk still reaches the original stderr, so a terminal or Xcode
console keeps working. It is process-wide and one at a time (a second `start`
throws), the handler runs on a private queue and must not block it, and lines
carry no level or source because the engine emits none. This is a Swift-side
tool on purpose: capturing fd 2 from Rust would have needed a new dependency.

### Telemetry

```swift
let phases = try await session.phases()
phases.calls              // forward passes served
phases.totalMsPerCall
phases.expertIoMs         // expert streaming
phases.gpuWaitMs
phases.expertHitRate      // nil before anything has been requested

TurboSparkSession.peakFootprintBytes   // process-wide, or nil
if let sys = TurboSparkSession.systemTelemetry {
    print("RAM: \(sys.physicalMemoryBytes), thermal: \(sys.thermalLevel)")
    print("memory pressure: \(sys.memoryPressure)")   // normal | warn | critical
}

result.peakMemoryPressure   // the worst level seen DURING that turn
```

Three caveats, all of which make a naive status panel wrong:

**The phase counters are cumulative over every forward pass, prefill
included.** A per-call number is an average across the whole context range,
not a number at the current context. To get a figure *at* a context,
difference two runs.

**They cover the inside of the forward pass only.** The sampler and the
detokenizer run after it returns and appear in none of the buckets, so the
phase total will not add up to wall-clock decode time.

**`peakMemoryPressure` is `normal` when nothing was watching, which is not
the same as memory being fine.** The in-loop probe follows the power
profile's stepping, and the default (`performance`) polls nothing -- so on a
default session that field is the ABSENCE of a reading. `systemTelemetry`
polls unconditionally and is what a status panel should read; the field on
the result exists to catch a SPIKE between polls, which is the event worth
reacting to.

**The engine paces itself under pressure and never unloads.** Under
`.balanced` or `.efficiency` it steps its own decode rate down, taking
whichever of thermal and memory pressure binds harder. It does not close
sessions, because it does not own them -- your handle is yours, and a session
that destroyed itself would leave you holding a dead pointer. Deciding to
call `close()` on an idle session when pressure goes critical is the app's
job.

`peakFootprintBytes` is the same mach counter every published memory figure
for this engine uses, so your number and the memory oracle's agree. What it
*counts* differs by install shape: a streamed MoE model's mapped weights are
counted, a dense model's are not. Read it beside `info.maxContext` rather
than comparing across models.

---

## The in-process server

Serves your already-open models over HTTP, in this process. Not a second copy
of the engine: each attached model is the same one your `TurboSparkSession`
is generating through.

```swift
// Start with nothing attached. The socket binds immediately, so you can show
// and copy the address before the user has picked a model.
var options = ServerOptions(apiKey: "sk-local")
options.embeddingModel = "~/models/snowflake-arctic-embed-m"
options.hfEndpoint = "https://hf-mirror.com"
options.defaultSystem = "You are an expert AI assistant."
options.defaultReasoning = .medium
let server = try TurboSparkServer.start(options: options)

let id = try server.attach(session)      // "gemma4.gturbo" -- the install's own name
print(try server.info().baseURL!)        // http://127.0.0.1:53411

try server.detach(modelId: id)           // stops serving it, releases the engine
server.stop()                            // stops serving everything

// Attach an embedding model (.safetensors directory or alias) for vector endpoints:
let embId = try server.attachEmbeddingModel("~/models/snowflake-arctic-embed-m")
```

Routes: OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
`/v1/models`, `/v1/embeddings`), Anthropic (`/v1/messages`,
`/v1/messages/count_tokens`), Ollama (`/api/tags`, `/api/version`, `/api/show`,
`/api/chat`, `/api/generate`, `/api/embeddings`, `/api/embed`), and `GET /health`.
Embeddings routes require an attached embedding model (via `attachEmbeddingModel(_:)`
or `ServerOptions.embeddingModel`).

**Which model serves a request.** An exact `model` id wins. Failing that, if
exactly ONE model is attached it serves the request whatever name was asked
for -- which is what lets a client sending its own default (Claude Code sends
`claude-sonnet-4-6`) work with no configuration. With two or more attached
and no match, the request is a 404 naming what is available.

**Claude Code model discovery.** Each generative attachment has two public
`GET /v1/models` rows: its canonical id and
`claude-turbospark-<canonical-id>`. Pass that discovery alias to Claude Code
at launch, so it selects the attached backend before `/model` is opened:

```sh
claude --settings '{"env":{"ANTHROPIC_BASE_URL":"http://127.0.0.1:53411","ANTHROPIC_API_KEY":"unused","CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY":"true","CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT":"1"}}' \
  --model "claude-turbospark-gemma4.gturbo"
```

Use `--settings` for the ephemeral app port. It gives this invocation an
explicit one-session setting instead of relying on the surrounding shell when
a saved Claude Code `env` block still names an old port.

`ServerInfo.models` intentionally remains canonical-only, in attachment
order, so it continues to identify the sessions that `attach` and `detach`
manage. Use the canonical id returned by `attach` to derive the Claude alias.
Embedding-only attachments have no discovery alias. `HEAD /api/hello` may
return 404 because it is only Claude Code's connection-warming probe; use
`GET /health` for liveness. The unknown-model-window override makes Claude
Code defer its built-in unknown-model context assumption to this gateway.

**Read the address off `info()`, never spell it.** `ServerOptions.port` of 0
asks the OS for a port, and `ServerInfo` reports both halves of what was
actually bound; `baseURL` builds the string. Restating `127.0.0.1` is correct
only for as long as the bind does not change, and cannot report the day it
does.

**Detaching is what frees a model.** A server holds its own reference to
every engine attached to it, so releasing your `TurboSparkSession` does NOT
unload the weights while the server is still serving them. `detach(modelId:)`
or `stop()` does. Anything in your UI that says a model is unloaded has to
have called one of them.

**Unauthenticated does not mean private.** The socket defaults to loopback, which
keeps it off the network and reachable by every process on this machine.
`apiKey` is the only access control there is, and an empty or
whitespace-only key means none at all -- `info().authEnabled` is what
actually happened.

**`ServerOptions.guardrails` IS A SECOND ENFORCEMENT POINT, and a host with
its own guardrails setting has to pass it here too.** `.on` / `.off` / absent
(the engine default, on), with the same grammar and the same process-level
scope `turbospark-server --guardrails` has. It applies to every model attached
to that server, including ones attached later, and cannot be changed without
restarting it. A host that applies its own repair to a reply it read itself
covers only that path: every HTTP client of this server bypasses it. See
`docs/FORGE_GUARDRAILS.md` section 0b.

**Idle unload.** `ServerOptions.idleUnloadSeconds` (or
`server.setIdleUnload(after:)` on a running server) detaches a model that has
served no request for that long, releasing its weights. Nil or 0 keeps every
model resident, which is the default and the only behaviour before 2026-10.
The sweep runs every 30 seconds, never detaches a model with a request in
flight, and its detach arrives as an ordinary `.modelDetached` event. A model
that was detached this way is gone from `info().models`; a host that wants it
back attaches a session again.

**Default system prompt and reasoning effort.** `ServerOptions.defaultSystem`
supplies a deployment-wide system message for requests that carry no system or
developer message of their own (`turbospark-server --system`).
`ServerOptions.defaultReasoning` sets the default reasoning effort (.off, .low,
.medium, .high, .xhigh) applied when a request omits `reasoning_effort`
(`turbospark-server --reasoning`).

### Watching it

```swift
let batch = server.poll()          // drains; each event is returned once
for event in batch.events {
    switch event {
    case .generated(let gen):
        print("gen: \(gen.tokenCount) tok, reused prefix: \(gen.reusedPrefixTokens), evicted: \(gen.sessionSlotEvicted)")
    default:
        break
    }
}
if batch.dropped > 0 { /* say so */ }
```

Poll on a timer and append what you get. Events are `requestStarted`,
`requestRouted`, `generated`, `requestFinished`, `modelAttached`,
`modelDetached`, tied together by a request id, plus an `unknown(kind:)` case
so a newer engine's event does not fail the whole batch. `generated` includes
KV cache prefix reuse counts (`reusedPrefixTokens`) and slot eviction notices
(`sessionSlotEvicted`).

`dropped` counts events the engine's ring discarded since your previous poll.
**Show it.** A console quietly missing rows reads exactly like a server that
was idle, and those are the two states somebody watching it is trying to tell
apart.

There is deliberately no time-to-first-token field. A caller means "request
in, first token out", which includes the wait behind the one-turn-at-a-time
lock, and nothing inside a generation can see that. What you get is
`prefillSeconds` and `decodeSeconds` off the decoder plus `durationMs` from
the HTTP layer, which does include the queue -- subtract to get the wait. A
field named `ttftMs` filled from the prefill would read as the first number
and be the second.

Token counts come off the decoder's own result. Do not substitute a count of
streamed chunks: special tokens render to the empty string, the detokenizer
withholds partial UTF-8, and reasoning goes to a different channel, so that
number is low by an amount that varies with the dialect and the turn.

---

## Model management

Available on every platform, including ones that cannot then run a model. The
artifact is the same either way.

```swift
let rows = try TurboSparkCatalog.available()      // curated table, with `installed`
let mine = try TurboSparkCatalog.installed()      // what is in ~/.turbospark
let cost = try TurboSparkCatalog.cost(of: "gemma4")
// Ranked by hardware fit. Pass the SAME guard your sessions open with, or
// the ranking and the loader's refusal describe different machines.
let recs = try TurboSparkCatalog.recommend(context: 4096, loadGuard: .balanced)
let report = try TurboSparkCatalog.probe(repo: "Qwen/Qwen3-30B-A3B-GGUF",
                                         file: "Qwen3-30B-A3B-Q4_K_M.gguf")

// List GGUF variant files in a Hugging Face repo without downloading headers:
let variants = try TurboSparkCatalog.variants(repo: "Qwen/Qwen3-30B-A3B-GGUF")

// Cost across context rungs for an installed model (KV cache scaling):
let ladder = try TurboSparkCatalog.contextLadder(modelPath: "gemma4", loadGuard: .balanced)

// Inspect control vector metadata before opening:
let cv = try TurboSparkCatalog.controlVectorInfo(path: "/path/to/vector.gguf")

// Delete an installed model to recover disk space:
try TurboSparkCatalog.delete("gemma4")

// Check if an alias or directory is installed, and resolve canonical path:
if try TurboSparkCatalog.isInstalled("gemma4") {
    let path = try TurboSparkCatalog.resolvePath(for: "gemma4")
    print("Resolved path: \(path ?? "")")
}
```

`probe` returns JSON rather than a struct, because a probe report's shape
follows what the engine learns to read. Its useful keys are `runnable`,
`refusedBecause`, and `slotCacheBytes` -- read that last one before
`downloadBytes`, since what decides whether a model runs here is
`slots x layers x expert stride`, not the model's size.

### Installing

```swift
var downloaded: UInt64 = 0
var expected: UInt64 = 0

for try await event in TurboSparkCatalog.install("gemma4") {
    switch event {
    case .stage(let line):
        status = line
    case .bytes(let done, let total):
        // MAX, not last: see below.
        downloaded = max(downloaded, done)
        if total > 0 { expected = max(expected, total) }
    case .finished(let model):
        print("installed at \(model.path)")
    }
}

// Or install an arbitrary probed Hugging Face repository:
for try await event in TurboSparkCatalog.install(
    repo: "Qwen/Qwen3-30B-A3B-GGUF",
    alias: "my-qwen",
    file: "Qwen3-30B-A3B-Q4_K_M.gguf"
) {
    // ...
}
```

Two things a progress UI has to get right:

**Pause preserves work while the app stays open.** `pauseInstall()` stops
text-model installs at download boundaries; in-flight requests finish first.
`resumeInstall()` continues the same worker with its buffers and output intact.
Cancellation, failure, or quitting requires a new install call. For an
immutable revision that call reuses SHA-256 checked ranges; conversion may
restart. The app's
bottom-right Downloads panel saves the last 12 attempts, exact sources, byte
counts, and timestamps in the encrypted profile vault. Running or paused rows
return as Interrupted after reopening, with Retry reusing verified ranges
where the repository is pinned.
Completed rows stay completed; loading a model is separate from installing it.
Progress snapshots are throttled to one write per two seconds and flushed on
status changes and shutdown. Transfer speed uses a ten-second moving average
of byte events, clears during pause, and resets on resume. Cancel retains the
install slot until the native stream closes, so Retry cannot race that writer.
The reservation also survives profile lock and unlock while that worker stops.

**Take the maximum of byte events, not the latest.** Ranged downloads are
split across connections, so byte progress arrives concurrently and out of
order. Using the last value makes the bar jump backwards.

### Hugging Face authentication and mirror endpoints

```swift
// Inspect current token resolution and source:
if let tokenInfo = TurboSparkCatalog.getHfTokenInfo() {
    print("HF Token active via \(tokenInfo.source)")
}

// Validate token with whoami API:
if let token = try TurboSparkCatalog.getHfToken() {
    let status = try TurboSparkCatalog.validateHfToken(token)
    switch status {
    case .valid(let name, _, _): print("Authenticated as @\(name ?? "user")")
    case .invalid(let msg): print("Invalid token: \(msg ?? "")")
    case .rateLimited: print("Rate limited")
    case .unavailable(let msg): print("Offline: \(msg)")
    case .missing: print("No token")
    }
}

// Set or clear token explicitly (saved to ~/.turbospark/hf_token):
try TurboSparkCatalog.setHfToken("hf_your_token_here")
try TurboSparkCatalog.clearHfToken()

// Override Hugging Face download endpoint (e.g., custom mirror or proxy):
try TurboSparkCatalog.setHfEndpoint("https://hf-mirror.com")
```

The catalog queries tokens hierarchically: CLI flag -> `$HF_TOKEN` ->
`~/.turbospark/hf_token` -> standard Hugging Face CLI token cache
(`~/.cache/huggingface/token`). `getHfTokenInfo()` reports both the token
and its origin `source`. Setting a custom `hfEndpoint` configures download and
probe calls to target the specified base mirror URL.

---

## Errors

```swift
do {
    let session = try await TurboSparkSession(modelPath: path)
} catch let error as TurboSparkError {
    error.code      // .invalidArgument .open .generate .json .unsupportedPlatform .panic
                    // .cancelled .busy
    error.message   // a sentence, from the library
}
```

`.cancelled` is the caller's own cancel request ending an install or an audio
job. It is not a failure and should not be shown as one. (A cancelled TEXT turn
is not an error at all: it returns normally with `stopReason == .cancelled` and
the partial text.) `.busy` means the resource is in use by another operation,
today the native audio device or session; unlike `.open`, retrying once that
job finishes can succeed. Before 2026-10 a cancelled install or audio job came back as `.generate` and
a busy audio open as `.open`.

`.panic` means a panic was caught at the boundary. The process is intact and
the operation did not happen; it is a library bug rather than anything the
caller did, and it is worth reporting with the message attached.

A failed header/archive match (see the top of this file) surfaces as code
`.unknown` with a message naming both ABI revisions.

`.unsupportedPlatform` is what `ts_session_open` returns off macOS. The engine
is macOS-only, so the catalog, probe and install calls work everywhere and
opening a model does not.

---

## The C ABI directly

For a host that is not Swift. The canonical description is
`crates/ffi/include/turbospark.h`; this is the contract in four rules.

**1. Errors.** Every fallible call returns `TS_OK` (0) or a non-zero code.
On a non-zero return, `ts_last_error()` **on the same thread** holds a
message. Read it before making another call on that thread.

**2. Ownership.** A `const char *` argument is borrowed for the duration of
the call and never retained. A `char **` out-parameter receives an allocation
you return through `ts_string_free()`. There is no third case, and nothing
hands back a pointer into library state.

**3. JSON.** Options and results are JSON strings, so adding a knob is never
an ABI break. Keys are camelCase. The per-token path carries no JSON: it is a
pointer and a length.

**4. Threading.** A session is single-threaded and `ts_generate` blocks for
the whole turn, so call it from a background thread. The one exception is
`ts_session_cancel`, safe from any thread and non-blocking. `ts_install`'s
byte callback is *also* called concurrently from worker threads.

### The surface

Image catalog and installation are separate from text model catalog rows. Text
installs are rooted at `~/.turbospark/models/text`, and image installs are
rooted at `~/.turbospark/models/image`. The C ABI exposes
`ts_image_catalog_json` for curated image sources,
`ts_image_installed_json` for verified image installs, and
`ts_image_install` for staged source download and packing. Swift wraps these
as `TurboSparkCatalog.imageAvailable()`,
`TurboSparkCatalog.imageInstalled()`, and
`TurboSparkCatalog.installImage(_:)`. The install event stream reports stage,
monotonic byte-progress handling remains the caller's responsibility, and a
finished event returns an `ImageInstalledModel` with the verified path and
image metadata.

Image generation is a separate surface from token generation. `TsImageSession`
opens a verified image install, `ts_image_generate` blocks on one
serialized heavyweight job, stage callbacks borrow their text only for the
callback, and the returned PNG must be released with `ts_image_buffer_free`.
Swift copies the PNG into `Data` before releasing the native buffer. An image
session must not be closed while its generation call is active.

The Swift package keeps the wire keys camelCase and exposes the request as a
Codable value:

```swift
let options = ImageGenerateOptions(
    prompt: "A red rabbit under a moonlit sky",
    seed: 42,
    width: 1024,
    height: 1024,
    steps: 9
)
let outputURL = URL(fileURLWithPath: "image.png")
for try await event in imageSession.generate(options) {
    if case .finished(let result) = event {
        try result.png.write(to: outputURL)
    }
}
```

The encoded JSON contains `prompt`, `seed`, `width`, `height`, and `steps`.
The app's `Images` destination stores the PNG below the active profile's
`AppStorageRoot` only after the job succeeds and the user saves it. The saved
request metadata powers deterministic regeneration and the Gallery carousel.

| function | notes |
|---|---|
| `ts_last_error(buf, cap)` | returns the message's own length, not bytes written; `buf` may be NULL to size |
| `ts_string_free(s)` | for every `char **` out-parameter |
| `ts_session_open(dir, options_json, out)` | expensive; open once |
| `ts_session_close(s)` | not while a generation is in flight |
| `ts_session_cancel(s)` | any thread, never blocks |
| `ts_session_info_json(s, out)` | resolved window, slots, family, dialect, speculation, specialTokens |
| `ts_session_phases_json(s, out)` | decode phase breakdown |
| `ts_peak_footprint_bytes()` | process-wide, 0 if unavailable |
| `ts_abi_version()` | the library's ABI revision; compare with the header's `TS_ABI_VERSION` |
| `ts_build_info_json(out)` | ABI revision, crate version, whether debug assertions are on |
| `ts_system_info_json(out)` | hardware RAM, chip, power, thermal status |
| `ts_session_count_tokens(s, messages, reasoning, out_count)` | evaluates exact prompt token count |
| `ts_session_render_prompt(s, messages, reasoning, out_prompt)` | formats conversation into raw prompt text |
| `ts_session_tokenize_json(s, text, add_special, out)` | tokenizes text into JSON array of token IDs |
| `ts_session_detokenize_json(s, tokens_json, skip_special, out)` | decodes token IDs into text string |
| `ts_session_count_text_tokens(s, text, add_special, out_count)` | evaluates raw text token count |
| `ts_session_fit_window_json(s, messages, reasoning, max_tokens, out)` | fits conversation into token budget |
| `ts_tokenizer_open(dir, out)` / `ts_tokenizer_close(t)` | a tokenizer with no engine behind it |
| `ts_tokenizer_count_text_tokens(t, text, add_special, out_count)` | token count with no session |
| `ts_tokenizer_tokenize_json(t, text, add_special, out)` / `ts_tokenizer_detokenize_json(t, tokens_json, skip_special, out)` | text to ids and back with no session |
| `ts_generate(s, messages, options, cb, ud, out)` | blocks for the turn; `options.tools` offers functions, `toolCalls` reports the calls |
| `ts_server_start(s, options_json, out)` | start server; `s` may be NULL, `options_json` configures port, api_key, embedding, hf_endpoint, defaults |
| `ts_server_attach_session(server, s, out_model_id)` | attach loaded session |
| `ts_server_attach_embedding_model(server, model_path, out)` | attach embedding model for /v1/embeddings |
| `ts_server_detach_model(server, model_id)` | detach model from server |
| `ts_server_set_idle_unload(server, seconds)` | detach a model after it has been idle this long; 0 turns it off |
| `ts_server_stop(server)` | stop server and drop all listeners |
| `ts_server_info_json(server, out)` | host, port, active model IDs, auth, uptime |
| `ts_server_poll_events_json(server, max, out)` | drain server event ring buffer |
| `ts_catalog_json(out)` | every platform |
| `ts_installed_json(out)` | every platform |
| `ts_image_catalog_json(out)` | curated image sources, every platform |
| `ts_image_installed_json(out)` | installed image rows under the image namespace, every platform |
| `ts_model_delete(alias)` | delete installed model directory and forget row |
| `ts_recommend_json(context, options_json, out)` | rank curated models by hardware fit; `options_json` takes `loadGuard` and may be NULL; each row also carries `evidence` (`discovered` / `caveat` / `runs` / `verified`) and `suspicious` |
| `ts_probe_json(repo, file, sidecar, out)` | header-only, no download |
| `ts_context_ladder_json(model_path, options_json, out)` | memory cost per context rung for installed model |
| `ts_repo_variants_json(repo, out)` | list all GGUF variants published by repository |
| `ts_control_vector_info_json(path, out)` | a `.gguf` control vector's shape: no model, no session, no network |
| `ts_kv_quant_supported(full_head_dim, layer_mask, len, num_layers)` | whether `kvBits` would be accepted for an install with these arch facts; 1 or 0, never fails |
| `ts_family_capabilities_json(family, out)` | `{family, known, steeringSupported}` from the persisted family spelling |
| `ts_install_bytes_json(alias, out)` | cost before committing |
| `ts_install(alias, cb, ud, out)` | blocks for minutes; later calls reuse verified ranges for pinned revisions |
| `ts_install_repo(repo, alias, file, sidecars, cb, ud, out)` | install arbitrary HF repository |
| `ts_image_install(alias, cb, ud, out)` | ranged download and pack; later calls reuse verified ranges |
| `ts_embedding_encode_json(model_path, texts_json, out)` | standalone batch text embedding generation |
| `ts_cosine_similarity(a, b, len)` | cosine similarity between two float vectors |
| `ts_image_session_open(model_dir, out)` | open a verified image install |
| `ts_image_session_close(s)` | close an image session when no job is running |
| `ts_image_session_cancel(s)` | cancel image generation from any thread |
| `ts_image_generate(s, options, cb, ud, png, len, metadata)` | blocking image job with stage callbacks, owned PNG bytes, and camelCase metadata |
| `ts_image_buffer_free(bytes, len)` | release PNG bytes returned by `ts_image_generate` |
| `ts_hf_token_get(out)` | read resolved HF token |
| `ts_hf_token_info_json(out)` | read resolved HF token and origin source |
| `ts_hf_token_set(token)` | save HF token to ~/.turbospark/hf_token |
| `ts_hf_token_clear()` | clear stored HF token |
| `ts_hf_token_validate_json(token, out)` | validate HF token against whoami API |
| `ts_hf_endpoint_get(out)` | read resolved HF endpoint / mirror URL |
| `ts_hf_endpoint_set(endpoint)` | set or clear $HF_ENDPOINT mirror override |
| `ts_model_resolve_path(alias_or_path, out)` | resolve alias or path to canonical install directory |
| `ts_daemon_status_json(out)` | read background daemon status (running, pid, port, endpoint, logPath, model) |
| `ts_daemon_stop()` | stop background daemon if running |
| `ts_daemon_start(args_json)` | start background server daemon with optional arguments |
| `ts_daemon_restart(args_json)` | restart background server daemon with optional arguments |
| `ts_audio_*`, `ts_heavy_work_*` | the native audio surface (catalog, install, adopt, delete, delete_legacy, sessions and jobs); see "Audio" below and the header's audio block |

### A complete C example

```c
#include "turbospark.h"
#include <stdio.h>
#include <string.h>

static void on_event(void *ud, int32_t kind, const char *text, size_t len,
                     uint32_t a, uint32_t b) {
    (void)ud;
    if (kind == TS_EVENT_PREFILL) {
        fprintf(stderr, "\rprompt %u/%u", a, b);
    } else if (kind == TS_EVENT_CONTENT) {
        /* `text` is NOT NUL-terminated and is valid only for this call. */
        fwrite(text, 1, len, stdout);
        fflush(stdout);
    }
}

static void die(const char *what) {
    char buf[1024];
    ts_last_error(buf, sizeof buf);
    fprintf(stderr, "%s: %s\n", what, buf);
}

int main(void) {
    TsSession *s = NULL;
    if (ts_session_open("/Users/me/models/gemma4.gturbo", "{}", &s) != TS_OK) {
        die("open");
        return 1;
    }

    char *info = NULL;
    if (ts_session_info_json(s, &info) == TS_OK) {
        fprintf(stderr, "%s\n", info);
        ts_string_free(info);
    }

    const char *messages = "[{\"role\":\"user\",\"content\":\"Hello\"}]";
    const char *options  = "{\"maxNewTokens\":200,\"temperature\":0.2}";
    char *result = NULL;

    if (ts_generate(s, messages, options, on_event, NULL, &result) != TS_OK) {
        die("generate");
        ts_session_close(s);
        return 1;
    }
    fprintf(stderr, "\n%s\n", result);   /* stopReason, tokensPerSecond, ... */
    ts_string_free(result);

    ts_session_close(s);
    return 0;
}
```

### Option and result shapes

`ts_session_open` options, all optional; `{}` means fully automatic:

```json
{
  "maxContext": 8192,
  "expertCacheSlots": "auto",
  "powerProfile": "efficiency",
  "maxTokensPerSec": 30,
  "loadGuard": "balanced",
  "minAutoContext": 8192,
  "speculation": "auto",
  "speculativeDrafter": "auto",
  "steering": "/path/to/vector.gguf",
  "steeringMode": "ablate",
  "steeringScale": 0.5,
  "steeringLayers": "20:45",
  "steeringTarget": 0.0,
  "steeringGate": 0.0,
  "visionSidecar": null,
  "expertResidency": "auto",
  "kvBits": "3.5",
  "prefixReuse": true
}
```

`maxContext` and `expertCacheSlots` accept a number, the string `"auto"`,
`null`, or absence; all four spellings of automatic mean the same thing,
because a caller's encoder may produce any of them. Any *other* string is an
error rather than a silent fallback.

`speculation` accepts `"off"`, `"auto"`, `null`, absence, or a block size as
either a number or a string. `speculativeDrafter` accepts `"auto"`, `"mtp"`
or `"dflash"`. Both are refused by name rather than defaulted when
misspelled, and both are mapped BEFORE the install is touched, so a bad
option is reported ahead of a bad path.

`loadGuard` accepts `"off"`, `"relaxed"`, `"balanced"`, `"strict"`, a
positive NUMBER (an absolute ceiling in bytes on what the engine allocates),
`null` or absence. Null and absence mean `"relaxed"`. An unrecognized string
is an error rather than a silent fallback, for `maxContext`'s reason and one
of its own: quietly ranking or opening under the default when the caller
asked for `"strict"` is exactly the disagreement the option exists to
prevent. `minAutoContext` accepts a number or `null`; 0 and absence both mean
no floor.

`steering` accepts a file path string to a `.gguf` control vector, or `null`.
`steeringMode` accepts `"ablate"`, `"add"`, `"clamp"`, `"renorm"`.
`steeringScale` accepts a finite number (default 1.0).
`steeringLayers` accepts `"START:END"` (0-based inclusive layer range).
`steeringTarget` and `steeringGate` accept finite numbers.

`expertResidency` accepts `"auto"`, `"streamed"`, `"mapped"`, `null` or absence
(auto). Auto maps the routed experts only when the minimum streamed cache
cannot fit the measured headroom; the mode the session actually opened with is
`sessionInfo.expertResidency`. A misspelling is refused before the install is
opened. `expertCacheSlots` as a number must be one of 8, 16, 24, 32, 48, 64,
96, 128 (`foundation::runtime_config::ALLOWED_CACHE_SLOTS`); `"auto"` climbs
through 8/16/24/32 only.

`prefixReuse` (default `true`, which is what every release did) continues each
turn from the previous turn's KV wherever the new render shares a prefix. Open
a session you will attach to the in-process server with `false` if you do not
want one conversation's cached prefix shared with every HTTP client of that
server: `turbospark-server` itself defaults reuse off for exactly that reason,
and an in-process server over a default session does not. The resolved value is
`sessionInfo.prefixReuse`.

`kvBits` accepts `"off"`, `"2"`, `"3"`, `"3.5"`, `"4"`, `null` or absence.
Null and absence mean `"off"`, which is what every release before this key
existed produced byte for byte. `"3.5"` splits into K3/V4, mlx-vlm's own
convention for its one fractional width. An unsupported family or
`head_dim` is an error rather than a silent fallback to FP16 -- see
`docs/TRUBOQUANT.md`.

`ts_session_info_json` reports what they resolved to:

```json
{
  "steering": { "active": true, "mode": "ablate", "scale": 0.5, "summary": "ablate at alpha 0.5 over 26 of 64 layers" },
  "speculation": { "block": 2, "drafter": "mtp", "reason": null },
  "specialTokens": {
    "bosId": 1,
    "eosId": 2,
    "padId": 0,
    "endOfTurnId": 151645,
    "stopTokenIds": [151643, 151645],
    "thinkStartId": 151648,
    "thinkEndId": 151649
  },
  "kvBits": "3.5 (K3/V4)",
  "expertResidency": "streamed",
  "prefixReuse": true,
  "vision": { "active": true, "maxPixels": 1003520 }
}
```

`steering.active` indicates whether a control vector is loaded and active.
A null `block` is the "is it off" test; `drafter` is non-null exactly when
`block` is, and `reason` is non-null only when a caller might have expected
it on.

`ts_generate` options and its result:

```json
{ "maxNewTokens": 512, "temperature": 0.2, "topK": 64, "topP": 0.95,
  "repetitionPenalty": 1.0, "minP": 0.0, "presencePenalty": 0.0, "frequencyPenalty": 0.0,
  "seed": null, "stop": [], "stopTokens": [], "tools": [], "reasoning": "off" }
```

```json
{ "promptTokens": 21, "newTokens": 120, "reusedPrefixTokens": 0,
  "prefillSeconds": 0.41, "decodeSeconds": 2.87, "stopReason": "maxTokens",
  "toolCalls": [], "tokensPerSecond": 41.8, "content": "...", "reasoning": "",
  "peakMemoryPressure": "normal" }
```

`tokensPerSecond` is `null` when no decoding happened, so nothing can plot a
rate that was never measured. `stopReason` is one of `endOfTurn`,
`toolCalls`, `eos`, `stopString`, `maxTokens`, `cancelled`.

`ts_session_fit_window_json` result:

```json
{
  "retained": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Latest user message" }
  ],
  "measuredTokens": 1420,
  "removedTurnCount": 2,
  "hasRoomForGeneration": true
}
```

---

## Audio

Audio family and checkpoint status lives in
[`crates/audio/MODELS.md`](../crates/audio/MODELS.md), and the per-family
capability table in
[`docs/AUDIO_WORKSPACE_CAPABILITIES.md`](AUDIO_WORKSPACE_CAPABILITIES.md). Rust
model support, runtime integration, C ABI support and Swift support are
separate gates, and a family being implemented in Rust does not make it
reachable from Swift.

### What is reachable from Swift

The native path is `ts_audio_*` in the header, wrapped by `Audio.swift`
(`AudioCatalog`, `AudioSession`, `AudioJob`). The runtime opens a session for
exactly three tasks:

| Task (`AudioTask`) | Family reached today | Notes |
|---|---|---|
| `.speechToText` | Whisper | Catalog install (`whisper-base`) or a chosen folder. Qwen3-ASR also opens, from the app's Advanced page, behind an explicit portable or experimental-Metal opt-in and a local folder. |
| `.textToSpeech` | Kokoro | Portable CPU path behind an explicit opt-in at the time of writing; the current backend list is in `crates/audio/MODELS.md`. |
| `.music` | MiniMax Music 3 | Catalog install covers `minimax-music3-4bit`. |

`AudioTask` has ten cases because the catalog and capability rows describe
families the engine has code for but no session entry point yet (VAD,
diarization, alignment, enhancement, separation, codec, language ID). Opening
one is refused at `ts_audio_session_open` as an unknown task. **Use
`AudioTask.isRunnable` (and `AudioTask.runnable`) before offering a control**:
the app's Clean Up page and the unrunnable Advanced operations are hidden for
exactly this reason, and `crates/ffi/tests/audio_surface.rs` pins the accepted
names so the Swift predicate cannot drift from the runtime.

```swift
let profiles = try AudioCatalog.profiles()               // pinned, installable rows
let report = try AudioCatalog.installed()                // see below
let record = try AudioCatalog.install(profile.identity)  // blocking: call off the main actor

let session = try AudioSession(modelPath: record.path, task: .music)  // verifies bytes: off main
var request = AudioRequest(task: .music)
request.caption = "piano"; request.lyrics = "[instrumental]"

let result = try await session.execute(request) { progress in ... }   // whole result at the end

// Live output, to start playback before the job finishes:
for try await event in session.stream(request) {
    switch event {
    case .progress(let p): ...
    case .pcm(let chunk): ...        // at most 32,000 samples, interleaved
    case .finished(let result): ...
    }
}
```

`execute` deliberately has no PCM parameter. A second closure parameter would
capture an existing caller's trailing closure (Swift 5 mode binds it to the
last closure parameter), turning a progress callback into a PCM one without a
compile error; `executeStreaming(_:pcm:progress:onPCM:)` and `stream(_:pcm:)`
are the live-output entry points. The full audio is still returned in the
result.

### Installs: installed, adopt, delete

`AudioCatalog.installed()` returns four lists. `installed` are receipt-backed
and usable. `needsAdoption` are installs the receipt store does not know yet
but whose identity matches a pinned profile: a model pulled by the CLI
(`pull-audio` of a music model) writes no receipt, so without adoption it is
invisible. `AudioCatalog.adopt(_:)` verifies it and makes it usable.
`incompatible` are installs that exist on disk but cannot be used, with the
engine's reason; `otherAudio` is audio the pinned profiles do not describe.
Removal is `AudioCatalog.delete(_:)` for a receipt-backed install and
`AudioCatalog.deleteLegacy(_:)` for an unadopted one (including a damaged one;
it refuses a receipt-backed install). Close any resident `AudioSession` on the
model first. The app exposes adopt, delete (with a confirmation) and a list of
installs that need attention in the audio model picker.

### Wire conventions that differ from the rest of the ABI

The audio block of the header is snake_case, not camelCase, because those
structs are the runtime's own serde types passed through unchanged. The header
documents the callback `kind` values (1 is progress JSON, 2 is borrowed PCM),
the append and run lifecycle, and the cancellation calls. The legacy
`ts_stt_*` path (`TurboSparkSTTModel`, `TurboSparkSTTStream`) is a separate,
older surface: it takes no heavy-work permit, falls back to Whisper for a
non-Qwen `model_type`, and cannot cancel an in-flight finish. Prefer
`AudioSession`.

### Known limits

- **Opening a managed model hashes every byte** with no progress and no cancel
  (`ts_audio_session_open`), including after each idle unload. Call it off the
  main thread and expect it to take a while on a multi-gigabyte model.
- **Music progress is indeterminate.** The runtime reports `completed: 0` with
  no total; the generator has determinate counters that are not wired through
  yet.
- **Whisper emits no per-window progress**, which is why the app still chops
  long recordings into windows and reconciles the boundaries itself.
- **Seed.** A nil music seed resolves to 0 in the engine, not a random one.
- **Platform.** Off macOS the audio symbols return `TS_ERR_UNSUPPORTED`.

### Binding qualification

Reuse `make swift-lib` to stage the static library and canonical header, then
run `make swift-test`. Checkpoint-dependent tests are opt-in and name the
pinned model they exercise; `AudioTests` uses the committed
`crates/audio/testdata/minimax_music3/converted_plain` fixture, which is what
exercises native ownership, PCM copying and cancellation end to end. Swift
binding validation, transcript quality, accelerated execution and app
packaging each require separate evidence.

---

## Building and shipping

```bash
make swift-lib                                     # required first
make swift-test                                    # ABI checks, no model
make swift-test-real MODEL=~/models/gemma4.gturbo  # end to end, minutes
make swift-demo                                    # the chat app

# BLOCKED opens a SECOND install, for the cases MODEL cannot reach.
make swift-test-real MODEL=~/models/qwen38-27b-mtp.gturbo \
                     BLOCKED=~/models/ornith35b.gturbo
```

**arm64 only.** The engine is Metal on Apple Silicon and has never been run
on an Intel Mac. Add a `lipo` step to `scripts/swift-lib.sh` if that changes.

**`MACOSX_DEPLOYMENT_TARGET` must match both `Package.swift` files.** The
script sets 13.0. Without it, cargo builds for the host SDK's default while
SwiftPM links for 13.0, which draws an `ld` warning per object file. The
warnings are the visible half; the real problem is an app claiming support
for macOS 13 while containing objects built against a much newer SDK.

**The demo has no app bundle**, because a `swift run` binary does not get
one. It promotes itself with `NSApp.setActivationPolicy(.regular)` so the
window can take focus; a shipping app carries an Info.plist and needs none of
that.

---

## What is tested, and by what

The three layers are covered by three different things, and each catches
something the others structurally cannot.

| layer | test | catches |
|---|---|---|
| Rust, no model | `cargo test -p turbospark-ffi` | ownership, error propagation, the panic guard, cancellation from a second thread |
| ABI, no model | `make swift-test` | **the hand-written header disagreeing with Rust** |
| whole stack | `make swift-test-real` | streaming, cancel, telemetry against a real Metal forward pass |
| app | `make swift-test-app` | the app's own logic over the package (adopt/delete decisions, attachment transcoding, localization parity) |

**The bottom row's coverage is a property of the install you point it at**,
which is easy to miss because the other two rows are not. Speculation resolves
from the artifact, so `MODEL` alone decides whether the drafter cases run or
skip; `BLOCKED` adds an install that architecturally cannot speculate, which is
the only way the refusal path is reached at all. The tests verify that
`BLOCKED` really is a MoE or sub-4-bit install rather than believing the
caller, because a merely headless one passes every other assertion while
re-testing a case already covered.

The middle row is not optional and is not redundant. `crates/ffi`'s own tests
reach the same function bodies through the `rlib`, so they pass against a
header that declares the wrong signature entirely; only linking the
`staticlib` and calling through `turbospark.h` can catch that. It has already
earned its place once, catching that the catalog's on-disk rows are
snake_case where this binding's own wire shapes are camelCase.

Measured on an M4 Max against the real Gemma 4 26B install, as an indication
of shape rather than a benchmark (`docs/BENCHMARKS.md` holds the frozen
numbers):

```
open:     family=gemma4 context=4096 slots=32 vocab=262144
generate: 120 tokens at 41.8 tok/s
cancel:   stopped after 21 tokens in 1.0s, against a 4000-token budget
phases:   23 calls at 25.4 ms, expert hit rate 0.74, peak 3743 MiB
```

The peak is at 32 auto-resolved slots. At the pinned 16 that every published
figure uses it is around 2,180 MiB.

**Which archive is staged decides what links.** `make swift-lib` stages a plain
archive for the package tests; `make swift-lib-app` (which `swift-test-app` and
`swift-app-build` depend on) localizes `_rust_eh_personality` for the app, and
a plain `swift test` in `swift/TurboSpark` against THAT archive fails to link
with `Undefined symbols ... _rust_eh_personality`. Run `make swift-lib` again
before package tests after an app build. A stale archive under a newer header
is now reported by name rather than as a missing JSON key
(`TurboSparkRuntime.verifyABI()`; see the top of this file), but the
`RuntimeABITests` that pin it can only fail once the two have been staged
from different builds.

**Scripted-session tests count PROMPT tokens too.** `ScriptedLogitProducer`
is called once per prompt token during prefill as well as once per generated
token, so a scripted reply needs `prompt_len - 1` filler steps in front of it
(`scripted_reply_after_tool_prompt` in `crates/ffi/tests/c_surface.rs`). The
symptom of forgetting is a reply that starts a few tokens late, or
`ScriptedLogitProducer exhausted its scripted steps`.

**One caveat about the middle row that is worth knowing before trusting a
red or a green from it.** SwiftPM does not treat `libturbospark_ffi.a` as a
build input -- the `-L` path is an unsafe linker flag, which it passes
through without a dependency edge -- so a changed archive under unchanged
`.swift` files used to trigger no relink, and `swift test` would report on
the PREVIOUS build. `scripts/swift-lib.sh` touches both packages' Swift
sources after staging the archive for that reason. If the seam ever returns,
its symptom is a mutation check whose result never moves.

---

## Not supported

Stated so the omissions are decisions on the record rather than gaps.

- **Running a tool, or tool calls on a checkpoint whose markup is not
  native.** Offering tools and parsing native calls IS supported now (see
  "Tool calling"); executing a tool is the host's job, and
  `info.toolCalling.native == false` checkpoints need the host's own repair
  layer (the app's guardrails), which the server also has and this surface does
  not.
- **Logprobs, grammar-constrained or structured output, perplexity, and KV
  save/restore.** These do not exist in the Rust runtime (the server only warns
  on `response_format`), so they are not binding gaps.
- **A throughput guarantee for multiple concurrent sessions.** Each session
  has its own runner, which takes `&mut self` to decode, so one session
  serializes its own calls. `TurboSparkApp` does open a second session per
  model it attaches to the in-process server, so two sessions in one process
  work; what is not promised is that they run in parallel at full speed, and
  each pins gigabytes of weights and KV.
- **iOS.** The Metal kernels and the expert streamer's `pread` path have
  never been run there.
- **Intel Macs.** See above.

Prompt caching across turns is no longer on this list -- see "Generating"
above. It read "Each `generate` renders the whole conversation and prefills
it. A long chat re-reads its history every turn" until 2026-09-01, which was
true of the FFI even though `runtime::kv_prefix` had already shipped and
`crates/cli`'s `--chat` REPL had already opted into it: this crate's `open`
mirrored the CLI's single-shot `open_session` rather than its multi-turn
`chat.rs`, so `TurboSparkApp` -- a multi-chat app built on exactly one
long-lived session per loaded model -- re-prefilled its whole transcript
every message with the 11.6x prefill win sitting unreachable one file away.

### Deliberately not exposed, with the reason

Found by an audit of the Rust crates against this surface, and left out on
purpose. Each would need new evidence or an explicit decision to change.

- **`pull --force`, installing past a probe refusal.** The CLI's flag installs a
  model the probe says would not run here. The FFI refuses that by design; a
  host that wants to attempt it should say so explicitly in its own UI and
  call the CLI.
- **Hub search and trending** (`catalog::HubClient`). Nothing in the CLI uses
  it yet, its result types are not serializable, and it needs a detected
  machine profile; projecting it over the ABI without a live network to check
  against would be guesswork.
- **Image fit/admission** (`image::admit_image_budget`). It models the native
  image pipeline, which the app does not run. The app's memory tiers
  (`recommendedImageModelAliases`) are measured envelopes for the vendored MLX
  Z-Image path, and quoting one path's budget for the other's footprint passes
  off a different measurement as the app's own.
- **Negative prompt, guidance and step count for Z-Image Turbo.** Its contract
  is nine steps at guidance 0 (`docs/ZIMAGE_TURBO.md`), and the native
  engine's request validation rejects anything else (`crates/image/src/
  runtime.rs`). A negative prompt acts through classifier-free guidance, which
  is off at guidance 0, so a control for it on the Turbo models the app ships
  would not do anything. The families that do run guidance (4, in
  `docs/IMAGE_GENERATION.md`) are where it could matter, and the app does not
  generate with them.
- **Cancellable or cached audio open verification.** `ts_audio_session_open`
  hashes every byte of a managed model. Caching the result would weaken an
  integrity check, and the cancel flag it could take is only consulted between
  files, not inside a large file's hash.
- **VAD, diarization, Moonshine, enhancement and separation** have Rust model
  code but no `Engine` variant in `runtime::native_audio`, so there is no
  session to open (`AudioTask.isRunnable` reports it). They also need catalog
  pins (revisions and hashes) that cannot be invented.
- **Determinate music progress.** `Music3Runner::generate_text_with_progress`
  has the counters, but it does not take the cancellation handle that
  `generate_text_cancellable` does, so wiring it needs a combined method in
  `runtime/src/music3.rs` and a call-site change in `native_audio.rs`.
- **TypeSafe/OpenKind** stays a separate process with no link to this FFI.
- **Native image ABI (`ts_image_*`).** Kept for C hosts and the tests; the app
  generates through its vendored MLX pipeline. Its request validation accepts
  only steps 9 and guidance 0 (`crates/image/src/runtime.rs`), so it exposes
  five fields on purpose.

---

## See also

- [`crates/ffi/AGENTS.md`](../crates/ffi/AGENTS.md): the crate's own
  gotchas, including why the cancel flag sits where it does and why the
  header is hand-written
- [`crates/ffi/include/turbospark.h`](../crates/ffi/include/turbospark.h):
  the canonical contract
- [`swift/docs/KEYBOARD_SHORTCUTS.md`](../swift/docs/KEYBOARD_SHORTCUTS.md): keyboard shortcuts and accessibility navigation reference for the SwiftUI app
- [`docs/MODELS.md`](MODELS.md): the catalog, the probe, and what `pull`
  does
- [`docs/GTURBO.md`](GTURBO.md): the install format a session opens
- [`docs/LOAD_GUARD.md`](LOAD_GUARD.md): the tiers behind `loadGuard`, the
  AutoFit floor, and what the pressure watcher does and deliberately does not
  do
- [`docs/BENCHMARKS.md`](BENCHMARKS.md): the frozen throughput, memory and
  quality numbers


### Server dashboard and transport diagnostics

The app's Server section keeps bind controls, model attachment, traffic charts,
and a resizable console visible even before startup. Its right inspector holds
server configuration and saved recipes instead of the chat sampling controls.
Recipes retain host, port, attached install paths, and context length in the
existing per-user settings file. Loading a recipe starts a stopped server and
attaches its models sequentially. Missing installs are reported. Keys and text
previews are excluded from recipes.

`ServerOptions.host` accepts loopback addresses or a Tailscale IPv4 address in
`100.64.0.0/10`. The default stays `127.0.0.1`; Tailscale binding also requires
an API key. LAN, wildcard, invalid, and occupied addresses fail visibly. These
are in-process options, not new CLI flags. `ServerInfo.host` reports the bind
address.

`ServerInfo.traffic` reports consumed request-body bytes and emitted response-body
bytes. It excludes HTTP headers, TCP overhead and traffic from other processes.
The dashboard samples byte deltas over elapsed wall time using the existing
2 Hz poll, retaining 120 points. Missing counters render as unavailable. Memory
uses the current Mach physical footprint for the whole app, including chat and
server, against total installed physical memory. It is not a per-model allocation
measurement or the process peak.

Optional `captureText` retains raw body fragments for debugging, including JSON
and SSE framing. It is off by default, applies at startup, never persists, and
captures at most 256 bytes per fragment from the first 4 KiB of each body, keeping
64 fragments. Fragments may truncate UTF-8 or JSON and are explicitly previews,
not a replayable request archive. No request headers enter this buffer. The body
adapter preserves frames, trailers, errors and backpressure without buffering a
whole request or response. Copy diagnostics omits keys and captured text; the
Live text tab has a separate explicit copy action.

Console pause freezes a local snapshot while polling and charts continue. Stopping
the server retains request logs and graph history for inspection. The menu-bar
popover displays the same bounded bandwidth and memory history without another
timer.

New localized UI strings: IP address; Port; Stop the server to edit. Leave the
port blank for automatic assignment.; HTTP body bandwidth; App + server memory;
Optional raw HTTP text. Kept in memory, truncated, cleared on restart.; Live text;
Activity; Capture text previews; Copy diagnostics.
