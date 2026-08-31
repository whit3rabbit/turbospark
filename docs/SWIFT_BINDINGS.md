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
info.steering.mode      // "ablate", "add", "clamp", "renorm" or nil
info.steering.scale     // active scale multiplier or nil
info.steering.summary   // human-readable one-line description or nil
info.speculation.block  // the RESOLVED block, or nil when off
info.speculation.drafter// .mtp | .dflash, non-nil exactly when block is
info.speculation.reason // why it is off, when you might expect otherwise
info.specialTokens.bosId        // e.g. 1 or nil
info.specialTokens.eosId        // e.g. 2 or nil
info.specialTokens.endOfTurnId  // e.g. 151645 or nil
info.specialTokens.stopTokenIds // [151643, 151645]
info.specialTokens.thinkStartId // e.g. 151648 or nil
info.specialTokens.thinkEndId   // e.g. 151649 or nil
```

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
options.seed = 20260721           // nil for nondeterministic
options.stop = ["\n\n---"]
options.stopTokens = [151643, 151645] // numerical stop token IDs
options.reasoning = .off
```

The defaults are the CLI's, so sending nothing gives what
`turbospark-check` gives with no flags. `maxNewTokens` is clamped rather than
refused when the conversation is long, so a full context generates into
whatever room is left instead of failing.

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

### Tokenization & detokenization

```swift
// Encode raw text to integer token IDs:
let tokenIDs = try await session.tokenize("Hello, world!", addSpecialTokens: false)
print("Token IDs: \(tokenIDs)")

// Decode token IDs back to text:
let reconstructed = try await session.detokenize(tokenIDs, skipSpecialTokens: false)
assert(reconstructed == "Hello, world!")
```

Exposes direct access to the model's tokenizer for token visualizers, token chip
highlighters, token-level editing, and span calculations in chat interfaces.

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
let server = try TurboSparkServer.start(options: ServerOptions(apiKey: "sk-local"))

let id = try server.attach(session)      // "gemma4.gturbo" -- the install's own name
print(try server.info().baseURL!)        // http://127.0.0.1:53411

try server.detach(modelId: id)           // stops serving it, releases the engine
server.stop()                            // stops serving everything
```

Routes: OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
`/v1/models`), Anthropic (`/v1/messages`, `/v1/messages/count_tokens`),
Ollama (`/api/tags`, `/api/version`, `/api/show`, `/api/chat`,
`/api/generate`), and `GET /health`. There is no `/v1/embeddings`: this
engine has no embedding path, and a route that 404s is worse than an absent
one.

**Which model serves a request.** An exact `model` id wins. Failing that, if
exactly ONE model is attached it serves the request whatever name was asked
for -- which is what lets a client sending its own default (Claude Code sends
`claude-sonnet-4-6`) work with no configuration. With two or more attached
and no match, the request is a 404 naming what is available.

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

**Unauthenticated does not mean private.** The socket is loopback, which
keeps it off the network and reachable by every process on this machine.
`apiKey` is the only access control there is, and an empty or
whitespace-only key means none at all -- `info().authEnabled` is what
actually happened.

### Watching it

```swift
let batch = server.poll()          // drains; each event is returned once
for event in batch.events { ... }
if batch.dropped > 0 { /* say so */ }
```

Poll on a timer and append what you get. Events are `requestStarted`,
`requestRouted`, `generated`, `requestFinished`, `modelAttached`,
`modelDetached`, tied together by a request id, plus an `unknown(kind:)` case
so a newer engine's event does not fail the whole batch.

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

// Delete an installed model to recover disk space:
try TurboSparkCatalog.delete("gemma4")
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

**Warn before starting, not after failing.** The walk streams gigabytes
without writing the checkpoint to disk whole, and **it cannot resume**: a
failure restarts from the beginning. A user who does not know that will kill
it at 90% and try again. The first `.stage` event says so;
`swift/TurboSparkApp/Sources/TurboSparkApp/Installation/CatalogSheet.swift` puts the
warning above the button.

**Take the maximum of byte events, not the latest.** Ranged downloads are
split across connections, so byte progress arrives concurrently and out of
order. Using the last value makes the bar jump backwards.

---

## Errors

```swift
do {
    let session = try await TurboSparkSession(modelPath: path)
} catch let error as TurboSparkError {
    error.code      // .invalidArgument .open .generate .json .unsupportedPlatform .panic
    error.message   // a sentence, from the library
}
```

`.panic` means a panic was caught at the boundary. The process is intact and
the operation did not happen; it is a library bug rather than anything the
caller did, and it is worth reporting with the message attached.

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
| `ts_system_info_json(out)` | hardware RAM, chip, power, thermal status |
| `ts_session_count_tokens(s, messages, reasoning, out_count)` | evaluates exact prompt token count |
| `ts_session_render_prompt(s, messages, reasoning, out_prompt)` | formats conversation into raw prompt text |
| `ts_session_tokenize_json(s, text, add_special, out)` | tokenizes text into JSON array of token IDs |
| `ts_session_detokenize_json(s, tokens_json, skip_special, out)` | decodes token IDs into text string |
| `ts_session_count_text_tokens(s, text, add_special, out_count)` | evaluates raw text token count |
| `ts_session_fit_window_json(s, messages, reasoning, max_tokens, out)` | fits conversation into token budget |
| `ts_generate(s, messages, options, cb, ud, out)` | blocks for the turn |
| `ts_catalog_json(out)` | every platform |
| `ts_installed_json(out)` | every platform |
| `ts_model_delete(alias)` | delete installed model directory and forget row |
| `ts_recommend_json(context, options_json, out)` | rank curated models by hardware fit; `options_json` takes `loadGuard` and may be NULL |
| `ts_probe_json(repo, file, sidecar, out)` | header-only, no download |
| `ts_install_bytes_json(alias, out)` | cost before committing |
| `ts_install(alias, cb, ud, out)` | blocks for minutes; cannot resume |
| `ts_install_repo(repo, alias, file, sidecars, cb, ud, out)` | install arbitrary HF repository |

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
  "steeringGate": 0.0
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
  }
}
```

`steering.active` indicates whether a control vector is loaded and active.
A null `block` is the "is it off" test; `drafter` is non-null exactly when
`block` is, and `reason` is non-null only when a caller might have expected
it on.

`ts_generate` options and its result:

```json
{ "maxNewTokens": 512, "temperature": 0.2, "topK": 64, "topP": 0.95,
  "repetitionPenalty": 1.0, "seed": null, "stop": [], "stopTokens": [], "reasoning": "off" }
```

```json
{ "promptTokens": 21, "newTokens": 120, "prefillSeconds": 0.41,
  "decodeSeconds": 2.87, "stopReason": "maxTokens", "tokensPerSecond": 41.8,
  "content": "...", "reasoning": "" }
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

- **Tool calling.** `crates/server` has it on both endpoints;
  `StructuredAssistantDecoder` is constructed here with an empty tool
  allowlist, so a Harmony `commentary` body arrives as reasoning. Wiring it
  means a way to *run* a tool, which a binding cannot supply on its own.
- **Multiple concurrent sessions per process.** One runner per process is the
  engine's shape: it takes `&mut self` to decode, so calls serialize. Two
  sessions in one process is untested and each pins gigabytes.
- **iOS.** The Metal kernels and the expert streamer's `pread` path have
  never been run there.
- **Prompt caching across turns.** Each `generate` renders the whole
  conversation and prefills it. A long chat re-reads its history every turn.
- **Intel Macs.** See above.

---

## See also

- [`crates/ffi/CLAUDE.md`](../crates/ffi/CLAUDE.md): the crate's own
  gotchas, including why the cancel flag sits where it does and why the
  header is hand-written
- [`crates/ffi/include/turbospark.h`](../crates/ffi/include/turbospark.h):
  the canonical contract
- [`docs/KEYBOARD_SHORTCUTS.md`](KEYBOARD_SHORTCUTS.md): keyboard shortcuts and accessibility navigation reference for the SwiftUI app
- [`docs/MODELS.md`](MODELS.md): the catalog, the probe, and what `pull`
  does
- [`docs/GTURBO.md`](GTURBO.md): the install format a session opens
- [`docs/LOAD_GUARD.md`](LOAD_GUARD.md): the tiers behind `loadGuard`, the
  AutoFit floor, and what the pressure watcher does and deliberately does not
  do
- [`docs/BENCHMARKS.md`](BENCHMARKS.md): the frozen throughput, memory and
  quality numbers

