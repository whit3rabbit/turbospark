# Swift bindings: driving the engine from a native app

`crates/ffi` exposes the inference engine as a C ABI, and
`swift/TurboSpark` wraps that in an idiomatic Swift package. A SwiftUI app
opens a `.gturbo` install, streams tokens, stops mid-generation, installs
models, and reads engine telemetry, all in-process.

`swift/TurboSparkDemo` is a small chat app that exercises every one of those.
It is deliberately minimal: it exists to verify the binding, not to be a
product.

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
everybody else. `swift/TurboSparkDemo/Package.swift` shows the shape:

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

let session = try await TurboSparkSession(modelPath: "gemma4", options: options)
```

Everything defaults to automatic, which is what a GUI should want. Two
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

### Reading what you actually got

```swift
let info = session.info
info.maxContext         // the RESOLVED window
info.expertCacheSlots   // the RESOLVED slot count
info.trainedContext     // the checkpoint's own, or nil
info.pastTrainedContext // true when the window exceeds it
info.family             // "gemma4", "qwen36", "llama", ...
info.reasoningSupport   // .level | .toggleOnly | .none
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
rejects `.high` and its top setting is `.xhigh`; Harmony and Muse Glimmer
accept `.high`. A level a template rejects throws an error naming it, rather
than being silently dropped. Check `info.reasoningSupport` first:

| value | meaning | what a UI should do |
|---|---|---|
| `.level` | the template takes an effort level | enable the picker |
| `.toggleOnly` | thinking turns on, the level is dropped | grey out the levels, keep the toggle |
| `.none` | no chat template at all | disable it; asking throws |

### Telemetry

```swift
let phases = try await session.phases()
phases.calls              // forward passes served
phases.totalMsPerCall
phases.expertIoMs         // expert streaming
phases.gpuWaitMs
phases.expertHitRate      // nil before anything has been requested

TurboSparkSession.peakFootprintBytes   // process-wide, or nil
```

Two caveats, both of which make a naive status panel wrong:

**The phase counters are cumulative over every forward pass, prefill
included.** A per-call number is an average across the whole context range,
not a number at the current context. To get a figure *at* a context,
difference two runs.

**They cover the inside of the forward pass only.** The sampler and the
detokenizer run after it returns and appear in none of the buckets, so the
phase total will not add up to wall-clock decode time.

`peakFootprintBytes` is the same mach counter every published memory figure
for this engine uses, so your number and the memory oracle's agree. What it
*counts* differs by install shape: a streamed MoE model's mapped weights are
counted, a dense model's are not. Read it beside `info.maxContext` rather
than comparing across models.

---

## Model management

Available on every platform, including ones that cannot then run a model. The
artifact is the same either way.

```swift
let rows = try TurboSparkCatalog.available()      // curated table, with `installed`
let mine = try TurboSparkCatalog.installed()      // what is in ~/.turbospark
let cost = try TurboSparkCatalog.cost(of: "gemma4")
let report = try TurboSparkCatalog.probe(repo: "Qwen/Qwen3-30B-A3B-GGUF",
                                         file: "Qwen3-30B-A3B-Q4_K_M.gguf")
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
```

Two things a progress UI has to get right:

**Warn before starting, not after failing.** The walk streams gigabytes
without writing the checkpoint to disk whole, and **it cannot resume**: a
failure restarts from the beginning. A user who does not know that will kill
it at 90% and try again. The first `.stage` event says so;
`swift/TurboSparkDemo/Sources/TurboSparkDemo/InstallSheet.swift` puts the
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
| `ts_session_info_json(s, out)` | resolved window, slots, family, dialect |
| `ts_session_phases_json(s, out)` | decode phase breakdown |
| `ts_peak_footprint_bytes()` | process-wide, 0 if unavailable |
| `ts_generate(s, messages, options, cb, ud, out)` | blocks for the turn |
| `ts_catalog_json(out)` | every platform |
| `ts_installed_json(out)` | every platform |
| `ts_probe_json(repo, file, sidecar, out)` | header-only, no download |
| `ts_install_bytes_json(alias, out)` | cost before committing |
| `ts_install(alias, cb, ud, out)` | blocks for minutes; cannot resume |

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
  "maxTokensPerSec": 30
}
```

`maxContext` and `expertCacheSlots` accept a number, the string `"auto"`,
`null`, or absence; all four spellings of automatic mean the same thing,
because a caller's encoder may produce any of them. Any *other* string is an
error rather than a silent fallback.

`ts_generate` options and its result:

```json
{ "maxNewTokens": 512, "temperature": 0.2, "topK": 64, "topP": 0.95,
  "repetitionPenalty": 1.0, "seed": null, "stop": [], "reasoning": "off" }
```

```json
{ "promptTokens": 21, "newTokens": 120, "prefillSeconds": 0.41,
  "decodeSeconds": 2.87, "stopReason": "maxTokens", "tokensPerSecond": 41.8,
  "content": "...", "reasoning": "" }
```

`tokensPerSecond` is `null` when no decoding happened, so nothing can plot a
rate that was never measured. `stopReason` is one of `endOfTurn`,
`toolCalls`, `eos`, `stopString`, `maxTokens`, `cancelled`.

---

## Building and shipping

```bash
make swift-lib                                     # required first
make swift-test                                    # ABI checks, no model
make swift-test-real MODEL=~/models/gemma4.gturbo  # end to end, minutes
make swift-demo                                    # the chat app
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
- [`docs/MODELS.md`](MODELS.md): the catalog, the probe, and what `pull`
  does
- [`docs/GTURBO.md`](GTURBO.md): the install format a session opens
- [`docs/BENCHMARKS.md`](BENCHMARKS.md): the frozen throughput, memory and
  quality numbers
