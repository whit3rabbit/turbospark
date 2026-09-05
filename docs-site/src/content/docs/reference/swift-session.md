---
title: Sessions and Generation (Swift)
description: TurboSparkSession, open and generate options, chat messages, session info, generation events, the phase report, and TurboSparkError, from the TurboSpark Swift package.
diataxisType: reference
---

<!-- generated: swift lane, signal: swift/TurboSpark/Package.swift (also swift/TurboSparkApp/Package.swift) -->

Reference for the `TurboSpark` SwiftPM package (`swift/TurboSpark`), the
async/await Swift binding over `crates/ffi`'s C ABI. Everything here is the
tracked public surface of `Sources/TurboSpark/`, read file by file: each
signature is copied verbatim from source and the prose under it is the symbol's
own DocC comment, condensed only where the comment is long. The Swift-DocC
plugin is not wired into this package, so this page is a source parse, not
`generate-documentation` output.

Scope note: the sibling package `TurboSparkApp` (`swift/TurboSparkApp`) is the
SwiftUI executable and is not covered here. Its views are declared `public`
only as a module-internal convention; the consumable surface of the repository
is the library below. The generator (`swift doc` / `swift package
generate-documentation`) is unavailable on this toolchain: there is no
`swift-doc` subcommand and no `swift-docc-plugin` dependency, so `swift
package generate-documentation` fails with "Unknown subcommand".

Driving a model is one class. Open it, stream turns from it, and read back
what the engine resolved. Both options structs are plain `Encodable` bags
whose `nil` means "let the engine decide"; `SessionInfo` is where the
resolution lands.

## TurboSparkSession

Source: `swift/TurboSpark/Sources/TurboSpark/TurboSparkSession.swift`.

```swift
public final class TurboSparkSession: @unchecked Sendable
```

One opened model.

**A class with a serial queue rather than an `actor`, and that is the point
rather than a preference.** `cancel()` has to be callable from the main thread
WHILE a turn is running, and an actor method cannot be: it would suspend
behind the in-flight generation, so the Stop button would only take effect
once the model had finished on its own. The C layer is built the same way for
the same reason (its cancel flag lives outside the engine mutex), and this
type would throw that away if it wrapped the handle in an actor.

Everything except `cancel()` runs on `queue`, which is serial, so the
library's one-generation-at-a-time contract is met by construction.
`@unchecked Sendable` is justified rather than assumed: the only stored
mutable state is the opaque handle, every use of it is confined to `queue`,
and `cancel()`'s single call is documented thread-safe by the C header and
implemented as an atomic store.

### Properties

```swift
public let info: SessionInfo
```

Everything resolved at open. Read the `maxContext` and `expertCacheSlots`
here rather than what you asked for: under automatic sizing you asked for
nothing, and these are what the KV cache was actually allocated at.

```swift
public static var peakFootprintBytes: UInt64?
```

This process's peak physical footprint in bytes, or nil where the counter is
unavailable. Process-wide rather than per session, and what it counts differs
by install shape: a streamed MoE model's mapped weights ARE counted, a dense
model's are not. Read it beside `info.maxContext` rather than comparing it
across models.

```swift
public static var systemTelemetry: SystemTelemetry?
```

Hardware and power telemetry for this machine, or nil where unavailable.

### Open

```swift
public init(modelPath: String, options: OpenOptions = OpenOptions()) async throws
```

Opens a model directory or an installed alias. A leading `~` is expanded,
because a Mac user writing a path by hand writes one and the C layer takes its
argument literally. An ALIAS is passed through untouched. Expensive: it maps
gigabytes and compiles Metal pipelines. Open once and keep the session. Runs
off the calling thread, so it is safe to `await` this from a SwiftUI view.

The initializer reads `ts_session_info_json` BEFORE any stored property is
assigned, and closes the C handle by hand when that read fails. A class whose
`init` throws part-way through never runs its `deinit`, so leaving cleanup to
it strands a whole open session (mapped weights, KV cache, compiled Metal
pipelines) for the life of the process.

`deinit` calls `ts_session_close`, and the C header forbids closing while a
generation is in flight. What keeps that true is `generate`'s worker capturing
`self` STRONGLY: the session cannot reach zero references while a turn is on
the queue, so `deinit` cannot run beside `ts_generate`.

### Generation

```swift
public func cancel()
```

Asks the running turn to stop. `nonisolated` in spirit and in fact: safe from
any thread, never blocks, and never queues behind the generation it is
stopping. A Stop pressed when nothing is running is discarded rather than
cancelling the next turn.

```swift
public func generate(
    _ messages: [ChatMessage],
    options: GenerateOptions = GenerateOptions()
) -> AsyncThrowingStream<GenerationEvent, Error>
```

Generates one assistant turn, streaming events as they arrive. The stream ends
with `.finished(result)`. Cancelling the consuming `Task` cancels the
generation too, so `for try await` inside a SwiftUI `.task` stops the model
when the view goes away. Accumulate `.content` as the assistant turn;
`.reasoning` is for display only (see `GenerationResult.reasoning`).

Dropping the stream without cancelling asks the model to stop via
`onTermination`, which is `[weak self]` deliberately: the handler must ASK the
session to stop, never extend its life past the consumer that owns it.

### Server

```swift
public func startServer(options: ServerOptions = ServerOptions()) async throws
    -> TurboSparkServer
```

Starts an in-process HTTP server sharing THIS session's engine, and returns a
handle to it. **THE SERVER OUTLIVES THIS `TurboSparkSession` IF YOU LET IT.**
It holds its own reference to the underlying engine on the Rust side, so this
session going out of scope after this call frees only this Swift object: the
model stays resident and the server keeps serving it until
`TurboSparkServer.stop()` (or its `deinit`) releases the last reference.

Serves the same OpenAI/Anthropic-compatible routes the standalone
`turbospark-server` binary does, EXCEPT vision and the standalone binary's
tool-call guardrails: a session opened through this type carries no vision
wiring, so an image request the server receives is refused by name rather
than silently dropped.

### Introspection and prompt tools

```swift
public func phases() async throws -> PhaseReport
```

The decode phase breakdown. Cheap enough to poll for a status panel.
Cumulative over every forward pass this session has served, prefill included.

```swift
public func countTokens(
    _ messages: [ChatMessage],
    reasoning: GenerateOptions.Reasoning = .off
) async throws -> Int
```

Evaluates exact prompt token count for `messages` using this session's chat
template and tokenizer, without running generation.

```swift
public func countTokens(
    in text: String,
    addSpecialTokens: Bool = false
) async throws -> Int
```

Evaluates the token count of a raw text string using this session's
tokenizer.

```swift
public func renderPrompt(
    _ messages: [ChatMessage],
    reasoning: GenerateOptions.Reasoning = .off
) async throws -> String
```

Formats a conversation transcript into raw prompt text using this session's
chat template and reasoning effort setting.

```swift
public func tokenize(
    _ text: String,
    addSpecialTokens: Bool = false
) async throws -> [Int32]
```

Tokenizes raw text into an array of integer token IDs using this session's
tokenizer.

```swift
public func detokenize(
    _ tokens: [Int32],
    skipSpecialTokens: Bool = false
) async throws -> String
```

Detokenizes an array of integer token IDs into text using this session's
tokenizer.

```swift
public func fitWindow(
    _ messages: [ChatMessage],
    maxTokens: UInt32? = nil,
    reasoning: GenerateOptions.Reasoning = .off
) async throws -> WindowFitOutcome
```

Fits a conversation transcript into a context token budget by pruning older
turns (preserving optional leading system instruction and newest user turn).
`maxTokens`: the token budget limit. When `nil`, defaults to the session's
resolved `maxContext`.

## OpenOptions

Source: `swift/TurboSpark/Sources/TurboSpark/Options.swift`.

```swift
public struct OpenOptions: Encodable, Sendable
```

How a session is opened. `nil` everywhere means fully automatic, which is what
the CLI and the server default to.

### Sizing

```swift
public enum Sizing: Encodable, Sendable
```

Sizing mode for context window or expert slot caching.

- `case auto` - Automatic sizing resolved by the engine.
- `case fixed(UInt32)` - Fixed capacity in slots or tokens.

### PowerProfile

```swift
public enum PowerProfile: String, Encodable, Sendable
```

Power and thermal management profile.

- `case performance` - Maximum GPU performance and frequency.
- `case balanced` - Balanced power and thermal profile.
- `case efficiency` - Energy-saving efficiency profile.

### Speculation

```swift
public enum Speculation: Encodable, Sendable
```

Whether and how far this session drafts ahead. Settled when the model is
OPENED, because that is where the drafter's state is allocated, and not per
turn. `.block` throws from `TurboSparkSession.init` when the install cannot
serve it, while `.auto` opens and explains itself in
`SessionInfo.speculation.reason`.

- `case auto` - Automatic block size resolution.
- `case off` - Speculative decoding disabled.
- `case block(UInt32)` - A named block size. The engine accepts 1 through 15;
  anything else throws, naming the range.

### SpeculativeDrafter

```swift
public enum SpeculativeDrafter: String, Encodable, Sendable
```

Which drafter `speculation` drives. The two are ALTERNATIVES rather than a
spectrum: the checkpoint's own MTP head drafts a token at a time, DFlash2
proposes a whole block in one pass.

- `case auto` - Whichever the install carries, but `auto` ENABLES an MTP head
  and only REPORTS a DFlash2 one, which is measured rather than stylistic:
  DFlash2 reads 1.43-1.50x on code and math and 0.96x throughput at +17.4%
  J/token on PROSE.
- `case mtp` - Multi-token prediction head drafter.
- `case dflash` - DFlash2 diffusion block drafter.

### LoadGuard

```swift
public enum LoadGuard: Encodable, Sendable
```

How much of the machine may be committed to loading a model. `relaxed` is the
default and is what this binding did before the option existed; every
published footprint figure for this engine was measured under it. See
`docs/LOAD_GUARD.md`.

**If you also call `TurboSparkCatalog.recommend`, pass it the SAME value.**
The ranking and the loader's refusal share one memory budget by construction;
recommending under one tier while opening under another promises a fit the
loader then refuses.

- `case off` - No memory precautions: reserves nothing and, more importantly,
  declines to REFUSE. A window too large for the machine becomes the Metal
  allocation failure you asked for rather than an error.
- `case relaxed` - The shipped behaviour: a 4 GiB reserve and a quarter of the
  rest.
- `case balanced` - A larger reserve and a smaller share, so a model and a
  browser can share the machine.
- `case strict` - Larger still, for a machine running work that must not be
  interrupted.
- `case custom(UInt64)` - `relaxed`'s shares plus an absolute ceiling, in
  BYTES, on what the engine ALLOCATES (slot cache plus KV). Not on the
  install's size: a large model streaming its experts from disk is what this
  engine is for, and a cap read against the install would refuse a 13 GB model
  on a 16 GB machine that runs it fine.

### SteeringMode

```swift
public enum SteeringMode: String, Encodable, Sendable
```

The edit applied along a control vector.

- `case ablate` - Suppress activations along the control vector.
- `case add` - Additive vector injection.
- `case clamp` - Clamp activations along the control vector.
- `case renorm` - Renormalize activations following modification.

### Fields

| Signature | Doc text |
|---|---|
| `public var maxContext: Sizing?` | Maximum context window sizing. |
| `public var expertCacheSlots: Sizing?` | Expert cache slot sizing for MoE models. |
| `public var powerProfile: PowerProfile?` | `nil` ASKS THE OS, so Low Power Mode selects efficiency. Name one explicitly when measuring anything. |
| `public var maxTokensPerSec: Double?` | Maximum throughput generation rate cap in tokens per second. |
| `public var loadGuard: LoadGuard?` | `nil` means `.relaxed`, which is what shipped before this existed. |
| `public var minAutoContext: UInt32?` | Refuse to open when `maxContext` is `.auto` and resolves below this many tokens. `nil` and 0 both mean no floor. **Constrains AUTOMATIC sizing only.** It says nothing about an explicit `.fixed(2048)`: a caller naming a number has decided how to spend their own machine. |
| `public var speculation: Speculation?` | `nil` means `.auto`, which is what the CLI and the server default to. |
| `public var speculativeDrafter: SpeculativeDrafter?` | `nil` means `.auto`. |
| `public var steering: String?` | Path to a .gguf control vector (llama.cpp layout). `nil` disables steering. |
| `public var steeringMode: SteeringMode?` | `nil` uses the vector's declared mode or `.ablate`. |
| `public var steeringScale: Double?` | Multiplier on edit strength (default 1.0; 0.0 is identity). |
| `public var steeringLayers: String?` | Layer range to steer, "START:END" inclusive and 0-based (default all). |
| `public var steeringTarget: Double?` | Target coefficient for `.clamp` mode (default 0.0). |
| `public var steeringGate: Double?` | Activation magnitude threshold to trigger the edit (default 0.0). |

```swift
public init(
    maxContext: Sizing? = nil,
    expertCacheSlots: Sizing? = nil,
    powerProfile: PowerProfile? = nil,
    maxTokensPerSec: Double? = nil,
    loadGuard: LoadGuard? = nil,
    minAutoContext: UInt32? = nil,
    speculation: Speculation? = nil,
    speculativeDrafter: SpeculativeDrafter? = nil,
    steering: String? = nil,
    steeringMode: SteeringMode? = nil,
    steeringScale: Double? = nil,
    steeringLayers: String? = nil,
    steeringTarget: Double? = nil,
    steeringGate: Double? = nil
)
```

Creates options for opening a model session.

## GenerateOptions

Source: `swift/TurboSpark/Sources/TurboSpark/Options.swift`.

```swift
public struct GenerateOptions: Encodable, Sendable
```

How one turn is generated. The defaults are the CLI's.

### Reasoning

```swift
public enum Reasoning: String, Codable, Sendable, CaseIterable, Identifiable
```

The accepted set is the CHECKPOINT'S, not this library's. Qwen 3.8 rejects
`.high` and its top setting is `.xhigh`; Harmony and Muse Glimmer accept
`.high`. A level a template rejects throws, naming it.

- `case off` - Reasoning turned off.
- `case low` - Low reasoning effort.
- `case medium` - Medium reasoning effort.
- `case high` - High reasoning effort.
- `case xhigh` - Extra high reasoning effort.

Computed members (no doc comment in source): `public var id: String`,
`public var label: String` (returns "Off", "Low", "Medium", "High", "Extra
High"), `public var descriptionText: String`.

### Fields

| Signature | Doc text |
|---|---|
| `public var maxNewTokens: UInt32 = 512` | Maximum new tokens to emit. |
| `public var temperature: Double = 0.2` | Sampling temperature (0 for greedy). |
| `public var topK: UInt32 = 64` | Top-K sampling cutoff. |
| `public var topP: Double = 0.95` | Nucleus top-P probability cutoff. |
| `public var repetitionPenalty: Double = 1.0` | Multiplicative repetition penalty. |
| `public var seed: UInt64?` | Deterministic RNG seed. |
| `public var stop: [String] = []` | Custom stop sequence strings. |
| `public var stopTokens: [UInt32] = []` | Explicit stop token IDs. |
| `public var reasoning: Reasoning = .off` | Reasoning effort level. |

```swift
public init()
```

Creates default generation options.

## ChatMessage, ChatImage, WindowFitOutcome

Source: `swift/TurboSpark/Sources/TurboSpark/ChatTypes.swift`. The Rust side
emits camelCase, so every type here decodes with no `CodingKeys` and the two
definitions cannot drift over a spelling.

```swift
public enum ChatImage: Codable, Sendable, Equatable
```

One image attached to a message. A placeholder as far as the prompt is
concerned: the template renders one marker per image and the tower's rows are
injected at that marker, so what travels here is only where the pixels can be
found.

- `case path(String)` - A file the engine process can read. The engine and
  the app share an address space, so this costs no copy of the bytes.
- `case base64(String)` - A bare base64 payload, or a full
  `data:<media>;base64,<data>` URL.

```swift
public struct ChatMessage: Codable, Sendable, Equatable
```

A single message in a conversation.

```swift
public enum Role: String, Codable, Sendable
```

The sender role of the message: `case system, developer, user, assistant,
tool` (no per-case doc comments in source).

| Signature | Doc text |
|---|---|
| `public var role: Role` | The role of the message sender. |
| `public var content: String` | The message text content. |
| `public var images: [ChatImage]` | Images on this message, in order. **Empty is the whole of the compatibility story.** With no images this type encodes `content` as a bare string, byte for byte what this binding sent before images existed; only a message that carries one switches to the ordered-parts shape. Gate on `SessionInfo.vision.active` before offering a way to fill this: an install can carry a tower and still refuse every image. |

```swift
public init(role: Role, content: String, images: [ChatImage] = [])
```

Creates a new chat message with the given role and content.

Convenience factories, one doc comment each: `public static func system(_
content: String) -> ChatMessage`, `public static func developer(_ content:
String) -> ChatMessage`, `public static func user(_ content: String) ->
ChatMessage`, `public static func assistant(_ content: String) ->
ChatMessage`, `public static func tool(_ content: String) -> ChatMessage`.

`encode(to:)` is documented with the rule that governs it: **IMAGES ARE
PREPENDED, NOT APPENDED, and that is matched to the reference rather than
chosen.** `apply_chat_template(processor, config, question, num_images=1)`
builds `[image, text]`, so the marker run comes first and the question
follows. Appending would move every mRoPE position past the image and produce
a different prompt for the same request. `init(from:)` accepts either wire
shape, so a value that made a round trip through the engine comes back equal
to what went in.

```swift
public struct WindowFitOutcome: Codable, Sendable, Equatable
```

The result of fitting a conversation into a context window budget.

| Signature | Doc text |
|---|---|
| `public let retained: [ChatMessage]` | The messages retained after pruning older turns. |
| `public let measuredTokens: Int` | The token count of the rendered retained messages. |
| `public let removedTurnCount: Int` | Number of older turns removed to fit the budget. |
| `public let hasRoomForGeneration: Bool` | Whether there is room remaining for generation within the budget. |

## SessionInfo

Source: `swift/TurboSpark/Sources/TurboSpark/SessionTypes.swift`.

```swift
public struct SessionInfo: Decodable, Sendable, Equatable
```

Everything resolved when the model was opened.

```swift
public enum ReasoningSupport: String, Decodable, Sendable
```

What KIND of reasoning control is meaningful (no doc comment on the enum
itself in source). Cases: `level` - the template takes a level;
`toggleOnly` - the template can only turn thinking ON, a level is dropped,
grey out the levels rather than hiding the toggle; `none` - no template at
all, asking for a level throws.

| Signature | Doc text |
|---|---|
| `public let maxContext: UInt32` | The RESOLVED window, not what was asked for. |
| `public let pastTrainedContext: Bool` | Above the trained context the model still runs; quality degrades. |
| `public let expertCacheSlots: Int` | The RESOLVED slot count. No throughput or footprint figure is readable without it. |
| `public let reasoningSupport: ReasoningSupport` | What KIND of reasoning control is meaningful. See `reasoningLevels` for what to put in it. |
| `public let reasoningLevels: [String]` | The spellings this checkpoint's own template can express, ascending, always opening at `"off"`. Prefer the typed `reasoningEfforts`. Stored as strings deliberately: `SessionInfo` is decoded inside `TurboSparkSession.init`, whose failure path has to close the C handle by hand, so a sixth level added to the engine one day must not take down every session open on this side. A field-NAME drift still fails the decode; an unknown VALUE is dropped. |
| `public let steering: Steering` | What directional steering resolved to for this session. |
| `public let speculation: Speculation` | What speculative decoding resolved to for this session. |
| `public let vision: Vision` | Whether this session would accept an image. |
| `public let specialTokens: SpecialTokens` | Special token identifiers for tokenizer inspection. |

No doc comment in source: `public let modelPath: String`, `public let family:
String`, `public let trainedContext: UInt32?`, `public let vocabSize: Int`,
`public let dialect: String`.

```swift
public var reasoningEfforts: [GenerateOptions.Reasoning] { get }
```

The reasoning levels this checkpoint accepts. **BUILD A PICKER FROM THIS AND
FROM NOTHING ELSE.** The set belongs to the checkpoint and cannot be derived
from the family: Qwen 3.8 answers `[.off, .low, .medium, .xhigh]` and RAISES
on `.high`, where gpt-oss and Muse Glimmer answer `[.off, .low, .medium,
.high]`. Sending a level absent from here fails the turn with the template's
own error, so a menu offering all five is a menu with a broken entry in it.
Levels rendering the same prompt are already collapsed by the engine, so a
`.toggleOnly` checkpoint answers exactly two. Its second entry is `.low` BY
POSITION and is not a label: read `reasoningSupport` and present that case as
an on/off switch.

### Nested types

`Vision` - whether an image sent to this session would actually be SERVED.
**`active` is not "does this family have a tower".** An install can carry one
and still refuse every image: the pixel budget is read from the checkpoint's
own `preprocessor_config.json` and has no default worth falling back to, so an
install streamed without that sidecar reports `active == false` with the
reason. Gate an attach control on THIS, or the control promises work the
engine then declines. Fields: `public let active: Bool` (true when this
session would encode and inject an image), `public let imageTokenId: Int32?`
(the `<|image_pad|>` id, or nil when inactive), `public let reason: String?`
(why images are refused, non-nil exactly when the install has a tower and
`active` is false).

`SpecialTokens` - special token identifiers for tokenizer introspection.
Fields `bosId`, `eosId`, `padId`, `endOfTurnId`, `stopTokenIds: [Int32]`,
`thinkStartId`, `thinkEndId` (all `public let`, none doc-commented in
source).

`Steering` - the session's resolved directional steering, reported once.
Fields: `active: Bool` (true when a control vector is active on this
session), `mode: String?` (`ablate` / `add` / `clamp` / `renorm`, present
only when active), `scale: Double?` (active scale multiplier, present only
when active), `summary: String?` (human-readable one-line description, or nil
when inactive).

`Speculation` - the session's resolved speculative decoding, reported once.
Its `Drafter` enum has `mtp` (the checkpoint's own multi-token-prediction
head, a token at a time) and `dflash` (the DFlash2 block-diffusion drafter, a
whole block per pass). Fields: `public let block: Int?` - how many tokens a
round proposes, or nil when this session does not draft ahead. THIS is the
"is it on" test; there is no separate flag that could disagree with it.
**Non-nil is a statement about the SESSION, not about the next turn.**
Acceptance is exact only at temperature 0, so a sampled turn decodes
sequentially whatever this says; send `GenerateOptions.temperature = 0` to
speculate. `public let drafter: Drafter?` - non-nil exactly when `block` is;
two drafters serve one family with different shapes and different measured
optima, so a throughput figure is unreadable without knowing which ran.
`public let reason: String?` - why speculation is off, when the caller might
have expected it on. Nil both when `.off` was asked for and when it is on.

## GenerationEvent, GenerationResult, PhaseReport

Source: `swift/TurboSpark/Sources/TurboSpark/GenerationTypes.swift`.

```swift
public enum GenerationEvent: Sendable, Equatable
```

One streamed event during generation.

- `case prefill(done: Int, total: Int)` - Prefill progress update with
  completed and total prompt tokens.
- `case content(String)` - Incremental generated assistant text content.
- `case reasoning(String)` - Incremental reasoning or thinking content.
- `case finished(GenerationResult)` - Generation completion event with final
  result.

```swift
public struct GenerationResult: Decodable, Sendable, Equatable
```

What a finished turn reports.

```swift
public enum StopReason: String, Decodable, Sendable
```

The condition that terminated text generation.

- `case endOfTurn` - Reached model end of turn token.
- `case toolCalls` - Generated a tool invocation block.
- `case eos` - Reached end of sequence.
- `case stopString` - Matched a stop string sequence.
- `case maxTokens` - Reached maximum token generation budget.
- `case cancelled` - The caller pressed Stop. The partial turn in `content`
  is valid and the conversation can continue from it.

| Signature | Doc text |
|---|---|
| `public let promptTokens: Int` | Number of tokens in the prompt prefix. |
| `public let newTokens: Int` | Number of new tokens generated. |
| `public let reusedPrefixTokens: Int` | How many of `promptTokens` continued from the previous turn's KV instead of being re-prefilled. Zero on a session's first turn, on one where the render diverged anywhere, or on a family this cannot help (recurrent state, a sliding-window ring past its slack). Never an error, just a full prefill that turn. Decoded with a default so a binding built against an older engine still decodes the rest of the struct. |
| `public let prefillSeconds: Double` | Time spent in the prompt prefill phase in seconds. |
| `public let decodeSeconds: Double` | Time spent in the token decode phase in seconds. |
| `public let stopReason: StopReason` | Termination reason for this turn. |
| `public let tokensPerSecond: Double?` | Nil when no decoding happened, so a caller cannot plot a rate that was never measured. |
| `public let content: String` | The reply. THIS is the assistant turn to append to history. |
| `public let reasoning: String` | The model's thinking. Do NOT append it to history: Harmony's own convention drops prior-turn analysis and Qwen's template drops prior-turn `<think>` blocks, so feeding it back sends the model something it was never trained to read. |
| `public let peakMemoryPressure: String` | The WORST memory pressure seen while this turn decoded: `normal`, `warn` or `critical`. **`normal` when nothing was watching, which is the default.** The in-loop probe follows the power profile's stepping, and `performance` (the default) polls nothing, so on an ordinary session this is the ABSENCE of a reading rather than a report that memory was fine. Read `TurboSparkSession.systemTelemetry` for the machine's current state; this field exists to catch a SPIKE that happened between two of those polls. |

```swift
public struct PhaseReport: Decodable, Sendable, Equatable
```

The decode phase breakdown. Cumulative over every forward pass this session
has served, prefill included. Fields (all `public let`, all plain
measurements): `calls: UInt64`, `totalMsPerCall: Double`, `gpuWaitMs`,
`finalWaitMs`, `routerMs`, `expertIoMs`, `bindMs`, `pipelineWaitMs`,
`cb1GpuMs`, `routedCbGpuMs`, `finalCbGpuMs` (all `Double`),
`expertRequests: UInt64`, `expertHits: UInt64`, and
`public let expertHitRate: Double?` - nil before anything has been requested,
rather than a 0% rate on no data. Only `expertHitRate` and the struct itself
carry doc comments in source.

## TurboSparkError

Source: `swift/TurboSpark/Sources/TurboSpark/Errors.swift`.

```swift
public struct TurboSparkError: Error, CustomStringConvertible
```

Anything the C layer refused to do. The message comes from the library's
per-thread error slot, read immediately after the failing call and on the same
thread, which is what its contract requires.

```swift
public enum Code: Int32, Sendable
```

Status codes mirrored from the C ABI: `invalidArgument = 1`, `open = 2`,
`generate = 3`, `json = 4`, `unsupportedPlatform = 5`, `panic = 6` (a panic
was caught inside the library; the process is intact and the operation did not
happen; this is a library bug rather than anything the caller did),
`unknown = -1`. Only `panic` carries a doc comment in source.

| Signature | Doc text |
|---|---|
| `public let code: Code` | (no doc comment in source) |
| `public let message: String` | (no doc comment in source) |
| `public var description: String { get }` | (no doc comment in source; renders `"\(code): \(message)"`) |

`fromLastError` is internal, not public, and is therefore not part of this
surface.
