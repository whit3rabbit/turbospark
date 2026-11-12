# TurboSpark (Swift binding)

## Overview
This target is the Swift wrapper over the C ABI in `CTurboSpark`. It turns the engine's JSON-and-callback surface into typed Swift: sessions, streaming generation, catalog and install control, model libraries, in-process HTTP server, audio, OCR, embeddings, image generation and diagnostics. Options travel to Rust as JSON and results are decoded from JSON; callbacks cross the boundary through retained boxes in `userdata`. Blocking engine calls are moved onto serial queues or dedicated threads so callers stay off the main actor.

## Interface
- `TurboSparkError`, `check`, `takeString`, `takeOptionalString`, `decode` ([Errors.swift:TurboSparkError]) - status-to-error mapping via `ts_last_error`, string ownership and JSON decode helpers used by every file.
- `TurboSparkRuntime` ([Runtime.swift:TurboSparkRuntime]) - `buildInfo()`, `verifyABI()`, `verifyABIOnce()` header-vs-archive ABI check.
- `TurboSparkSession` ([TurboSparkSession.swift:TurboSparkSession]) - one open model on a serial queue: `init(modelPath:options:)`, `generate`, `humanize`, `cancel`, `startServer`, `countTokens`, `renderPrompt`, `tokenize`, `detokenize`, `fitWindow`, `releaseVision`, `phases`, `systemTelemetry`, `peakFootprintBytes`.
- `QueuedGenerationStream` ([QueuedGenerationStream.swift:QueuedGenerationStream]) - ties cancellation to the queued request that owns a stream.
- `OpenOptions`, `GenerateOptions` ([Options.swift:OpenOptions]) - encodable open and per-turn knobs (sizing, residency, speculation, load guard, KV bits, steering, vision sidecar, ANE prefill, sampling, tools).
- `ChatMessage`, `ChatImage`, `WindowFitOutcome` ([ChatTypes.swift:ChatMessage]) - conversation wire shape with images prepended as ordered parts.
- `GenerationEvent`, `GenerationResult`, `GenerationToolCall`, `PhaseReport` ([GenerationTypes.swift:GenerationEvent]) - streamed events and the final turn report.
- `SessionInfo` ([SessionTypes.swift:SessionInfo]) - resolved open state (context, reasoning levels, steering, tool calling, speculation, vision, ANE prefill, wiring).
- `JSONValue`, `ToolSpec` ([JSONValue.swift:JSONValue]) - untyped JSON passthrough and tool definitions.
- `HumanizeOptions`, `HumanizeAudit` ([HumanizeTypes.swift:HumanizeAudit]) - humanizer options and rewrite audit.
- `TurboSparkServer`, `ServerOptions`, `ServerInfo`, `ServerEvent`, `ServerEventBatch` ([TurboSparkServer.swift:TurboSparkServer]) - in-process HTTP server lifecycle, attach/detach, polling, image and speech request bridges.
- `TurboSparkCatalog` ([Catalog.swift:TurboSparkCatalog]) - store root, catalog, installed lists, probe, recommend, context ladder, install streams, cancel/pause/resume, delete.
- `CatalogEntry`, `InstalledModel`, `ImageInstalledModel`, `ImageCatalogEntry`, `InstallEvent`, `RecommendationEvent` ([Catalog.swift:CatalogEntry]) - catalog row types (snake_case on-disk formats).
- `TurboSparkCatalog.hubSearch`, `hubCheck` ([CatalogHub.swift:hubSearch]) - Hugging Face search and per-repo compatibility check.
- `HubSearchPage`, `HubSearchResult`, `HubCheck`, `HubCheckState`, `HubArchHint` ([HubSearchTypes.swift:HubCheck]) - hub result types with unknown-value fallbacks.
- `ProbeReport`, `ProbeFit`, `ContextLadder`, `RepoVariants`, `CountedSource` ([ProbeTypes.swift:ProbeReport]) - header-probe and context-cost report types.
- `ModelRecommendation`, `SystemTelemetry` ([SystemTypes.swift:ModelRecommendation]) - hardware-fit ranking rows and machine telemetry.
- `TurboSparkLibraries`, `ModelLibrary`, `MoveOutcome`, `ImportedStore` ([Libraries.swift:TurboSparkLibraries]) - model-library registry calls; `move` runs on a dedicated thread.
- `LibraryErrorCode`, `LibraryFault`, `LibraryRefusal`, `UntrustedText` ([LibraryErrors.swift:LibraryFault]) - typed library error codes parsed from the `<code>: <sentence>` prefix and safe display of untrusted text.
- `TurboSparkCapabilities`, `FamilyCapabilities`, `ExpertResidencyPreflight` ([Capabilities.swift:TurboSparkCapabilities]) - capability predicates answered by Rust.
- `HfTokenValidationStatus`, `HfTokenInfo`, `TurboSparkCatalog` token and endpoint extension ([Auth.swift:getHfToken]) - Hugging Face token and mirror endpoint control.
- `AudioCatalog`, `AudioSession`, `AudioJob`, `AudioOpenToken`, `NativeHeavyWorkPermit`, `AudioRequest`, `AudioResult` ([Audio.swift:AudioSession]) - native audio catalog, install, cancellable open and job execution (snake_case wire).
- `AudioStreamSession`, `AudioStreamUpdate` ([AudioStream.swift:AudioStreamSession]) - live transcription stream over a streaming speech install.
- `TurboSparkSTTModel`, `TurboSparkSTTStream`, `STTTranscription` ([Speech.swift:TurboSparkSTTModel]) - resident speech-to-text model and buffered streams.
- `TurboSparkEmbedding`, `EmbeddingCatalog`, `EmbeddingProfile`, `EmbeddingProbe`, `runPinnedInstall` ([Embedding.swift:TurboSparkEmbedding]) - embedding encode, rank, catalog and pinned install.
- `TurboSparkOcr`, `OcrCatalog`, `OcrProfile`, `OcrReadResult` ([Ocr.swift:TurboSparkOcr]) - document OCR session and pinned installs.
- `TurboSparkImageSession`, `ImageGenerateOptions`, `ImageGenerationEvent`, `GeneratedImageMetadata` ([TurboSparkImageSession.swift:TurboSparkImageSession]) - native image generation with PNG copied out of Rust memory.
- `TurboSparkTokenizer` ([TurboSparkTokenizer.swift:TurboSparkTokenizer]) - tokenizer opened without a session.
- `TurboSparkDaemon`, `TurboSparkDaemonStatus`, `TurboSparkAgent` ([Daemon.swift:TurboSparkDaemon]) - background server control and agent launch command builders.
- `StderrCapture` ([Diagnostics.swift:StderrCapture]) - tee of process stderr (fd 2) to a line handler.

## Invariants
- Session work is confined to a serial `DispatchQueue`; only `cancel()` may run concurrently, and the class is deliberately not an actor so Stop is never queued behind the turn ([TurboSparkSession.swift:TurboSparkSession]).
- `generate` captures `self` strongly so the session cannot deinit (and call `ts_session_close`) while a turn runs ([TurboSparkSession.swift:deinit]).
- If `ts_session_info_json` or its decode fails in `init`, the C handle is closed by hand because a throwing class `init` never runs `deinit` ([TurboSparkSession.swift:init(modelPath:options:)]).
- Session, tokenizer and server opens call `TurboSparkRuntime.verifyABIOnce()` first, and a header/archive ABI mismatch throws ([Runtime.swift:verify], [TurboSparkTokenizer.swift:init(modelPath:)], [TurboSparkServer.swift:start]).
- `TurboSparkServer` holds its lock across every C call, including `info()`, because `ts_server_stop` frees the handle; `stop()` is idempotent and `deinit` also stops ([TurboSparkServer.swift:info], [TurboSparkServer.swift:stopLocked]).
- Detaching, not releasing the Swift session, is what frees a model attached to a server ([TurboSparkServer.swift:detach]).
- `poll` and the image/speech pollers return empty after stop instead of throwing; an undecodable row is skipped without dropping its batch ([TurboSparkServer.swift:poll], [TurboSparkServer.swift:LossyImageEvent]).
- A queued stream request that is cancelled before it starts is skipped, and a cancel racing `ts_generate` setup is replayed at the first event ([QueuedGenerationStream.swift:Request]).
- `takeString` frees the C string in a `defer`, and keys are decoded without a key strategy because Rust emits camelCase; snake_case types (catalog, installed, audio, embedding, OCR) use explicit keys or `convertFromSnakeCase` ([Errors.swift:takeString], [Errors.swift:decode], [Catalog.swift:CatalogEntry], [Audio.swift:audioDecode]).
- `GenerationToolCall(parsingJSON:)` avoids `data(withJSONObject:)` on a non-container to prevent an uncatchable ObjC exception inside a C callback ([GenerationTypes.swift:GenerationToolCall]).
- `ChatMessage` encodes `content` as a bare string unless images exist, and prepends image parts before text ([ChatTypes.swift:encode]).
- `LibraryFault` keys on the `<code>:` prefix only; `TurboSparkError.Code.busy` alone is not the library `busy` ([LibraryErrors.swift:LibraryFault]).
- Names, aliases and error sentences from drives are untrusted and are escaped and truncated through `UntrustedText.display` ([LibraryErrors.swift:UntrustedText]).
- Unknown enum values from a newer engine decode to a fallback (`StopReason.unknown`, `FitVerdict.unknown`, `HubSearchStatus.error`, `ServerEvent.unknown`, `MoveOutcome.unrecognized`) rather than failing the decode ([GenerationTypes.swift:StopReason], [SystemTypes.swift:FitVerdict], [HubSearchTypes.swift:HubSearchStatus], [TurboSparkServer.swift:ServerEvent], [Libraries.swift:MoveOutcome]).
- Install, move, recommend and store-relocation calls run on dedicated `Thread`s, not shared queue workers; install byte events may arrive out of order ([Catalog.swift:install], [Libraries.swift:move]).
- `TurboSparkSTTModel.close()` latches `closed` only when `ts_stt_close` succeeds, since Rust refuses while a stream is open ([Speech.swift:close]).
- `AudioSession` serializes jobs on an `operation` lock but cancels via a separate `control` lock so cancel never waits; audio input is limited to finite samples and 30 minutes, appended in chunks of 32,000 ([Audio.swift:executeBlocking]).
- `AudioStreamSession.close()` cancels, then waits for any running call; cancel is terminal ([AudioStream.swift:close]).
- `TurboSparkImageSession` serializes generation and close on its queue; the PNG is copied into `Data` before the Rust buffer is freed ([TurboSparkImageSession.swift:deinit], [TurboSparkImageSession.swift:generate]).
- Shell and AppleScript strings built for agent launch are escaped for their quoting context ([Daemon.swift:shellDoubleQuoted], [Daemon.swift:terminalDoScript]).
- `StderrCapture` is process-wide, allows one capture at a time, tees to the original stderr, and handles `stop()` from inside the handler without deadlock ([Diagnostics.swift:start], [Diagnostics.swift:stop]).
- `kvQuantSupported` rejects non-byte layer masks and non-positive counts as false ([Capabilities.swift:kvQuantSupported]).

## Side Effects
- FFI calls: every public entry point calls `ts_*` functions in the linked `libturbospark_ffi.a` ([Errors.swift:takeString], [TurboSparkSession.swift:init(modelPath:options:)]).
- Disk: store root, install, delete, move, library add/remove, forget and import write the engine's registry under `~/.turbospark` and library folders ([Catalog.swift:install], [Catalog.swift:deleteInstall], [Libraries.swift:add], [Libraries.swift:move]).
- Network: catalog probe, hub search/check, pinned installs and HF token validation reach Hugging Face through the engine ([CatalogHub.swift:hubSearch], [Catalog.swift:probe], [Auth.swift:validateHfToken]).
- Credentials: HF token written to and cleared from the local store; HF mirror endpoint set via the engine ([Auth.swift:setHfToken], [Auth.swift:clearHfToken], [Auth.swift:setHfEndpoint]).
- Sockets: `TurboSparkServer.start` and `TurboSparkSession.startServer` bind a local HTTP listener ([TurboSparkServer.swift:start], [TurboSparkSession.swift:startServer]).
- Processes: daemon start, stop and restart manage a background server process; launch builders only produce command strings and the Terminal AppleScript, they do not run them ([Daemon.swift:TurboSparkDaemon], [Daemon.swift:launchCommand]).
- File descriptors: `StderrCapture` replaces fd 2 with a pipe and restores it on stop ([Diagnostics.swift:start]).
- Threads and queues: named queues `com.turbospark.session`, `com.turbospark.image-session`, and threads `com.turbospark.install`, `.image-install`, `.recommend`, `.model-move`, `.store-relocation` ([TurboSparkSession.swift:init(modelPath:options:)], [Catalog.swift:install], [Libraries.swift:move]).
- Memory and GPU: opening a session maps weights and compiles Metal pipelines; `releaseVision` frees tower resources ([TurboSparkSession.swift:releaseVision]).
- Audio and Metal permits: `NativeHeavyWorkPermit` reserves native heavy execution ([Audio.swift:NativeHeavyWorkPermit]).
- Keychain, UserDefaults and notifications: None.

## Verification
- Run: `swift test --package-path swift/TurboSpark`
