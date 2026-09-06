import CTurboSpark
import Foundation

/// One opened model.
///
/// **A class with a serial queue rather than an `actor`, and that is the
/// point rather than a preference.** `cancel()` has to be callable from the
/// main thread WHILE a turn is running, and an actor method cannot be: it
/// would suspend behind the in-flight generation, so the Stop button would
/// only take effect once the model had finished on its own. The C layer is
/// built the same way for the same reason -- its cancel flag lives outside
/// the engine mutex -- and this type would throw that away if it wrapped the
/// handle in an actor.
///
/// Everything except `cancel()` runs on `queue`, which is serial, so the
/// library's one-generation-at-a-time contract is met by construction.
///
/// `@unchecked Sendable` is justified rather than assumed: the only stored
/// mutable state is the opaque handle, every use of it is confined to
/// `queue`, and `cancel()`'s single call is documented thread-safe by the C
/// header and implemented as an atomic store.
public final class TurboSparkSession: @unchecked Sendable {
    /// The C handle, wrapped so it can cross into `queue`'s `@Sendable`
    /// closures.
    ///
    /// `OpaquePointer` is not `Sendable`, correctly: the compiler cannot
    /// know what is behind one. The unchecked conformance is the claim that
    /// THIS pointer is safe to move, and it rests on two facts stated in the
    /// C header -- every use is confined to the serial `queue` except
    /// `ts_session_cancel`, which is documented safe from any thread and is
    /// an atomic store.
    private struct Handle: @unchecked Sendable {
        let raw: OpaquePointer
    }

    private let handle: Handle
    private let queue: DispatchQueue

    /// Lends the raw handle for one synchronous C call.
    ///
    /// **`internal`, and the ONLY caller is `TurboSparkServer.attach`.**
    /// `ts_server_attach_session` needs this session's pointer and takes it
    /// for the duration of the call only -- it clones the shared engine
    /// reference out from under it and retains nothing. Every other use of
    /// this handle stays confined to `queue`, which is what Gotcha 1's
    /// threading claim rests on; this one is safe outside it because the C
    /// side takes no engine lock and mutates no session state.
    ///
    /// **`self` IS RETAINED FOR THE CALL** by the closure being executed
    /// here rather than escaping, which matters for the same reason
    /// `generate`'s capture list does: a caller releasing its session
    /// mid-call would otherwise run `ts_session_close` beside this.
    func withRawHandle<T>(_ body: (OpaquePointer) throws -> T) rethrows -> T {
        try withExtendedLifetime(self) { try body(handle.raw) }
    }

    /// Everything resolved at open. Read the `maxContext` and
    /// `expertCacheSlots` here rather than what you asked for: under
    /// automatic sizing you asked for nothing, and these are what the KV
    /// cache was actually allocated at.
    public let info: SessionInfo

    /// Opens a model directory or an installed alias.
    ///
    /// A leading `~` is expanded, because a Mac user writing a path by hand
    /// writes one and the C layer takes its argument literally -- an alias
    /// and an absolute path both work there, and `~/models/x` would be the
    /// one plausible spelling that silently is not found. An ALIAS is passed
    /// through untouched: expansion only ever affects a string starting with
    /// `~`, and no alias does.
    ///
    /// Expensive: it maps gigabytes and compiles Metal pipelines. Open once
    /// and keep the session. Runs off the calling thread, so it is safe to
    /// `await` this from a SwiftUI view.
    public init(modelPath: String, options: OpenOptions = OpenOptions()) async throws {
        let modelPath = NSString(string: modelPath).expandingTildeInPath
        let queue = DispatchQueue(label: "com.turbospark.session", qos: .userInitiated)
        let optionsJSON = try Self.encode(options)

        let handle: OpaquePointer = try await withCheckedThrowingContinuation { cont in
            queue.async {
                var out: OpaquePointer?
                let status = modelPath.withCString { path in
                    optionsJSON.withCString { opts in
                        ts_session_open(path, opts, &out)
                    }
                }
                guard status == 0, let out else {
                    cont.resume(throwing: TurboSparkError.fromLastError(status))
                    return
                }
                cont.resume(returning: out)
            }
        }

        // Read the session's description BEFORE any stored property is
        // assigned, and close the C handle BY HAND when that fails.
        //
        // A class whose `init` throws part-way through was never fully
        // initialized, so Swift does not run its `deinit` -- which means the
        // `ts_session_close` below cannot be what releases the handle on this
        // path, however obviously it looks like it is. Leaving it to `deinit`
        // stranded a whole open session (mapped weights, KV cache, compiled
        // Metal pipelines) for the life of the process on every failed
        // `ts_session_info_json` or decode, and the failure is silent: the
        // caller sees the error it expected and the memory never comes back.
        let info: SessionInfo
        do {
            info = try decode(
                SessionInfo.self,
                from: try takeString { ts_session_info_json(handle, $0) }
            )
        } catch {
            ts_session_close(handle)
            throw error
        }

        self.handle = Handle(raw: handle)
        self.queue = queue
        self.info = info
    }

    deinit {
        // The C header forbids closing while a generation is in flight, and
        // what keeps that true is `generate`'s worker capturing `self`
        // STRONGLY: the session cannot reach zero references while a turn is
        // on `queue`, so `deinit` cannot run beside `ts_generate`. That
        // capture is load-bearing rather than incidental -- this comment used
        // to claim the stream's task provided it, which it never did (the
        // capture list named `handle` alone), leaving the whole contract
        // resting on every caller happening to hold the session themselves.
        ts_session_close(handle.raw)
    }

    /// Asks the running turn to stop.
    ///
    /// `nonisolated` in spirit and in fact: safe from any thread, never
    /// blocks, and never queues behind the generation it is stopping. A Stop
    /// pressed when nothing is running is discarded rather than cancelling
    /// the next turn.
    public func cancel() {
        ts_session_cancel(handle.raw)
    }

    /// Generates one assistant turn, streaming events as they arrive.
    ///
    /// The stream ends with `.finished(result)`. Cancelling the consuming
    /// `Task` cancels the generation too, so `for try await` inside a
    /// SwiftUI `.task` stops the model when the view goes away.
    ///
    /// Accumulate `.content` as the assistant turn. `.reasoning` is for
    /// display only; see `GenerationResult.reasoning`.
    public func generate(
        _ messages: [ChatMessage],
        options: GenerateOptions = GenerateOptions()
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let messagesJSON: String
            let optionsJSON: String
            do {
                messagesJSON = try Self.encode(messages)
                optionsJSON = try Self.encode(options)
            } catch {
                continuation.finish(throwing: error)
                return
            }

            // A dropped consumer must stop the model rather than leave it
            // decoding into a stream nobody reads. `weak` deliberately: this
            // handler must ASK the session to stop, never extend its life
            // past the consumer that owns it.
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }

            // `self` is captured STRONGLY here and that is the whole reason
            // `deinit` is safe. The C header forbids `ts_session_close` while
            // `ts_generate` is in flight, so the session has to outlive the
            // turn; nothing else in this function does that, since
            // `onTermination` above is weak and the returned stream holds no
            // reference back. Dropping `self` from this list compiles, passes
            // every test, and reintroduces a use-after-free reachable by any
            // caller that lets its session go while a turn is running.
            queue.async { [self] in
                let box = Box(continuation)
                let userdata = Unmanaged.passRetained(box).toOpaque()
                // Balanced on every path out, including the throwing one.
                defer { Unmanaged<Box>.fromOpaque(userdata).release() }

                do {
                    let json = try takeString { out in
                        messagesJSON.withCString { m in
                            optionsJSON.withCString { o in
                                ts_generate(self.handle.raw, m, o, streamCallback, userdata, out)
                            }
                        }
                    }
                    continuation.yield(.finished(try decode(GenerationResult.self, from: json)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Starts an in-process HTTP server sharing THIS session's engine, and
    /// returns a handle to it.
    ///
    /// **THE SERVER OUTLIVES THIS `TurboSparkSession` IF YOU LET IT.** It
    /// holds its own reference to the underlying engine on the Rust side, so
    /// this session going out of scope (and closing) after this call frees
    /// only this Swift object -- the model stays resident and the server
    /// keeps serving it until `TurboSparkServer.stop()` (or its `deinit`)
    /// releases the last reference. Keep a strong reference to the returned
    /// `TurboSparkServer` for as long as it should keep running.
    ///
    /// Serves the same OpenAI/Anthropic-compatible routes the standalone
    /// `turbospark-server` binary does, EXCEPT vision and the standalone
    /// binary's tool-call guardrails: a session opened through this type
    /// carries no vision wiring, so an image request the server receives is
    /// refused by name rather than silently dropped.
    public func startServer(options: ServerOptions = ServerOptions()) async throws
        -> TurboSparkServer
    {
        let optionsJSON = try Self.encode(options)
        return try await withCheckedThrowingContinuation { cont in
            queue.async { [handle] in
                var out: OpaquePointer?
                let status = optionsJSON.withCString { opts in
                    ts_server_start(handle.raw, opts, &out)
                }
                guard status == 0, let out else {
                    cont.resume(throwing: TurboSparkError.fromLastError(status))
                    return
                }
                cont.resume(returning: TurboSparkServer(raw: out))
            }
        }
    }

    /// The decode phase breakdown. Cheap enough to poll for a status panel.
    public func phases() async throws -> PhaseReport {
        try await withCheckedThrowingContinuation { cont in
            queue.async { [handle] in
                cont.resume(
                    with: Result {
                        try decode(
                            PhaseReport.self,
                            from: try takeString { ts_session_phases_json(handle.raw, $0) }
                        )
                    })
            }
        }
    }

    /// Evaluates exact prompt token count for `messages` using this session's
    /// chat template and tokenizer, without running generation.
    public func countTokens(
        _ messages: [ChatMessage],
        reasoning: GenerateOptions.Reasoning = .off
    ) async throws -> Int {
        let messagesJSON = try Self.encode(messages)
        let reasoningStr = reasoning.rawValue
        return try await withCheckedThrowingContinuation { cont in
            queue.async { [handle] in
                var count: UInt32 = 0
                let status = messagesJSON.withCString { m in
                    reasoningStr.withCString { r in
                        ts_session_count_tokens(handle.raw, m, r, &count)
                    }
                }
                guard status == 0 else {
                    cont.resume(throwing: TurboSparkError.fromLastError(status))
                    return
                }
                cont.resume(returning: Int(count))
            }
        }
    }

    /// Evaluates the token count of a raw text string using this session's tokenizer.
    public func countTokens(
        in text: String,
        addSpecialTokens: Bool = false
    ) async throws -> Int {
        try await withCheckedThrowingContinuation { cont in
            queue.async { [handle] in
                var count: UInt32 = 0
                let status = text.withCString { t in
                    ts_session_count_text_tokens(handle.raw, t, addSpecialTokens, &count)
                }
                guard status == 0 else {
                    cont.resume(throwing: TurboSparkError.fromLastError(status))
                    return
                }
                cont.resume(returning: Int(count))
            }
        }
    }

    /// Formats a conversation transcript into raw prompt text using this session's
    /// chat template and reasoning effort setting.
    public func renderPrompt(
        _ messages: [ChatMessage],
        reasoning: GenerateOptions.Reasoning = .off
    ) async throws -> String {
        let messagesJSON = try Self.encode(messages)
        let reasoningStr = reasoning.rawValue
        return try await withCheckedThrowingContinuation { cont in
            queue.async { [handle] in
                do {
                    let prompt = try takeString { out in
                        messagesJSON.withCString { m in
                            reasoningStr.withCString { r in
                                ts_session_render_prompt(handle.raw, m, r, out)
                            }
                        }
                    }
                    cont.resume(returning: prompt)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Tokenizes raw text into an array of integer token IDs using this session's tokenizer.
    public func tokenize(
        _ text: String,
        addSpecialTokens: Bool = false
    ) async throws -> [Int32] {
        return try await withCheckedThrowingContinuation { cont in
            queue.async { [handle] in
                do {
                    let json = try takeString { out in
                        text.withCString { t in
                            ts_session_tokenize_json(handle.raw, t, addSpecialTokens, out)
                        }
                    }
                    cont.resume(returning: try decode([Int32].self, from: json))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Detokenizes an array of integer token IDs into text using this session's tokenizer.
    public func detokenize(
        _ tokens: [Int32],
        skipSpecialTokens: Bool = false
    ) async throws -> String {
        let tokensJSON = try Self.encode(tokens)
        return try await withCheckedThrowingContinuation { cont in
            queue.async { [handle] in
                do {
                    let text = try takeString { out in
                        tokensJSON.withCString { t in
                            ts_session_detokenize_json(handle.raw, t, skipSpecialTokens, out)
                        }
                    }
                    cont.resume(returning: text)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Fits a conversation transcript into a context token budget by pruning older turns
    /// (preserving optional leading system instruction and newest user turn).
    ///
    /// `maxTokens`: the token budget limit. When `nil`, defaults to the session's resolved `maxContext`.
    public func fitWindow(
        _ messages: [ChatMessage],
        maxTokens: UInt32? = nil,
        reasoning: GenerateOptions.Reasoning = .off
    ) async throws -> WindowFitOutcome {
        let messagesJSON = try Self.encode(messages)
        let reasoningStr = reasoning.rawValue
        let limit = maxTokens ?? info.maxContext
        return try await withCheckedThrowingContinuation { cont in
            queue.async { [handle] in
                do {
                    let json = try takeString { out in
                        messagesJSON.withCString { m in
                            reasoningStr.withCString { r in
                                ts_session_fit_window_json(handle.raw, m, r, limit, out)
                            }
                        }
                    }
                    cont.resume(returning: try decode(WindowFitOutcome.self, from: json))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// This process's peak physical footprint in bytes, or nil where the
    /// counter is unavailable.
    ///
    /// Process-wide rather than per session, and what it counts differs by
    /// install shape: a streamed MoE model's mapped weights ARE counted, a
    /// dense model's are not. Read it beside `info.maxContext` rather than
    /// comparing it across models.
    public static var peakFootprintBytes: UInt64? {
        let bytes = ts_peak_footprint_bytes()
        return bytes == 0 ? nil : bytes
    }

    /// Hardware and power telemetry for this machine, or nil where unavailable.
    public static var systemTelemetry: SystemTelemetry? {
        guard let json = try? takeString({ ts_system_info_json($0) }) else { return nil }
        return try? decode(SystemTelemetry.self, from: json)
    }

    private static func encode<T: Encodable>(_ value: T) throws -> String {
        let data = try JSONEncoder().encode(value)
        guard let json = String(data: data, encoding: .utf8) else {
            throw TurboSparkError(code: .json, message: "could not encode arguments as UTF-8")
        }
        return json
    }
}

/// Carries the continuation across the C boundary as a `void *`.
private final class Box {
    let continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation
    init(_ continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation) {
        self.continuation = continuation
    }
}

/// The C event callback.
///
/// A file-scope function rather than a closure: a C function pointer cannot
/// capture, so the continuation has to travel in `userdata`.
private let streamCallback: TsEventCallback = {
    userdata, kind, text, len, a, b in
    guard let userdata else { return }
    // `takeUnretainedValue`: the retain is balanced by `generate`'s `defer`,
    // not per call.
    let box = Unmanaged<Box>.fromOpaque(userdata).takeUnretainedValue()

    switch kind {
    case TS_EVENT_PREFILL:
        box.continuation.yield(.prefill(done: Int(a), total: Int(b)))
    case TS_EVENT_CONTENT, TS_EVENT_REASONING:
        // The pointer is valid only for this call, so the String
        // initializer's copy is required rather than an optimization.
        guard let text, len > 0 else { return }
        let bytes = UnsafeRawBufferPointer(start: text, count: len)
        guard let s = String(bytes: bytes, encoding: .utf8) else { return }
        box.continuation.yield(kind == TS_EVENT_CONTENT ? .content(s) : .reasoning(s))
    case TS_EVENT_TOOL:
        guard let text, len > 0 else { return }
        let bytes = UnsafeRawBufferPointer(start: text, count: len)
        guard let s = String(bytes: bytes, encoding: .utf8) else { return }
        if let call = GenerationToolCall(parsingJSON: s) {
            box.continuation.yield(.toolCall(call))
        }
    case TS_EVENT_FINISH:
        // The terminal stream event; `.finished` still follows with the
        // full result once ts_generate returns.
        guard let text, len > 0 else { return }
        let bytes = UnsafeRawBufferPointer(start: text, count: len)
        guard let s = String(bytes: bytes, encoding: .utf8) else { return }
        box.continuation.yield(.stopped(stopReason: s, newTokens: Int(a), promptTokens: Int(b)))
    default:
        // An unknown kind is a newer library than this wrapper. Dropping it
        // is right: the alternative is crashing on a field nobody asked for.
        return
    }
}
