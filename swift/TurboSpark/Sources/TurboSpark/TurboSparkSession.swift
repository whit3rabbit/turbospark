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

        self.handle = Handle(raw: handle)
        self.queue = queue
        // Read on the opening thread, immediately, so a later failure cannot
        // leave a session with no description of itself.
        self.info = try decode(
            SessionInfo.self,
            from: try takeString { ts_session_info_json(handle, $0) }
        )
    }

    deinit {
        // The C header forbids closing while a generation is in flight. A
        // `deinit` cannot run while `generate` holds a strong reference to
        // `self`, which the stream's task does for its whole lifetime, so
        // this cannot race.
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
            // decoding into a stream nobody reads.
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }

            queue.async { [handle] in
                let box = Box(continuation)
                let userdata = Unmanaged.passRetained(box).toOpaque()
                // Balanced on every path out, including the throwing one.
                defer { Unmanaged<Box>.fromOpaque(userdata).release() }

                do {
                    let json = try takeString { out in
                        messagesJSON.withCString { m in
                            optionsJSON.withCString { o in
                                ts_generate(handle.raw, m, o, streamCallback, userdata, out)
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
    default:
        // An unknown kind is a newer library than this wrapper. Dropping it
        // is right: the alternative is crashing on a field nobody asked for.
        return
    }
}
