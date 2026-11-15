import CTurboSpark
import Foundation

// The GGUF quantizer: writes a new GGUF from a float one under a per-tensor
// recipe. The work and its limits live in docs/QUANTIZE.md; this file is the
// Swift shape of the five `ts_quantize_*` entry points (crates/ffi/src/api/
// quantize.rs). Every key is camelCase, the wire shape the Rust side emits,
// so no CodingKeys are needed.
//
// What it will NOT do, so a UI can say so before the run: the source must be
// a float GGUF (F32/F16/BF16), a split GGUF (`-0000N-of-0000M.gguf`) is
// refused (merge it with llama-gguf-split first), and the types it can write
// are the ones `supportedTypes()` lists. macOS only: elsewhere every call
// throws `.unsupportedPlatform`.

/// Namespace for the quantizer calls.
public enum GGUFQuantizer {

    /// What to quantize each tensor to.
    public enum Recipe: Sendable, Equatable {
        /// One type for every eligible tensor, by name (`"q4_k"`). The names
        /// are the `name` column of `supportedTypes()`.
        case type(String)
        /// A recipe object as `recipe(fromGGUF:)` writes it: a `default` type
        /// plus per-tensor overrides. See docs/QUANTIZE.md for the format.
        case object(JSONValue)
        /// Follow the per-tensor types of an existing GGUF (for example an
        /// Unsloth file). Types this quantizer cannot write are refused with
        /// the tensor named.
        case followGGUF(path: String)
    }

    /// Where an input lives on Hugging Face, so a later `checkUpdates` can ask
    /// the Hub whether it changed. Each is `owner/repo:file`.
    public struct HuggingFaceLocations: Encodable, Sendable, Equatable {
        public var source: String?
        public var reference: String?
        public var imatrix: String?

        public init(source: String? = nil, reference: String? = nil, imatrix: String? = nil) {
            self.source = source
            self.reference = reference
            self.imatrix = imatrix
        }
    }

    /// Tie-break order for IQ1 quantization. `.stable` is the default and is
    /// deterministic everywhere; `.apple` matches llama.cpp built against
    /// Apple's libc sort.
    public enum IQ1TieOrder: String, Encodable, Sendable {
        case stable
        case apple
    }

    public struct Options: Sendable, Equatable {
        public var source: String
        public var destination: String
        public var recipe: Recipe
        /// An imatrix file (GGUF or the legacy `.dat`). Low-bit types need one.
        public var imatrix: String?
        /// Worker threads. Nil uses every core.
        public var threads: Int?
        public var iq1TieOrder: IQ1TieOrder?
        /// Write `<destination>.provenance.json`. The engine's default is true.
        public var provenance: Bool?
        public var huggingFace: HuggingFaceLocations?

        public init(
            source: String, destination: String, recipe: Recipe, imatrix: String? = nil,
            threads: Int? = nil, iq1TieOrder: IQ1TieOrder? = nil, provenance: Bool? = nil,
            huggingFace: HuggingFaceLocations? = nil
        ) {
            self.source = source
            self.destination = destination
            self.recipe = recipe
            self.imatrix = imatrix
            self.threads = threads
            self.iq1TieOrder = iq1TieOrder
            self.provenance = provenance
            self.huggingFace = huggingFace
        }

        /// The ABI's `optionsJson`. Exposed for tests; the wire keys are
        /// `recipe` or `recipeFromGguf` (never both) and `hf`.
        func wireJSON() throws -> String {
            struct Wire: Encodable {
                let source: String
                let destination: String
                let recipe: JSONValue?
                let recipeFromGguf: String?
                let imatrix: String?
                let threads: Int?
                let iq1TieOrder: IQ1TieOrder?
                let provenance: Bool?
                let hf: HuggingFaceLocations?
            }
            var recipeValue: JSONValue?
            var follow: String?
            switch recipe {
            case .type(let name): recipeValue = .string(name)
            case .object(let value): recipeValue = value
            case .followGGUF(let path): follow = path
            }
            let wire = Wire(
                source: source, destination: destination, recipe: recipeValue,
                recipeFromGguf: follow, imatrix: imatrix, threads: threads,
                iq1TieOrder: iq1TieOrder, provenance: provenance, hf: huggingFace)
            return String(decoding: try JSONEncoder().encode(wire), as: UTF8.self)
        }
    }

    /// What a finished run wrote.
    public struct Report: Decodable, Sendable, Equatable {
        public let tensors: UInt64
        public let bytesIn: UInt64
        public let bytesOut: UInt64
        /// Output tensor count per ggml type name.
        public let byType: [String: UInt64]
        /// Already-quantized tensors that were copied, not re-quantized.
        public let keptQuantized: [String]
        public let imatrixUsed: UInt64
        /// Tensors of an imatrix-aware type with no imatrix entry; they were
        /// quantized without weights, as llama.cpp does.
        public let imatrixMissing: [String]
        /// Output types this engine's install gate would not run. The file is
        /// still a valid GGUF for llama.cpp. Show it before offering to load.
        public let notInExecutableList: [String]
        /// The provenance file, nil when `provenance` was false.
        public let provenancePath: String?
    }

    /// One progress event.
    public enum Event: Sendable, Equatable {
        /// The tensor now being written.
        case tensor(String)
        /// Output bytes written of the planned total.
        case bytes(done: UInt64, total: UInt64)
        case finished(Report)
    }

    /// A type the quantizer can write.
    public struct TypeInfo: Decodable, Sendable, Equatable, Identifiable {
        public var id: String { name }
        /// The recipe spelling (`"q4_k"`).
        public let name: String
        public let ggmlType: Int
        public let bitsPerWeight: Double
        /// Elements per block: a tensor row must be a multiple of it.
        public let blockElements: Int
        /// ggml's hard rule. llama.cpp's policy for other very low-bit types
        /// is in docs/QUANTIZE.md.
        public let requiresImatrix: Bool
        /// Whether this engine's install gate lists the type, i.e. whether a
        /// file made of it can be loaded here.
        public let executable: Bool
    }

    /// How a recorded input compares with Hugging Face now.
    public enum FreshnessStatus: String, Decodable, Sendable, Equatable {
        case upToDate, changed, sizeOnly, missing, untracked, unreachable
    }

    public struct InputFreshness: Decodable, Sendable, Equatable {
        /// `source`, `reference` or `imatrix`.
        public let role: String
        public let fileName: String
        public let status: FreshnessStatus
        public let detail: String?
        public let commit: String?
        /// True for `changed`, `sizeOnly` and `missing`. `unreachable` is NOT
        /// "up to date": show it as unknown.
        public let needsAttention: Bool
    }

    // MARK: - Calls

    /// Quantizes `options.source` to `options.destination`, streaming
    /// progress, and ends with `.finished`.
    ///
    /// Runs on its own thread and blocks it for minutes. One run at a time per
    /// process: a second call's stream throws `TurboSparkError` with code
    /// `.busy`. Cancelling the consuming task (or calling `cancel()`) stops
    /// the run at its next batch; the stream then throws code `.cancelled`
    /// (not a failure to show) and no destination file is left. An existing
    /// destination, a split source, a quantized source tensor the recipe would
    /// re-quantize, and a recipe type the quantizer cannot write all throw
    /// `.invalidArgument` with the reason and (where there is one) the tensor.
    public static func quantize(_ options: Options) -> AsyncThrowingStream<Event, Error> {
        AsyncThrowingStream { continuation in
            let state = RunState()
            continuation.onTermination = { reason in
                // Only a consumer that walked away stops the engine, and only
                // while this run is still the one in flight: after `finish`
                // the engine slot may already belong to the next caller.
                if case .cancelled = reason { state.cancelIfRunning() }
            }
            let thread = Thread {
                let box = QuantizeBox(continuation)
                let userdata = Unmanaged.passRetained(box).toOpaque()
                defer { Unmanaged<QuantizeBox>.fromOpaque(userdata).release() }
                do {
                    let wire = try options.wireJSON()
                    state.markRunning()
                    let json = try takeString { out in
                        wire.withCString { w in
                            ts_quantize_gguf(w, quantizeCallback, userdata, out)
                        }
                    }
                    state.markDone()
                    continuation.yield(.finished(try decode(Report.self, from: json)))
                    continuation.finish()
                } catch {
                    state.markDone()
                    continuation.finish(throwing: error)
                }
            }
            thread.name = "com.turbospark.quantize"
            thread.start()
        }
    }

    /// Asks the run in flight to stop. True when one was running and was
    /// signalled, false when none was (not an error).
    @discardableResult
    public static func cancel() -> Bool {
        ts_quantize_cancel() != 0
    }

    /// The types the quantizer can write.
    public static func supportedTypes() throws -> [TypeInfo] {
        let json = try takeString { ts_quantize_types_json($0) }
        return try decode([TypeInfo].self, from: json)
    }

    /// The recipe that would reproduce the per-tensor types of the GGUF at
    /// `path`. Feed it back as `Recipe.object`, or save it as the recipe file
    /// the CLI takes.
    public static func recipe(fromGGUF path: String) throws -> JSONValue {
        let json = try takeString { out in
            path.withCString { ts_quantize_recipe_from_gguf_json($0, out) }
        }
        return try JSONValue(jsonString: json)
    }

    /// Asks Hugging Face whether the inputs recorded in a provenance file
    /// still have the recorded hashes.
    ///
    /// **Blocks on the network** for up to about 30 seconds per repo, so this
    /// runs on a background thread and suspends the caller. An unreachable Hub
    /// is reported per input, never as up to date.
    public static func checkUpdates(provenancePath: String) async throws -> [InputFreshness] {
        try await withCheckedThrowingContinuation { continuation in
            let thread = Thread {
                do {
                    let json = try takeString { out in
                        provenancePath.withCString { ts_quantize_check_updates_json($0, out) }
                    }
                    continuation.resume(returning: try decode([InputFreshness].self, from: json))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
            thread.name = "com.turbospark.quantize.check"
            thread.start()
        }
    }
}

/// Tracks whether THIS run holds the engine's single run slot, so a late
/// consumer cancellation cannot stop somebody else's run.
private final class RunState: @unchecked Sendable {
    private let lock = NSLock()
    private var running = false
    private var cancelRequested = false

    func markRunning() {
        lock.lock()
        running = true
        let again = cancelRequested
        lock.unlock()
        // A cancel that arrived before the engine call started: honour it.
        if again { _ = ts_quantize_cancel() }
    }

    func markDone() {
        lock.lock()
        running = false
        lock.unlock()
    }

    func cancelIfRunning() {
        lock.lock()
        cancelRequested = true
        let live = running
        lock.unlock()
        if live { _ = ts_quantize_cancel() }
    }
}

/// Progress arrives on the quantizing thread only (the writer's contract), so
/// the box needs no lock.
private final class QuantizeBox: @unchecked Sendable {
    let continuation: AsyncThrowingStream<GGUFQuantizer.Event, Error>.Continuation
    init(_ continuation: AsyncThrowingStream<GGUFQuantizer.Event, Error>.Continuation) {
        self.continuation = continuation
    }
}

private let quantizeCallback: TsInstallCallback = { userdata, kind, text, len, done, total in
    guard let userdata else { return }
    let box = Unmanaged<QuantizeBox>.fromOpaque(userdata).takeUnretainedValue()
    switch kind {
    case TS_INSTALL_STAGE:
        guard let text, len > 0,
            let s = String(bytes: UnsafeRawBufferPointer(start: text, count: len), encoding: .utf8)
        else { return }
        box.continuation.yield(.tensor(s))
    case TS_INSTALL_BYTES:
        box.continuation.yield(.bytes(done: done, total: total))
    default:
        return
    }
}
