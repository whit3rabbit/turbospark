import Foundation

/// Carries a finished codemode result (including partial output) out of
/// `AppToolRegistry.execute`'s switch so the result can land in an
/// `AppToolResult` with `isError` set and its call telemetry archived, the
/// same channel `DeferredMcpContinuationStop` uses for hook stops.
struct CodemodeResultStop: LocalizedError {
    let output: String
    let isError: Bool
    let archivalOutput: String?

    var errorDescription: String? { output }
}

/// The codemode run entry: parses the script source (including the
/// optional `// @options:` line), snapshots and applies the per-chat store,
/// launches one supervisor process, and formats the final result into the
/// text the model sees.
///
/// Output policy (pi's the part that makes "only useful output reaches the
/// model" true): the prelude hard-caps `text()`/`console` volume, each
/// nested tool result is tail-truncated before it enters the VM, and this
/// type joins the script's own output items with the return value. The
/// `calls` telemetry is returned separately and must only ever land in
/// archival output, never the prompt projection.
enum CodemodeSandbox {
    /// An optional first line `// @options: { ... }` with `timeout_ms` and
    /// `max_output_chars`. The sandbox does not act on unknown fields, and
    /// a malformed line is a source error, matching pi's CodemodeSourceError
    /// behavior. The line is replaced with an empty line so stack-trace
    /// line numbers match the original source.
    struct ParsedSource: Equatable {
        var code: String
        var timeoutOverride: TimeInterval?
        var maxOutputCharsOverride: Int?

        /// Shortest deadline an `// @options:` line may ask for. A worker
        /// process cannot even start in a few milliseconds, so a smaller
        /// request could only ever time out.
        static let minimumTimeout: TimeInterval = 1

        static func parse(_ raw: String, limits: CodemodeLimits) throws -> ParsedSource {
            var lines = raw.components(separatedBy: "\n")
            // The options line is the first NON-BLANK line rather than strictly
            // line 1: a model often starts with a blank line, and ignoring the
            // line then would silently drop the timeout it asked for. The
            // line is blanked in place, so line numbers do not move.
            guard let optionsIndex = lines.firstIndex(where: {
                !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }) else {
                throw sourceError("the script is empty.")
            }
            let trimmed = lines[optionsIndex].trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("// @options:") else {
                return ParsedSource(code: raw, timeoutOverride: nil, maxOutputCharsOverride: nil)
            }
            let jsonText = String(trimmed.dropFirst("// @options:".count))
                .trimmingCharacters(in: .whitespaces)
            guard let data = jsonText.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                throw sourceError(
                    "the // @options: line is not a valid JSON object. "
                        + "Expected: // @options: {\"timeout_ms\": 120000}")
            }
            var timeoutOverride: TimeInterval?
            var maxOutputCharsOverride: Int?
            for (key, value) in object {
                switch key {
                case "timeout_ms":
                    guard let milliseconds = positiveNumber(value) else {
                        throw sourceError("// @options: timeout_ms must be a positive number.")
                    }
                    timeoutOverride = min(
                        max(milliseconds / 1000, minimumTimeout), limits.maximumTimeout)
                case "max_output_chars":
                    guard let characters = positiveNumber(value), characters >= 1 else {
                        throw sourceError("// @options: max_output_chars must be a positive integer.")
                    }
                    maxOutputCharsOverride = min(Int(min(characters, 1e9)), 200_000)
                default:
                    throw sourceError(
                        "// @options: unknown field '\(key)'. Supported fields: "
                            + "timeout_ms, max_output_chars.")
                }
            }
            if raw.trimmingCharacters(in: .whitespacesAndNewlines) == trimmed {
                throw sourceError("// @options: was given without any script code after it.")
            }
            lines[optionsIndex] = ""
            return ParsedSource(
                code: lines.joined(separator: "\n"),
                timeoutOverride: timeoutOverride,
                maxOutputCharsOverride: maxOutputCharsOverride)
        }

        /// A JSON number greater than zero. `NSNumber` also wraps `true` and
        /// `false` (as 1 and 0), which are not numbers a script author means.
        private static func positiveNumber(_ value: Any) -> Double? {
            guard let number = value as? NSNumber,
                  CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.doubleValue > 0
            else { return nil }
            return number.doubleValue
        }

        private static func sourceError(_ message: String) -> NSError {
            NSError(domain: "TurboSparkCodemode", code: 1, userInfo: [
                NSLocalizedDescriptionKey: message
            ])
        }
    }

    /// Per-chat store for `store()`/`load()`, held in memory and bounded:
    /// the prelude enforces the per-value and total character caps at write
    /// time; this box additionally bounds the key count so many chats and
    /// many scripts cannot grow it without limit.
    actor StoreBox {
        static let shared = StoreBox()
        static let maximumKeys = 256
        /// Chats whose stores are kept. The least recently used is dropped
        /// past this, so the box cannot grow with every chat the app opens.
        static let maximumChats = 64

        private var stores: [String: [String: String]] = [:]
        /// Chat keys, least recently used first.
        private var recency: [String] = []

        func snapshot(for key: String) -> [String: String] {
            guard let store = stores[key] else { return [:] }
            touch(key)
            return store
        }

        func apply(_ writes: [CodemodeStoreWrite], forKey key: String) {
            var store = stores[key] ?? [:]
            for write in writes {
                if let valueJSON = write.valueJSON {
                    store[write.key] = valueJSON
                } else {
                    store.removeValue(forKey: write.key)
                }
            }
            if store.count > Self.maximumKeys {
                // The prelude already refuses a write that would pass the
                // cap, so this is a backstop (an older snapshot, a changed
                // limit). It must be deterministic and must keep what the
                // script just wrote: drop keys it did NOT touch first, in
                // name order, then touched ones only if still over.
                let touched = Set(writes.map(\.key))
                let untouched = store.keys.filter { !touched.contains($0) }.sorted()
                let touchedSorted = store.keys.filter { touched.contains($0) }.sorted()
                for name in (untouched + touchedSorted).prefix(store.count - Self.maximumKeys) {
                    store.removeValue(forKey: name)
                }
            }
            stores[key] = store
            touch(key)
            while recency.count > Self.maximumChats {
                stores.removeValue(forKey: recency.removeFirst())
            }
        }

        private func touch(_ key: String) {
            recency.removeAll { $0 == key }
            recency.append(key)
        }
    }

    /// Runs one script and returns the sandbox result. Store writes are
    /// applied only on success, matching pi.
    static func run(
        code: String,
        entries: [CodemodeToolEntry],
        storeKey: String?,
        callHandler: @escaping CodemodeWorkerSupervisor.CallHandler,
        limits: CodemodeLimits = CodemodeLimits(),
        executableURL: URL? = REPLWorkerProcess.defaultExecutableURL(),
        workerArguments: [String]? = nil
    ) async -> CodemodeResult {
        func failure(_ message: String, kind: CodemodeErrorKind = .script) -> CodemodeResult {
            CodemodeResult(
                ok: false, valueJSON: nil, output: [],
                calls: [],
                error: CodemodeError(kind: kind, name: nil, message: message, stack: nil),
                storeWrites: [])
        }

        guard !entries.isEmpty else {
            return failure(
                "codemode has no MCP tools to call: no deferred MCP tools are currently "
                    + "advertised. Connect an MCP server or use its tools directly.")
        }
        guard Data(code.utf8).count <= limits.maximumScriptBytes else {
            return failure(String(
                format: "the script is %d bytes, over the %d byte limit.",
                Data(code.utf8).count, limits.maximumScriptBytes))
        }
        let source: ParsedSource
        do {
            source = try ParsedSource.parse(code, limits: limits)
        } catch {
            return failure(error.localizedDescription)
        }

        guard let executableURL else {
            return failure(
                "the codemode worker executable is not available in this environment.",
                kind: .sandbox)
        }

        // A nil key (no owning chat) means an empty store that is never saved.
        let snapshot: [String: String]
        if let storeKey {
            snapshot = await StoreBox.shared.snapshot(for: storeKey)
        } else {
            snapshot = [:]
        }
        let snapshotData = (try? JSONEncoder().encode(snapshot)) ?? Data("{}".utf8)
        let snapshotJSON = String(data: snapshotData, encoding: .utf8) ?? "{}"

        var effectiveLimits = limits
        if let override = source.maxOutputCharsOverride {
            effectiveLimits.maximumOutputCharacters = override
        }

        let request = CodemodeWire.RunRequest(
            type: "codemode",
            id: 0,
            code: source.code,
            tools: entries,
            storeSnapshot: snapshotJSON,
            limits: effectiveLimits)
        let timeout = source.timeoutOverride ?? limits.defaultTimeout

        let supervisor: CodemodeWorkerSupervisor
        do {
            supervisor = try CodemodeWorkerSupervisor(
                request: request,
                executableURL: executableURL,
                arguments: workerArguments,
                callHandler: callHandler,
                timeout: timeout,
                limits: effectiveLimits)
        } catch {
            return failure(
                "the codemode worker could not start: \(error.localizedDescription).",
                kind: .sandbox)
        }
        defer { supervisor.terminateAndWait() }
        let result = await supervisor.run()
        if result.ok, let storeKey {
            await StoreBox.shared.apply(result.storeWrites, forKey: storeKey)
        }
        return result
    }

    /// Formats a sandbox result into the tool output the model sees: the
    /// script's own output items in order, then the return value, then the
    /// failure text. The error stack is included for script errors so the
    /// model can fix the line that threw; stack length is bounded.
    static func promptOutput(for result: CodemodeResult) -> (output: String, isError: Bool) {
        var lines: [String] = []
        for item in result.output {
            lines.append(item.text)
        }
        if result.ok {
            if let valueJSON = result.valueJSON {
                lines.append("")
                lines.append("Return value: " + valueJSON)
            }
            if lines.isEmpty {
                lines.append("(the script produced no output and returned nothing)")
            }
            return (lines.joined(separator: "\n"), false)
        }

        var failure = "Script failed"
        if let error = result.error {
            if let name = error.name {
                failure += " (" + name + ")"
            }
            failure += ": " + error.message
            if let line = error.line {
                failure += " (line \(line))"
            }
            if let stack = error.stack, !stack.isEmpty {
                let bounded = stack.count > 2_000 ? String(stack.prefix(2_000)) + "..." : stack
                failure += "\n" + bounded
            }
            if error.kind == .timeout {
                failure += "\nPartial output above is what the script printed before the deadline."
            }
        } else {
            failure += ": the sandbox reported a failure without a reason."
        }
        if !lines.isEmpty {
            return (lines.joined(separator: "\n") + "\n\n" + failure, true)
        }
        return (failure, true)
    }

    /// Telemetry JSON for archival output. Never assembled into the prompt
    /// projection.
    static func archivalCalls(for result: CodemodeResult) -> String? {
        guard !result.calls.isEmpty else { return nil }
        let payload: [String: Any] = [
            "calls": result.calls.map { call in
                ["name": call.name, "status": call.status.rawValue, "duration_ms": call.durationMs]
            }
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8)
        else { return nil }
        return text
    }
}
