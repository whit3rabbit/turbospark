import Foundation

/// Value types for the codemode sandbox: a port of pi's codemode pattern
/// (one script tool whose `tools.<name>(args)` calls run deferred MCP tools
/// inside a fresh JavaScriptCore worker process). The host-side result shape
/// mirrors pi's `{ ok, value, output, calls, error }`; `calls` is telemetry
/// and never re-enters model context.

/// One tool the script may call, as shipped to the worker. The worker only
/// needs the callable surface; schemas stay host-side (they shape the
/// description's declaration listing, they do not gate anything here --
/// gating happens in the nested runner on the host).
struct CodemodeToolEntry: Sendable, Equatable, Codable {
    /// The advertised `mcp__<server>__<tool>` name. This is the name the
    /// nested runner re-gates and executes; the script uses `jsName` (or
    /// this name via bracket access) to call it.
    var name: String
    /// Sanitized JavaScript identifier exposed as `tools.<jsName>`.
    var jsName: String
    var description: String
}

/// One captured output item, in emission order. `console` items carry the
/// level they were logged at; `text` items come from `text()`.
struct CodemodeOutputItem: Sendable, Equatable, Codable {
    enum Kind: String, Sendable, Codable {
        case text
        case console
    }

    var kind: Kind
    var text: String
    var level: String?
}

/// Outcome of one `tools.<name>(...)` call made from inside the script.
enum CodemodeCallStatus: String, Sendable, Codable {
    case ok
    case error
    case cancelled
}

struct CodemodeCallRecord: Sendable, Equatable, Codable {
    var name: String
    var status: CodemodeCallStatus
    /// Duration of the host-side nested call in milliseconds.
    var durationMs: Int
}

/// Why a script execution ended when it did not succeed. Mirrors pi's
/// error kinds: `script` (threw or failed to parse), `timeout` (deadline
/// expired and the worker was terminated), `aborted` (the enclosing task
/// was cancelled), `sandbox` (the worker or VM failed outright).
enum CodemodeErrorKind: String, Sendable, Codable {
    case script
    case timeout
    case aborted
    case sandbox
}

struct CodemodeError: Sendable, Equatable, Codable {
    var kind: CodemodeErrorKind
    var name: String?
    var message: String
    var stack: String?
    /// 1-based line in the model's script where the failure happened, when
    /// the engine could attribute one. JavaScriptCore's own `stack` carries
    /// no frame text without a source URL, so this is what lets the model
    /// fix the line that threw.
    var line: Int?
}

struct CodemodeStoreWrite: Sendable, Equatable, Codable {
    /// JSON text of the stored value, or nil for a deletion.
    var key: String
    var valueJSON: String?
}

struct CodemodeResult: Sendable, Equatable {
    var ok: Bool
    /// JSON text of the script's return value, when it returned one.
    var valueJSON: String?
    /// Ordered output items (text and console) captured before the end.
    var output: [CodemodeOutputItem]
    /// Every nested call with status and duration. Telemetry only.
    var calls: [CodemodeCallRecord]
    var error: CodemodeError?
    /// Store writes to apply, reported on success only.
    var storeWrites: [CodemodeStoreWrite]
}

/// Bounds for one codemode execution. Sized for a single tool result in
/// this app rather than pi's process-wide maxima; see the README table.
struct CodemodeLimits: Sendable, Equatable, Codable {
    var maximumOutputCharacters: Int
    var maximumOutputItems: Int
    var maximumStoreValueCharacters: Int
    var maximumStoreTotalCharacters: Int
    /// Keys in the per-chat store. Enforced inside the VM so the script gets a
    /// deterministic RangeError; the host box keeps its own cap as a backstop.
    var maximumStoreKeys: Int
    /// Bound on one call's JSON arguments, in characters. A script can build
    /// arguments of any size; this keeps one call from putting a huge line on
    /// the wire and a huge object into an MCP server.
    var maximumCallArgumentCharacters: Int
    /// Bound, in UTF-8 bytes, on one nested tool result handed into the VM.
    /// A result over the bound is REJECTED, never truncated: a script that
    /// parses or counts a clipped result computes a wrong answer with no
    /// error. The bound exists to protect memory and the wire line limit
    /// (`CodemodeWire.maximumLineBytes`); the output budget is separate and
    /// only counts what the script prints or returns.
    var maximumCallResultBytes: Int
    /// Total nested calls one script may make. Each call spawns an MCP
    /// server process, so an unbounded loop of calls is a fork-exhaustion
    /// and side-effect amplifier for the whole app.
    var maximumCalls: Int
    /// Nested calls in flight at once. Calls past this wait for a slot
    /// instead of failing, so `Promise.all` over a long list still works.
    var maximumConcurrentCalls: Int
    /// Physical memory footprint the worker process may reach, in bytes;
    /// 0 disables the check. JavaScriptCore has no in-process heap limit and
    /// macOS does not enforce `RLIMIT_AS`, so the host polls the worker's
    /// footprint (`proc_pid_rusage`) and kills it past the bound, which also
    /// works for a script spinning in a synchronous loop.
    var maximumWorkerMemoryBytes: Int
    var defaultTimeout: TimeInterval
    var maximumTimeout: TimeInterval
    /// Script size bound, matching the REPL request bound.
    var maximumScriptBytes: Int

    init(
        maximumOutputCharacters: Int = 32_000,
        maximumOutputItems: Int = 1_000,
        maximumStoreValueCharacters: Int = 64 * 1_024,
        maximumStoreTotalCharacters: Int = 256 * 1_024,
        maximumStoreKeys: Int = 256,
        maximumCallArgumentCharacters: Int = 1_024 * 1_024,
        maximumCallResultBytes: Int = 2 * 1_024 * 1_024,
        maximumCalls: Int = 200,
        maximumConcurrentCalls: Int = 8,
        maximumWorkerMemoryBytes: Int = 1_024 * 1_024 * 1_024,
        defaultTimeout: TimeInterval = 120,
        maximumTimeout: TimeInterval = 600,
        maximumScriptBytes: Int = 1_024 * 1_024
    ) {
        self.maximumOutputCharacters = maximumOutputCharacters
        self.maximumOutputItems = maximumOutputItems
        self.maximumStoreValueCharacters = maximumStoreValueCharacters
        self.maximumStoreTotalCharacters = maximumStoreTotalCharacters
        self.maximumStoreKeys = maximumStoreKeys
        self.maximumCallArgumentCharacters = maximumCallArgumentCharacters
        self.maximumCallResultBytes = maximumCallResultBytes
        self.maximumWorkerMemoryBytes = maximumWorkerMemoryBytes
        self.maximumCalls = maximumCalls
        self.maximumConcurrentCalls = maximumConcurrentCalls
        self.defaultTimeout = defaultTimeout
        self.maximumTimeout = maximumTimeout
        self.maximumScriptBytes = maximumScriptBytes
    }
}

/// Failure of one nested call made from inside a script. The message
/// becomes the rejection reason the script sees.
struct CodemodeCallError: Error {
    var message: String
}

/// Sanitizes an advertised `mcp__<server>__<tool>` name into a JavaScript
/// identifier, the way pi's toCodemodeIdentifier does: invalid characters
/// become `_`, a leading digit is prefixed, and reserved words gain a
/// trailing underscore. Collisions resolve first-wins at catalog build.
enum CodemodeIdentifier {
    static let reservedWords: Set<String> = [
        "break", "case", "catch", "class", "const", "continue", "debugger",
        "default", "delete", "do", "else", "enum", "export", "extends",
        "false", "finally", "for", "function", "if", "import", "in",
        "instanceof", "new", "null", "return", "super", "switch", "this",
        "throw", "true", "try", "typeof", "var", "void", "while", "with",
        "let", "static", "yield", "await"
    ]

    static func sanitize(_ raw: String) -> String {
        var allowed = raw.map { character -> Character in
            let scalar = character.unicodeScalars.first!
            let isAsciiLetter = (scalar.value >= 65 && scalar.value <= 90)
                || (scalar.value >= 97 && scalar.value <= 122)
            let isDigit = scalar.value >= 48 && scalar.value <= 57
            if isAsciiLetter || isDigit || character == "_" || character == "$" {
                return character
            }
            return "_"
        }
        if let first = allowed.first, first.isNumber {
            allowed = ["_"] + allowed
        }
        var name = String(allowed)
        if name.isEmpty { name = "_" }
        if reservedWords.contains(name) { name += "_" }
        return name
    }

    /// Builds the callable entries for a descriptor list. A tool is callable
    /// as `tools.<jsName>` and, where the prelude can bind it, by its
    /// advertised name too. Advertised names repeated (compared
    /// case-insensitively, as the catalog does) collapse to the first. When
    /// two DIFFERENT names sanitize to one identifier (`mcp__a-b__c` and
    /// `mcp__a_b__c`), the first keeps the identifier and each later one gets
    /// a numeric suffix (`..._2`), so no tool becomes unreachable.
    static func entries(for descriptors: [DeferredToolDescriptor]) -> [CodemodeToolEntry] {
        var usedJSNames: Set<String> = []
        var seenAdvertised: Set<String> = []
        var result: [CodemodeToolEntry] = []
        for descriptor in descriptors {
            guard seenAdvertised.insert(descriptor.name.lowercased()).inserted else { continue }
            let base = sanitize(descriptor.name)
            var jsName = base
            var suffix = 2
            while usedJSNames.contains(jsName) {
                jsName = "\(base)_\(suffix)"
                suffix += 1
            }
            usedJSNames.insert(jsName)
            result.append(CodemodeToolEntry(
                name: descriptor.name, jsName: jsName, description: descriptor.description))
        }
        return result
    }
}
