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
    /// Cap on one nested tool result handed into the VM as JSON.
    var maximumCallPayloadCharacters: Int
    var defaultTimeout: TimeInterval
    var maximumTimeout: TimeInterval
    /// Script size bound, matching the REPL request bound.
    var maximumScriptBytes: Int

    init(
        maximumOutputCharacters: Int = 32_000,
        maximumOutputItems: Int = 1_000,
        maximumStoreValueCharacters: Int = 64 * 1_024,
        maximumStoreTotalCharacters: Int = 256 * 1_024,
        maximumCallPayloadCharacters: Int = 64 * 1_024,
        defaultTimeout: TimeInterval = 120,
        maximumTimeout: TimeInterval = 600,
        maximumScriptBytes: Int = 1_024 * 1_024
    ) {
        self.maximumOutputCharacters = maximumOutputCharacters
        self.maximumOutputItems = maximumOutputItems
        self.maximumStoreValueCharacters = maximumStoreValueCharacters
        self.maximumStoreTotalCharacters = maximumStoreTotalCharacters
        self.maximumCallPayloadCharacters = maximumCallPayloadCharacters
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

    /// Builds the callable entries for a descriptor list. Both the sanitized
    /// identifier and the original advertised name are callable; when two
    /// names sanitize to the same identifier the first (descriptors arrive
    /// name-sorted) wins for the identifier form.
    static func entries(for descriptors: [DeferredToolDescriptor]) -> [CodemodeToolEntry] {
        var byJSName: [String: CodemodeToolEntry] = [:]
        var byAdvertised: Set<String> = []
        var result: [CodemodeToolEntry] = []
        for descriptor in descriptors {
            guard byAdvertised.insert(descriptor.name.lowercased()).inserted else { continue }
            let jsName = sanitize(descriptor.name)
            if byJSName[jsName] == nil {
                byJSName[jsName] = CodemodeToolEntry(
                    name: descriptor.name, jsName: jsName, description: descriptor.description)
                result.append(CodemodeToolEntry(
                    name: descriptor.name, jsName: jsName, description: descriptor.description))
            }
        }
        return result
    }
}
