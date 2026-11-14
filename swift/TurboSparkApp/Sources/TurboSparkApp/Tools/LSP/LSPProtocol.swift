import Foundation

/// OC-10: the small typed subset of LSP the feature reads. Servers vary (a
/// diagnostic `code` is an Int or a String, a documentSymbol reply is a tree
/// or a flat SymbolInformation list), so parsing is lenient and a malformed
/// item is dropped rather than failing the whole reply.
struct LSPPosition: Equatable, Sendable {
    var line: Int
    var character: Int
}

struct LSPRange: Equatable, Sendable {
    var start: LSPPosition
    var end: LSPPosition
}

struct LSPDiagnostic: Equatable, Sendable {
    /// LSP severity: 1 error, 2 warning, 3 information, 4 hint. Missing means
    /// the client picks; this client treats it as an error.
    var severity: Int
    var range: LSPRange
    var message: String
    var source: String?
    var code: String?

    var isError: Bool { severity == 1 }
    var isWarning: Bool { severity == 2 }
}

struct LSPDocumentSymbol: Equatable, Sendable {
    var name: String
    var kind: Int
    var detail: String?
    var range: LSPRange
    var children: [LSPDocumentSymbol]
}

struct LSPWorkspaceSymbol: Equatable, Sendable {
    var name: String
    var kind: Int
    var container: String?
    var uri: String
    /// Nil for a symbol the server returned without a range (the lazy
    /// `WorkspaceSymbol` form).
    var range: LSPRange?
}

enum LSPParse {
    static func position(_ value: Any?) -> LSPPosition? {
        guard let dict = value as? [String: Any],
              let line = int(dict["line"]), let character = int(dict["character"]),
              line >= 0, character >= 0 else { return nil }
        return LSPPosition(line: line, character: character)
    }

    static func range(_ value: Any?) -> LSPRange? {
        guard let dict = value as? [String: Any],
              let start = position(dict["start"]), let end = position(dict["end"]) else { return nil }
        return LSPRange(start: start, end: end)
    }

    static func int(_ value: Any?) -> Int? {
        if let number = value as? Int { return number }
        if let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() { return number.intValue }
        return nil
    }

    static func diagnostics(_ value: Any?) -> [LSPDiagnostic] {
        guard let array = value as? [Any] else { return [] }
        return array.compactMap { item in
            guard let dict = item as? [String: Any], let range = range(dict["range"]),
                  let message = dict["message"] as? String else { return nil }
            var code: String?
            if let text = dict["code"] as? String { code = text } else if let number = int(dict["code"]) { code = String(number) }
            return LSPDiagnostic(
                severity: int(dict["severity"]) ?? 1, range: range, message: message,
                source: dict["source"] as? String, code: code)
        }
    }

    /// Parses either reply shape of `textDocument/documentSymbol`. The flat
    /// `SymbolInformation` form is lifted into leaf symbols.
    static func documentSymbols(_ value: Any?) -> [LSPDocumentSymbol] {
        guard let array = value as? [Any] else { return [] }
        return array.compactMap { documentSymbol($0, depth: 0) }
    }

    private static func documentSymbol(_ value: Any, depth: Int) -> LSPDocumentSymbol? {
        guard depth < 32, let dict = value as? [String: Any], let name = dict["name"] as? String else { return nil }
        let kind = int(dict["kind"]) ?? 0
        if let range = range(dict["range"]) {
            let children = ((dict["children"] as? [Any]) ?? []).compactMap { documentSymbol($0, depth: depth + 1) }
            return LSPDocumentSymbol(
                name: name, kind: kind, detail: dict["detail"] as? String, range: range, children: children)
        }
        if let location = dict["location"] as? [String: Any], let range = range(location["range"]) {
            return LSPDocumentSymbol(name: name, kind: kind, detail: dict["containerName"] as? String, range: range, children: [])
        }
        return nil
    }

    static func workspaceSymbols(_ value: Any?) -> [LSPWorkspaceSymbol] {
        guard let array = value as? [Any] else { return [] }
        return array.compactMap { item in
            guard let dict = item as? [String: Any], let name = dict["name"] as? String,
                  let location = dict["location"] as? [String: Any], let uri = location["uri"] as? String
            else { return nil }
            return LSPWorkspaceSymbol(
                name: name, kind: int(dict["kind"]) ?? 0, container: dict["containerName"] as? String,
                uri: uri, range: range(location["range"]))
        }
    }

    /// LSP `SymbolKind` names, for output the model reads. Unknown kinds print
    /// as "symbol" so a new server kind never breaks the line format.
    static func kindName(_ kind: Int) -> String {
        let names = [
            1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class", 6: "method",
            7: "property", 8: "field", 9: "constructor", 10: "enum", 11: "interface", 12: "function",
            13: "variable", 14: "constant", 15: "string", 16: "number", 17: "boolean", 18: "array",
            19: "object", 20: "key", 21: "null", 22: "enumMember", 23: "struct", 24: "event",
            25: "operator", 26: "typeParameter",
        ]
        return names[kind] ?? "symbol"
    }
}
