import Foundation

/// OC-10: how server output reaches the model. Everything a language server
/// says is untrusted text, so it is flattened to one line, escaped so it
/// cannot close the wrapper tag, and bounded before it is shown.
enum LSPOutputFormatting {
    static let postEditItemLimit = 10
    static let toolItemLimit = 50
    static let messageCharacterLimit = 240
    static let symbolLimit = 300
    static let symbolDepthLimit = 6
    static let workspaceSymbolDefaultLimit = 30
    static let workspaceSymbolMaximumLimit = 100

    /// A single bounded line: control characters and newlines become spaces,
    /// angle brackets are escaped, and the result is capped.
    static func sanitize(_ text: String, limit: Int = messageCharacterLimit) -> String {
        var out = ""
        out.reserveCapacity(min(text.count, limit + 8))
        var count = 0
        for scalar in text.unicodeScalars {
            if count >= limit { out += "..."; break }
            switch scalar {
            case "<": out += "&lt;"
            case ">": out += "&gt;"
            case "\n", "\r", "\t": out += " "
            default:
                if scalar.value < 0x20 || scalar.value == 0x7F { out += " " } else { out.unicodeScalars.append(scalar) }
            }
            count += 1
        }
        return out
    }

    private static func position(_ range: LSPRange) -> String {
        // LSP is 0-based; the model reads 1-based lines and columns.
        "\(range.start.line + 1):\(range.start.character + 1)"
    }

    private static func severityName(_ diagnostic: LSPDiagnostic) -> String {
        switch diagnostic.severity {
        case 2: return "warning"
        case 3: return "info"
        case 4: return "hint"
        default: return "error"
        }
    }

    /// The compact summary the post-edit step and `project_diagnostics` share.
    /// Errors sort before warnings; information and hints are not shown.
    /// `itemLimit` caps the diagnostic lines across all files.
    static func diagnosticsSummary(
        _ reports: [LSPFileReport], itemLimit: Int, budgetSeconds: Double? = nil
    ) -> String {
        var lines: [String] = []
        var shown = 0
        var hidden = 0
        for report in reports {
            let path = sanitize(report.path, limit: 200)
            let server = sanitize(report.server, limit: 60)
            switch report.status {
            case .diagnostics(let all):
                let errors = all.filter(\.isError)
                let warnings = all.filter(\.isWarning)
                if errors.isEmpty && warnings.isEmpty {
                    lines.append("\(path): no errors or warnings (\(server))")
                    continue
                }
                var head = "\(path): "
                var parts: [String] = []
                if !errors.isEmpty { parts.append("\(errors.count) error\(errors.count == 1 ? "" : "s")") }
                if !warnings.isEmpty { parts.append("\(warnings.count) warning\(warnings.count == 1 ? "" : "s")") }
                head += parts.joined(separator: ", ") + " (\(server))"
                lines.append(head)
                let ordered = errors.sorted { $0.range.start.line < $1.range.start.line }
                    + warnings.sorted { $0.range.start.line < $1.range.start.line }
                for diagnostic in ordered {
                    guard shown < itemLimit else { hidden += 1; continue }
                    shown += 1
                    var line = "  \(severityName(diagnostic)) \(position(diagnostic.range)) \(sanitize(diagnostic.message))"
                    if let code = diagnostic.code { line += " [\(sanitize(code, limit: 40))]" }
                    lines.append(line)
                }
            case .timedOut:
                let waited = budgetSeconds.map { " within \(formatSeconds($0))" } ?? ""
                lines.append("\(path): \(server) had not reported\(waited). Call project_diagnostics with this file to check again.")
            case .unavailable(let reason):
                lines.append("\(path): \(server) language server unavailable (\(sanitize(reason, limit: 160))). No diagnostics were checked.")
            case .skipped(let reason):
                lines.append("\(path): not checked (\(sanitize(reason, limit: 160)))")
            }
        }
        if hidden > 0 { lines.append("  ...\(hidden) more not shown. Call project_diagnostics with a file for the full list.") }
        return lines.joined(separator: "\n")
    }

    private static func formatSeconds(_ value: Double) -> String {
        value == value.rounded() ? "\(Int(value))s" : String(format: "%.1fs", value)
    }

    /// Indented symbol tree, one line each: `kind name line:col`. Bounded in
    /// count and depth so a generated file cannot flood the context.
    static func symbolTree(_ symbols: [LSPDocumentSymbol]) -> String {
        var lines: [String] = []
        var total = 0
        var truncated = false
        func walk(_ items: [LSPDocumentSymbol], depth: Int) {
            for item in items {
                if total >= symbolLimit { truncated = true; return }
                total += 1
                var line = String(repeating: "  ", count: depth)
                line += "\(LSPParse.kindName(item.kind)) \(sanitize(item.name, limit: 120)) \(position(item.range))"
                lines.append(line)
                if depth + 1 < symbolDepthLimit { walk(item.children, depth: depth + 1) }
                else if !item.children.isEmpty { lines.append(String(repeating: "  ", count: depth + 1) + "...\(item.children.count) nested not shown") }
            }
        }
        walk(symbols, depth: 0)
        if truncated { lines.append("...symbol list truncated at \(symbolLimit) entries.") }
        return lines.joined(separator: "\n")
    }
}
