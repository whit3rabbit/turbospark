import Foundation

/// Shared parser for YAML frontmatter embedded within Markdown documentation (Skills, Agents, Prompts).
public enum MarkdownFrontmatterParser {
    /// Normalizes CRLF and bare CR into LF before splitting into lines.
    public static func normalizedLines(_ text: String) -> [String] {
        text.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .components(separatedBy: "\n")
    }

    /// Strips surrounding single or double quotes from a YAML value string.
    public static func stripQuotes(_ s: String) -> String {
        var str = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if (str.hasPrefix("\"") && str.hasSuffix("\"")) || (str.hasPrefix("'") && str.hasSuffix("'")) {
            str = String(str.dropFirst().dropLast())
        }
        return str
    }

    /// Parses comma-separated or bracketed array strings into an array of unquoted strings.
    public static func parseListField(_ raw: String?) -> [String]? {
        guard let raw = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        var cleaned = raw
        if cleaned.hasPrefix("[") && cleaned.hasSuffix("]") {
            cleaned = String(cleaned.dropFirst().dropLast())
        }
        let items = cleaned.components(separatedBy: ",")
            .map { stripQuotes($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            .filter { !$0.isEmpty }
        return items.isEmpty ? nil : items
    }

    /// Returns true if the string is one of the YAML block scalar indicators.
    public static func isBlockScalarMarker(_ value: String) -> Bool {
        ["|", ">", "|-", ">-", "|+", ">+"].contains(value)
    }

    /// Splits `---` delimited YAML frontmatter from the markdown body.
    public static func extractFrontmatterAndBody(from text: String) -> (frontmatter: String?, body: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("---") else {
            return (nil, text)
        }

        let lines = normalizedLines(text)
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---" else {
            return (nil, text)
        }

        var closingIndex: Int?
        for i in 1..<lines.count {
            if lines[i].trimmingCharacters(in: .whitespaces) == "---" {
                closingIndex = i
                break
            }
        }

        guard let closing = closingIndex else {
            return (nil, text)
        }

        let frontmatterLines = lines[1..<closing]
        let frontmatter = frontmatterLines.joined(separator: "\n")

        let bodyLines = lines[(closing + 1)...]
        let body = bodyLines.joined(separator: "\n")

        return (frontmatter, body)
    }
}
