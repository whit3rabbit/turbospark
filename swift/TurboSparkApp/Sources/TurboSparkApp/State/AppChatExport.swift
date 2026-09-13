import AppKit
import Foundation
import UniformTypeIdentifiers

/// Conversation export, the qwen-code `/export` feature: a chat rendered as
/// Markdown (for reading, archiving, pasting into a doc) or as JSON (the
/// lossless row dump, the whole `AppChat` as its Codable form).
///
/// Ghost chats are refused by the caller before reaching here. An export IS
/// a persistence surface -- a file on disk the vault knows nothing about --
/// so "temporary" and "exportable" cannot both hold.
public enum AppChatExportFormat: String, CaseIterable, Identifiable {
    case markdown
    case json
    /// The qwen-code HTML export: one self-contained page per conversation.
    case html

    public var id: String { rawValue }

    public var fileExtension: String {
        switch self {
        case .markdown: return "md"
        case .json: return "json"
        case .html: return "html"
        }
    }

    public var contentType: UTType {
        switch self {
        case .markdown: return UTType(filenameExtension: "md") ?? .plainText
        case .json: return .json
        case .html: return .html
        }
    }
}

public enum AppChatExport {
    /// Renders `chat` as a human-readable Markdown document: title header,
    /// then one section per turn, with tool calls folded into fenced
    /// summaries and reasoning kept under a details heading. Only the ACTIVE
    /// version of each row exports; alternates are noted by count, not
    /// inlined, or a heavily-retried chat would export its whole history
    /// three times over.
    public static func markdown(for chat: AppChat, modelAlias: String?) -> String {
        var lines: [String] = []
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .long
        dateFormatter.timeStyle = .short

        lines.append("# \(chat.title)")
        lines.append("")
        var meta: [String] = []
        if !chat.isGhost {
            meta.append("Created: \(dateFormatter.string(from: chat.createdAt))")
        }
        if let alias = modelAlias, !alias.isEmpty {
            meta.append("Model: \(alias)")
        }
        if chat.compactedMessageCount > 0 {
            meta.append("Note: the first \(chat.compactedMessageCount) message(s) were compacted into a summary")
        }
        if !meta.isEmpty {
            lines.append(contentsOf: meta)
            lines.append("")
        }
        if let summary = chat.contextSummary, !summary.isEmpty {
            lines.append("> [Conversation summary of earlier turns]")
            for summaryLine in summary.split(separator: "\n", omittingEmptySubsequences: false) {
                lines.append("> \(summaryLine)")
            }
            lines.append("")
        }

        for message in chat.messages {
            switch message.role {
            case .user:
                lines.append("## User")
                lines.append("")
                lines.append(message.content.isEmpty ? "*(attachment only)*" : message.content)
            case .assistant:
                lines.append("## Assistant")
                lines.append("")
                if !message.reasoning.isEmpty {
                    lines.append("<details><summary>Reasoning</summary>")
                    lines.append("")
                    lines.append(message.reasoning)
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
                }
                if !message.toolCalls.isEmpty {
                    for call in message.toolCalls {
                        let status = message.toolResults.first(where: { $0.callID == call.id })
                        lines.append("**Tool: \(call.name)**\(status.map { statusText($0) }.map { " (\($0))" } ?? "")")
                        lines.append("")
                        lines.append("```json")
                        lines.append(argumentJSON(call.arguments))
                        lines.append("```")
                        if let status, !status.output.isEmpty {
                            lines.append("")
                            lines.append("```")
                            lines.append(truncate(status.output))
                            lines.append("```")
                        }
                        lines.append("")
                    }
                }
                if !message.content.isEmpty {
                    lines.append(message.content)
                }
                if !message.alternates.isEmpty {
                    lines.append("")
                    lines.append("*(\(message.alternates.count) earlier version(s) not shown)*")
                }
            default:
                lines.append("## System")
                lines.append("")
                lines.append(message.content)
            }
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    /// Lossless export: the chat row itself as pretty-printed JSON.
    public static func jsonData(for chat: AppChat) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(chat)
    }

    /// Renders the chat as one self-contained HTML page (the qwen-code
    /// HTML export). The conversation is already exported to Markdown by
    /// `markdown(for:)`, and that renderer is the single place the per-turn
    /// rules live, so the page carries the MARKDOWN TEXT converted per
    /// block: a small converter for the shapes this app's own exports
    /// contain (headings, paragraphs, fenced code, lists, quotes, inline
    /// styles). No external dependency and no network, matching the
    /// offline-first rule the HTML render panel keeps.
    public static func html(for chat: AppChat, modelAlias: String?) -> String {
        let markdownBody = markdown(for: chat, modelAlias: modelAlias)
        let escaped = escapeHTML(markdownBody)
        // Line-based block pass over the escaped markdown.
        var blocks: [String] = []
        var paragraph: [String] = []
        var listItems: [String] = []
        var listOrdered = false
        var inFence = false
        var fenceLines: [String] = []
        var fenceLanguage = ""

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            let text = inlineHTML(paragraph.joined(separator: " "))
            blocks.append("<p>\(text)</p>")
            paragraph = []
        }
        func flushList() {
            guard !listItems.isEmpty else { return }
            let items = listItems.map { "<li>\(inlineHTML($0))</li>" }
                .joined(separator: "\n")
            blocks.append(listOrdered ? "<ol>\n\(items)\n</ol>" : "<ul>\n\(items)\n</ul>")
            listItems = []
        }

        for line in escaped.split(separator: "\n", omittingEmptySubsequences: false) {
            let raw = String(line)
            if raw.hasPrefix("```") {
                if inFence {
                    blocks.append(
                        "<pre><code class=\"lang-\(fenceLanguage)\">\(fenceLines.joined(separator: "\n"))</code></pre>")
                    fenceLines = []
                    inFence = false
                } else {
                    flushParagraph()
                    flushList()
                    inFence = true
                    fenceLanguage = String(raw.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                }
                continue
            }
            if inFence {
                fenceLines.append(raw)
                continue
            }
            if raw.isEmpty {
                flushParagraph()
                flushList()
                continue
            }
            if raw.hasPrefix("#") {
                flushParagraph()
                flushList()
                let level = min(raw.prefix { $0 == "#" }.count, 6)
                let text = inlineHTML(String(raw.dropFirst(level)).trimmingCharacters(in: .whitespaces))
                blocks.append("<h\(level)>\(text)</h\(level)>")
                continue
            }
            if raw.hasPrefix("&gt; ") || raw == "&gt;" {
                flushParagraph()
                flushList()
                let text = inlineHTML(raw == "&gt;" ? "" : String(raw.dropFirst(5)))
                blocks.append("<blockquote><p>\(text)</p></blockquote>")
                continue
            }
            if let unordered = unorderedItem(raw) {
                flushParagraph()
                if listOrdered && !listItems.isEmpty { flushList() }
                listOrdered = false
                listItems.append(unordered)
                continue
            }
            if let ordered = orderedItem(raw) {
                flushParagraph()
                if !listOrdered && !listItems.isEmpty { flushList() }
                listOrdered = true
                listItems.append(ordered)
                continue
            }
            flushList()
            paragraph.append(raw)
        }
        if inFence {
            blocks.append("<pre><code>\(fenceLines.joined(separator: "\n"))</code></pre>")
        }
        flushParagraph()
        flushList()

        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>\(escapeHTML(chat.title))</title>
        <style>
        body { font-family: -apple-system, "SF Pro Text", Helvetica, Arial, sans-serif;
               max-width: 820px; margin: 2rem auto; padding: 0 1.25rem;
               color: #1d1d1f; line-height: 1.55; }
        pre { background: #f5f5f7; border-radius: 8px; padding: 0.9rem 1rem;
              overflow-x: auto; font-size: 0.85em; }
        code { font-family: "SF Mono", Menlo, monospace; }
        blockquote { border-left: 3px solid #c8c8cc; margin-left: 0;
                     padding-left: 1rem; color: #555; }
        table { border-collapse: collapse; }
        td, th { border: 1px solid #d8d8dc; padding: 4px 10px; }
        h1, h2, h3, h4 { line-height: 1.25; }
        footer { margin-top: 3rem; color: #888; font-size: 0.8em;
                 border-top: 1px solid #e5e5ea; padding-top: 0.6rem; }
        </style>
        </head>
        <body>
        \(blocks.joined(separator: "\n"))
        <footer>Exported from TurboSpark</footer>
        </body>
        </html>
        """
    }

    private static func unorderedItem(_ line: String) -> String? {
        for marker in ["- ", "* ", "+ "] where line.hasPrefix(marker) {
            return String(line.dropFirst(marker.count))
        }
        return nil
    }

    private static func orderedItem(_ line: String) -> String? {
        guard let dot = line.firstIndex(of: "."),
            dot != line.startIndex,
            line[line.startIndex..<dot].allSatisfy(\.isNumber),
            line.index(after: dot) < line.endIndex
        else { return nil }
        return String(line[line.index(after: dot)...]).dropFirst().isEmpty
            ? nil : String(line[line.index(after: dot)...])
    }

    /// Inline pass: bold, italic, inline code, and links, AFTER escaping.
    /// Fenced blocks never reach here, so code spans are the only
    /// backtick content.
    private static func inlineHTML(_ text: String) -> String {
        var result = text
        // Inline code first, so its content is not styled twice.
        while let open = result.range(of: "`") {
            guard let close = result.range(of: "`", range: open.upperBound..<result.endIndex)
            else { break }
            let inner = String(result[open.upperBound..<close.lowerBound])
            result.replaceSubrange(
                open.lowerBound..<close.upperBound,
                with: "<code>\(inner)</code>")
        }
        for (marker, tag) in [("**", "strong"), ("*", "em")] {
            result = wrapAlternating(result, marker: marker, tag: tag)
        }
        result = linkify(result)
        return result
    }

    private static func wrapAlternating(_ text: String, marker: String, tag: String) -> String {
        var result = text
        var isOpening = true
        while let range = result.range(of: marker) {
            let replacement = isOpening ? "<\(tag)>" : "</\(tag)>"
            result.replaceSubrange(range, with: replacement)
            isOpening.toggle()
        }
        // An odd marker count leaves one unclosed tag; browsers render
        // past it as plain text, which is tolerable for an export.
        return result
    }

    private static func linkify(_ text: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: "https?://[^\\s<>()\"]+") else {
            return text
        }
        let ns = text as NSString
        var result = ""
        var cursor = 0
        regex.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
            guard let match else { return }
            result += ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            let url = ns.substring(with: match.range)
            result += "<a href=\"\(url)\">\(url)</a>"
            cursor = match.range.location + match.range.length
        }
        result += ns.substring(from: cursor)
        return result
    }

    static func escapeHTML(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&#39;")
    }

    private static func statusText(_ result: AppToolResult) -> String {
        result.isError ? "error" : "ok"
    }

    private static func argumentJSON(_ arguments: [String: String]) -> String {
        guard let data = try? JSONSerialization.data(
            withJSONObject: arguments, options: [.prettyPrinted, .sortedKeys]),
            let text = String(data: data, encoding: .utf8)
        else { return String(arguments.map { "\($0): \($1)" }.joined(separator: "\n").prefix(2000)) }
        return text
    }

    /// Tool output is evidence, not the payload; a runaway dump would make
    /// the export unreadable. Same tail rule the transcript's own output
    /// view applies, at a slightly higher cap.
    private static func truncate(_ output: String, maxChars: Int = 4000) -> String {
        guard output.count > maxChars else { return output }
        let tail = String(output.suffix(maxChars))
        return "...[truncated]\n\(tail)"
    }
}

extension AppModel {
    /// Asks for a destination and writes the export. Runs the save panel
    /// modally and toasts the outcome; returns without side effects when
    /// the user cancels.
    public func exportChat(_ chatID: UUID, format: AppChatExportFormat) {
        guard let index = chats.firstIndex(where: { $0.id == chatID }) else { return }
        let chat = chats[index]
        guard !chat.isGhost else {
            showToast("Temporary chats cannot be exported.", style: .warning)
            return
        }
        guard !chat.messages.isEmpty else {
            showToast("This chat has no conversation to export yet.", style: .warning)
            return
        }

        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [format.contentType]
        panel.nameFieldStringValue = sanitizedFileName(chat.title) + "." + format.fileExtension
        guard panel.runModal() == .OK, let url = panel.url else { return }

        do {
            switch format {
            case .markdown:
                try AppChatExport.markdown(for: chat, modelAlias: selected?.alias)
                    .write(to: url, atomically: true, encoding: .utf8)
            case .json:
                try AppChatExport.jsonData(for: chat).write(to: url, options: .atomic)
            case .html:
                try AppChatExport.html(for: chat, modelAlias: selected?.alias)
                    .write(to: url, atomically: true, encoding: .utf8)
            }
            showToast("Exported to \(url.lastPathComponent)", style: .success)
        } catch {
            showToast("Export failed: \(error.localizedDescription)", style: .error)
        }
    }

    /// Exports the SELECTED chat (the `/export` path).
    public func exportSelectedChat(format: AppChatExportFormat) {
        exportChat(selectedChatID, format: format)
    }

    private func sanitizedFileName(_ title: String) -> String {
        let invalid = CharacterSet(charactersIn: "/:\\?%*|\"<>")
        let cleaned = title.components(separatedBy: invalid).joined(separator: "-")
        let trimmed = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        let capped = String(trimmed.prefix(60))
        return capped.isEmpty ? "chat" : capped
    }
}
