import Foundation
import TurboSpark

/// One-shot rewrite of a short image idea into a richer text-to-image prompt
/// by the already-loaded LOCAL text model. Same shape as
/// `AppTitleGeneration`: a side completion on the chat session that never
/// adds a row to any chat. Remote providers are never used here, so the
/// prompt never leaves the Mac and no remote-consent path is needed.
enum ImagePromptEnhancement {
    /// Longest user prompt sent to the model; an image idea is short, and a
    /// pasted essay would be prefilled for nothing.
    static let promptCharacterLimit = 2_000
    /// Longest suggestion offered. Image text encoders truncate long prompts,
    /// and the system prompt asks for about 120 words.
    static let suggestionCharacterLimit = 1_000
    static let maxNewTokens: UInt32 = 320

    typealias Completion = @Sendable ([ChatMessage], GenerateOptions) async throws -> String

    /// Why Enhance can or cannot run. Each blocked case has its own tooltip,
    /// since a disabled control must say why (DESIGN.md, Action hierarchy).
    enum Availability: Equatable {
        case ready
        case emptyPrompt
        case noTextModel
        case textModelBusy
        case imageGenerating
        case temporaryChat
    }

    /// Ordered so the reason shown is the one the user can act on first: a
    /// missing model outranks a busy flag, and typing comes last.
    static func availability(
        prompt: String,
        textModelLoaded: Bool,
        textModelBusy: Bool,
        imageGenerating: Bool,
        temporaryChat: Bool
    ) -> Availability {
        if temporaryChat { return .temporaryChat }
        if !textModelLoaded { return .noTextModel }
        if textModelBusy { return .textModelBusy }
        if imageGenerating { return .imageGenerating }
        if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return .emptyPrompt }
        return .ready
    }

    static func messages(for prompt: String) -> [ChatMessage] {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return [
            ChatMessage.system("""
                You rewrite short image ideas into detailed prompts for a text-to-image model.
                Keep the subject and every detail the user gave. Add concrete visual detail: \
                setting, composition, lighting, colors, materials, mood and style.
                Do not add words or lettering to the image unless the user asked for them.
                Return only the prompt as one paragraph of plain text, under 120 words. \
                No title, label, quotes or explanation.
                """),
            ChatMessage.user(String(trimmed.prefix(promptCharacterLimit))),
        ]
    }

    static func options() -> GenerateOptions {
        var options = GenerateOptions()
        // Thinking would spend the token budget before the prompt starts.
        options.reasoning = .off
        // Some variety is the point of a rewrite; titles use 0.2 for stability.
        options.temperature = 0.7
        options.maxNewTokens = maxNewTokens
        return options
    }

    /// The production completion: a plain-text side request on the loaded
    /// local session, collecting only content (never reasoning or tools).
    static func localCompletion(_ session: TurboSparkSession) -> Completion {
        { messages, options in
            var response = ""
            for try await event in session.generate(messages, options: options) {
                try Task.checkCancellation()
                if case .content(let chunk) = event { response += chunk }
            }
            return response
        }
    }

    /// Failures and unusable output are auxiliary and return nil; the user's
    /// prompt is never touched here. The completion is injected so tests
    /// cover prompt settings and failure handling without a model.
    static func suggest(prompt: String, completion: Completion) async -> String? {
        do {
            let reply = try await completion(messages(for: prompt), options())
            return cleanedSuggestion(reply: reply, original: prompt)
        } catch {
            return nil
        }
    }

    // MARK: - Reply cleanup

    /// One paragraph of prompt text, or nil when the reply is empty, only
    /// reasoning, or an echo of the original.
    static func cleanedSuggestion(reply: String, original: String) -> String? {
        guard var text = removingThinking(reply) else { return nil }
        text = removingCodeFence(text.trimmingCharacters(in: .whitespacesAndNewlines))

        // A preface line ("Sure! Here's a richer version:") ends with a colon
        // and carries no prompt; trailing notes are later paragraphs.
        var paragraphs = text
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if paragraphs.count > 1, paragraphs[0].hasSuffix(":") { paragraphs.removeFirst() }
        text = paragraphs.first ?? ""

        // Labels and wrapping can nest ("**Prompt:** \"...\"").
        for _ in 0..<3 {
            let next = removingWrapping(removingLabel(text))
            if next == text { break }
            text = next
        }

        text = collapsedWhitespace(text)
        guard !text.isEmpty else { return nil }
        guard text.caseInsensitiveCompare(collapsedWhitespace(original)) != .orderedSame else {
            return nil
        }
        return capped(text)
    }

    /// Nil when the reply is all reasoning (an unclosed think block).
    private static func removingThinking(_ reply: String) -> String? {
        var text = replacing(#"<think>[\s\S]*?</think>"#, in: reply, with: "")
        // A template that opens the block itself leaves only the close tag.
        if let close = text.range(of: "</think>", options: [.backwards, .caseInsensitive]) {
            text = String(text[close.upperBound...])
        }
        if text.range(of: "<think>", options: .caseInsensitive) != nil { return nil }
        return text
    }

    private static func removingCodeFence(_ text: String) -> String {
        guard text.hasPrefix("```"), text.hasSuffix("```"), text.count >= 6 else { return text }
        var inner = text.dropFirst(3).dropLast(3)
        // Drop a language tag on the opening line ("```text").
        if let newline = inner.firstIndex(of: "\n"),
            !inner[..<newline].contains(where: \.isWhitespace) {
            inner = inner[inner.index(after: newline)...]
        }
        return inner.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func removingLabel(_ text: String) -> String {
        let pattern = #"^\s*(?:\*\*|__)?\s*(?:here(?:'s| is) (?:the |an? |your )?)?"#
            + #"(?:(?:enhanced|improved|rewritten|detailed|refined|final|new|image|richer|more|revised|expanded)\s+)*"#
            + #"prompt\s*(?:\*\*|__)?\s*:\s*(?:\*\*|__)?\s*"#
        return replacing(pattern, in: text, with: "")
    }

    /// Strips one matching pair of quotes or backticks around the whole text,
    /// but not when the closing mark also appears inside: `"Open" sign ...
    /// "Closed"` is a prompt that quotes, not a quoted prompt.
    private static func removingWrapping(_ text: String) -> String {
        let pairs: [(Character, Character)] = [
            ("\"", "\""), ("'", "'"), ("`", "`"),
            ("\u{201C}", "\u{201D}"), ("\u{2018}", "\u{2019}"),
        ]
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        for (open, close) in pairs where trimmed.count >= 2
            && trimmed.first == open && trimmed.last == close {
            let inner = trimmed.dropFirst().dropLast()
            guard !inner.contains(close), !inner.contains(open) else { return trimmed }
            return inner.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return trimmed
    }

    private static func collapsedWhitespace(_ text: String) -> String {
        text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// Cut at the last word boundary inside the limit, so the field never
    /// ends mid-word.
    private static func capped(_ text: String) -> String {
        guard text.count > suggestionCharacterLimit else { return text }
        var head = String(text.prefix(suggestionCharacterLimit))
        if let space = head.lastIndex(of: " ") { head = String(head[..<space]) }
        return head.trimmingCharacters(in: CharacterSet(charactersIn: " ,;:"))
    }

    private static func replacing(_ pattern: String, in text: String, with template: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return text
        }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, range: range, withTemplate: template)
    }
}
