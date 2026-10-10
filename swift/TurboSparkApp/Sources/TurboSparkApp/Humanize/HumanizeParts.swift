import Foundation
import TurboSpark

/// One piece of a draft that the model rewrites in a single call.
struct HumanizePart: Equatable, Sendable {
    /// UTF-16 range of the part in the draft.
    let range: NSRange
    /// The part's tokens, summed over the paragraphs or sentences it joins.
    let tokens: Int
}

/// Which draft text produced which rewrite text, for highlighting.
struct HumanizeSegment: Equatable, Sendable {
    let source: NSRange
    let rewrite: NSRange
}

/// How a draft will be sent to the model.
struct HumanizePlan: Equatable, Sendable {
    /// The whole draft's tokens.
    let tokens: Int
    let parts: [HumanizePart]

    /// Every part fits the window with its prompt and output budget.
    var fits: Bool { parts.allSatisfy(HumanizePlanner.fits) }
}

/// Splits a long draft into parts the model can rewrite whole.
///
/// The model's window is 8,192 tokens and the publisher's output rule caps a
/// rewrite at 2,048 tokens, so one pass over a long draft either does not fit
/// or ends cut off. The publisher's own demo cuts long drafts at paragraph
/// breaks and rewrites each part on its own; this does the same, falling back
/// to sentence ends inside a paragraph that is too long by itself. The draft's
/// own text between parts is kept and re-inserted between their rewrites.
enum HumanizePlanner {
    static let window = 8192
    /// Instruction, separator and BOS, rounded up. The engine owns the exact
    /// prompt and checks it again before rewriting; this avoids keeping a
    /// second copy of the checkpoint-specific wrapper in Swift.
    static let promptReserve = 256
    /// The automatic output budget for 800 tokens is 2,000, under the 2,048
    /// cap, so every part keeps the publisher's full 2.5x allowance.
    static let partTokens = 800

    /// The engine's automatic output budget (`default_max_new` in core).
    static func outputTokens(for draftTokens: Int) -> Int {
        min(2048, max(256, draftTokens * 5 / 2))
    }

    static func fits(_ part: HumanizePart) -> Bool {
        part.tokens + promptReserve + outputTokens(for: part.tokens) <= window
    }

    static func plan(
        _ draft: String, count: (String) async throws -> Int
    ) async throws -> HumanizePlan {
        let text = draft as NSString
        let whole = NSRange(location: 0, length: text.length)
        guard trimmed(whole, in: text).length > 0 else { return HumanizePlan(tokens: 0, parts: []) }
        let total = try await count(draft)
        // The common case: one call, and the draft goes to the engine as typed.
        if total <= partTokens {
            return HumanizePlan(tokens: total, parts: [HumanizePart(range: whole, tokens: total)])
        }

        var pieces: [HumanizePart] = []
        for paragraph in paragraphs(in: whole, of: text) {
            let tokens = try await count(text.substring(with: paragraph))
            guard tokens > partTokens else {
                pieces.append(HumanizePart(range: paragraph, tokens: tokens))
                continue
            }
            for sentence in sentences(in: paragraph, of: text) {
                pieces.append(HumanizePart(range: sentence, tokens: try await count(text.substring(with: sentence))))
            }
        }

        // Greedy: join neighbors while they fit. A part's range runs from its
        // first piece to its last, so the breaks between them come along.
        var parts: [HumanizePart] = []
        var start: Int?
        var end = 0, tokens = 0
        for piece in pieces {
            if let first = start, tokens + piece.tokens > partTokens {
                parts.append(HumanizePart(range: NSRange(location: first, length: end - first), tokens: tokens))
                start = nil
                tokens = 0
            }
            if start == nil { start = piece.range.location }
            end = NSMaxRange(piece.range)
            tokens += piece.tokens
        }
        if let start {
            parts.append(HumanizePart(range: NSRange(location: start, length: end - start), tokens: tokens))
        }
        return HumanizePlan(tokens: total, parts: parts)
    }

    private static let paragraphBreak = try? NSRegularExpression(pattern: #"\n[ \t]*\n\s*"#)

    private static func paragraphs(in range: NSRange, of text: NSString) -> [NSRange] {
        guard let paragraphBreak else { return [trimmed(range, in: text)] }
        var out: [NSRange] = []
        var cursor = range.location
        for match in paragraphBreak.matches(in: text as String, range: range) {
            out.append(trimmed(NSRange(location: cursor, length: match.range.location - cursor), in: text))
            cursor = NSMaxRange(match.range)
        }
        out.append(trimmed(NSRange(location: cursor, length: NSMaxRange(range) - cursor), in: text))
        return out.filter { $0.length > 0 }
    }

    private static func sentences(in range: NSRange, of text: NSString) -> [NSRange] {
        var out: [NSRange] = []
        text.enumerateSubstrings(in: range, options: [.bySentences, .substringNotRequired]) { _, sentence, _, _ in
            let sentence = trimmed(sentence, in: text)
            if sentence.length > 0 { out.append(sentence) }
        }
        return out.isEmpty ? [range] : out
    }

    private static func trimmed(_ range: NSRange, in text: NSString) -> NSRange {
        var start = range.location, end = NSMaxRange(range)
        while start < end, isSpace(text.character(at: start)) { start += 1 }
        while end > start, isSpace(text.character(at: end - 1)) { end -= 1 }
        return NSRange(location: start, length: end - start)
    }

    private static func isSpace(_ unit: unichar) -> Bool {
        Unicode.Scalar(unit).map(CharacterSet.whitespacesAndNewlines.contains) ?? false
    }
}

/// Tokenizer IO and counting stay off the main actor, including large pastes.
actor HumanizeTokenCounter {
    private var tokenizer: TurboSparkTokenizer?
    private var path: String?

    func count(_ draft: String, at path: String) throws -> Int {
        if self.path != path || tokenizer == nil {
            tokenizer = try TurboSparkTokenizer(modelPath: path)
            self.path = path
        }
        return try tokenizer!.count(draft.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

/// Context calculation for a draft's prompt, draft tokens, and automatic output reservation.
struct HumanizeContextBudget: Equatable, Sendable {
    var draftTokens: Int
    var outputLimit: Int?

    var outputTokens: Int {
        let desired = max(256, draftTokens * 5 / 2)
        if let limit = outputLimit {
            return min(limit, desired)
        }
        return min(2048, desired)
    }

    var estimatedTotal: Int {
        draftTokens + HumanizePlanner.promptReserve + outputTokens
    }

    var fits: Bool {
        estimatedTotal <= HumanizePlanner.window
    }

    var fraction: Double {
        min(1.0, Double(estimatedTotal) / Double(HumanizePlanner.window))
    }
}
