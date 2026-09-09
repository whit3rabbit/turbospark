import Foundation

/// Keyword search over previous chats, behind the Cmd+K "Search Chats"
/// dialog.
///
/// Pure and free of SwiftUI so it can be unit tested (the `ModelHubFilter`
/// precedent). Documents are built from the in-memory `chats` array -- every
/// committed turn enters that array at the same moment it is persisted, so a
/// search needs no disk I/O and no index store -- and the matching is
/// deliberately the unsloth studio shape: prebuilt lowercased haystacks, a
/// whitespace-tokenized query, AND-of-substring semantics, and results in
/// plain recency order with no scoring.
///
/// **GHOST EXCLUSION IS BY THE FLAG, NOT BY EMPTINESS.** `buildDocuments`
/// skips every `isGhost` row and never touches `GhostChatVault`. The row of
/// a temporary chat carries no plaintext anyway, but the exclusion must not
/// DEPEND on that: a future change that puts content back on the row must
/// still not make a temporary chat surface in search results, because a
/// match would leak that the chat exists.
enum ChatSearch {
    /// A parsed query: lowercased, whitespace-separated tokens, ALL of which
    /// must appear as substrings. No operators, no phrases, no scoring --
    /// every token is matched independently of where it lands.
    struct Query: Equatable, Sendable {
        let tokens: [String]

        var isEmpty: Bool { tokens.isEmpty }

        static func parse(_ text: String) -> Query {
            Query(tokens: text.lowercased().split(whereSeparator: \.isWhitespace).map(String.init))
        }

        func matches(_ haystack: String) -> Bool {
            tokens.allSatisfy { haystack.contains($0) }
        }
    }

    /// How load-bearing a piece of searchable text is. A content match wins
    /// the snippet over a metadata match: "why a chat matched" should read
    /// like the conversation, not like tool plumbing.
    enum EntryTier: Equatable, Sendable {
        case content
        case metadata
    }

    /// One searchable text blob inside a chat, carried in both original
    /// casing (snippets are cut from this) and lowercased (matching runs on
    /// this, so a keystroke never re-lowercases a transcript).
    struct Entry: Equatable, Sendable {
        let tier: EntryTier
        /// What the blob came from: "message", "reasoning", "tool",
        /// "tool result" or "attachment". Not displayed; for tests.
        let kind: String
        let text: String
        let haystack: String
    }

    /// Everything searchable about one chat.
    struct Document: Equatable, Sendable {
        let id: UUID
        let title: String
        /// Carried so opening a hit can move to the chat's project first;
        /// the sidebar list is project-scoped.
        let projectID: UUID?
        let createdAt: Date
        let updatedAt: Date
        let titleHaystack: String
        let entries: [Entry]
    }

    /// A windowed excerpt of matching text, pre-cut into plain and
    /// highlighted segments so a view can color the matches without
    /// re-searching anything.
    struct Snippet: Equatable, Sendable {
        struct Segment: Equatable, Sendable {
            let text: String
            let isMatch: Bool
        }

        let segments: [Segment]
        let leadsWithEllipsis: Bool
        let trailsWithEllipsis: Bool

        var plainText: String {
            segments.map(\.text).joined()
        }
    }

    /// One result row.
    struct Hit: Identifiable, Equatable, Sendable {
        let id: UUID
        let title: String
        let projectID: UUID?
        let createdAt: Date
        let updatedAt: Date
        /// Matching entries, plus one when the title itself matches.
        let matchCount: Int
        /// nil for the empty query, where the dialog lists recent chats.
        let snippet: Snippet?
    }

    // MARK: - Building

    /// Builds one document per non-ghost chat. Ordering here does not
    /// matter; `hits` sorts.
    static func buildDocuments(from chats: [AppChat]) -> [Document] {
        chats.filter { !$0.isGhost }.map { chat in
            var entries: [Entry] = []
            for message in chat.messages {
                if !message.content.isEmpty {
                    entries.append(
                        Entry(
                            tier: .content, kind: "message", text: message.content,
                            haystack: message.content.lowercased()))
                }
                if !message.reasoning.isEmpty {
                    entries.append(
                        Entry(
                            tier: .metadata, kind: "reasoning", text: message.reasoning,
                            haystack: message.reasoning.lowercased()))
                }
                for call in message.toolCalls {
                    var parts = [call.name]
                    for (_, value) in call.arguments.sorted(by: { $0.key < $1.key }) {
                        parts.append(value)
                    }
                    if !call.rawInvocation.isEmpty {
                        parts.append(call.rawInvocation)
                    }
                    let text = parts.filter { !$0.isEmpty }.joined(separator: "\n")
                    if !text.isEmpty {
                        entries.append(
                            Entry(
                                tier: .metadata, kind: "tool", text: text,
                                haystack: text.lowercased()))
                    }
                }
                for result in message.toolResults where !result.output.isEmpty {
                    entries.append(
                        Entry(
                            tier: .metadata, kind: "tool result", text: result.output,
                            haystack: result.output.lowercased()))
                }
            }
            for attachment in chat.draftAttachments {
                let text = [attachment.fileName, attachment.extractedText]
                    .filter { !$0.isEmpty }.joined(separator: "\n")
                if !text.isEmpty {
                    entries.append(
                        Entry(
                            tier: .metadata, kind: "attachment", text: text,
                            haystack: text.lowercased()))
                }
            }
            return Document(
                id: chat.id,
                title: chat.title,
                projectID: chat.projectID,
                createdAt: chat.createdAt,
                updatedAt: chat.updatedAt,
                titleHaystack: chat.title.lowercased(),
                entries: entries)
        }
    }

    // MARK: - Matching

    /// Applies a query to documents. The empty query matches nothing and
    /// returns every document as a snippet-less recency row -- that is what
    /// the dialog shows before the user types. Results never carry a
    /// per-match score; recency is the whole order.
    static func hits(query: Query, documents: [Document]) -> [Hit] {
        let sorted = documents.sorted { $0.updatedAt > $1.updatedAt }
        guard !query.isEmpty else {
            return sorted.map { doc in
                Hit(
                    id: doc.id, title: doc.title, projectID: doc.projectID,
                    createdAt: doc.createdAt, updatedAt: doc.updatedAt,
                    matchCount: 0, snippet: nil)
            }
        }
        return sorted.compactMap { doc in
            var matchCount = 0
            var firstMatch: Entry?
            var firstContentMatch: Entry?
            for entry in doc.entries where query.matches(entry.haystack) {
                matchCount += 1
                if firstMatch == nil { firstMatch = entry }
                if firstContentMatch == nil, entry.tier == .content {
                    firstContentMatch = entry
                }
            }
            let titleMatches = query.matches(doc.titleHaystack)
            if titleMatches { matchCount += 1 }
            guard titleMatches || matchCount > 0 else { return nil }

            // Snippet source: a content entry first, then any entry, then
            // the title itself for a title-only match.
            let source = firstContentMatch ?? firstMatch
            let snippet = makeSnippet(for: source?.text ?? doc.title, tokens: query.tokens)
            return Hit(
                id: doc.id, title: doc.title, projectID: doc.projectID,
                createdAt: doc.createdAt, updatedAt: doc.updatedAt,
                matchCount: matchCount, snippet: snippet)
        }
    }

    /// Cuts a window out of `text` around the earliest token occurrence and
    /// marks every token occurrence inside the window. The original casing
    /// is preserved; matching is case-insensitive.
    static func makeSnippet(for text: String, tokens: [String]) -> Snippet {
        let plain = Snippet(
            segments: [Snippet.Segment(text: text, isMatch: false)],
            leadsWithEllipsis: false, trailsWithEllipsis: false)
        guard !text.isEmpty, !tokens.isEmpty else { return plain }

        guard let anchor = tokens.compactMap({ text.range(of: $0, options: .caseInsensitive) })
            .min(by: { $0.lowerBound < $1.lowerBound })
        else { return plain }

        // A short lead-in before the match, then a bounded window after it,
        // snapped outward to whitespace so words are not cut mid-token.
        let leadIn = text.index(
            anchor.lowerBound, offsetBy: -40, limitedBy: text.startIndex) ?? text.startIndex
        var windowStart = leadIn
        if windowStart > text.startIndex {
            // Walk forward to the next word boundary -- the first position
            // whose preceding character is whitespace, i.e. a word start --
            // and open the window THERE. No second step: a boundary before
            // the anchor is already past the whitespace, and stepping moved
            // the window one character INTO a word. When that boundary word
            // was the anchor itself (a 40-char lead-in with no whitespace
            // at all: a long URL, a path, CJK text), the step clipped the
            // MATCH's first character out of the window and lost the
            // highlight with it.
            while windowStart < anchor.lowerBound,
                !windowStart.isNextWhitespaceBoundary(in: text)
            {
                windowStart = text.index(after: windowStart)
            }
        }
        let windowEnd = text.index(
            anchor.lowerBound, offsetBy: 160, limitedBy: text.endIndex) ?? text.endIndex

        var matchRanges: [Range<String.Index>] = []
        for token in tokens {
            var cursor = windowStart
            while cursor < windowEnd,
                let range = text.range(
                    of: token, options: .caseInsensitive, range: cursor..<windowEnd)
            {
                matchRanges.append(range)
                cursor = range.upperBound
            }
        }
        let merged = mergeSortedTouchingRanges(matchRanges.sorted { $0.lowerBound < $1.lowerBound })

        var segments: [Snippet.Segment] = []
        if leadIn > text.startIndex { segments.append(.init(text: "\u{2026}", isMatch: false)) }
        var cursor = windowStart
        for range in merged {
            if cursor < range.lowerBound {
                segments.append(.init(text: String(text[cursor..<range.lowerBound]), isMatch: false))
            }
            segments.append(.init(text: String(text[range]), isMatch: true))
            cursor = range.upperBound
        }
        if cursor < windowEnd {
            segments.append(.init(text: String(text[cursor..<windowEnd]), isMatch: false))
        }
        if windowEnd < text.endIndex { segments.append(.init(text: "\u{2026}", isMatch: false)) }
        return Snippet(
            segments: segments,
            leadsWithEllipsis: leadIn > text.startIndex,
            trailsWithEllipsis: windowEnd < text.endIndex)
    }

    private static func mergeSortedTouchingRanges(_ ranges: [Range<String.Index>])
        -> [Range<String.Index>]
    {
        var merged: [Range<String.Index>] = []
        for range in ranges {
            guard var last = merged.last else {
                merged.append(range)
                continue
            }
            if range.lowerBound <= last.upperBound {
                if range.upperBound > last.upperBound {
                    last = last.lowerBound..<range.upperBound
                    merged[merged.count - 1] = last
                }
            } else {
                merged.append(range)
            }
        }
        return merged
    }

    // MARK: - Presentation values

    /// The relative date bucket a result row shows. Pure in `now` so tests
    /// can pin it.
    static func dateBucket(_ date: Date, now: Date = Date()) -> String {
        let day: TimeInterval = 86_400
        let age = now.timeIntervalSince(date)
        if age < day { return "Today" }
        if age < 7 * day { return "Past week" }
        if age < 30 * day { return "Past month" }
        return "Older"
    }
}

private extension String.Index {
    /// Whether the character BEFORE this index is whitespace -- a place a
    /// snippet window may open without cutting a word. Named against the
    /// String extension so it does not read as a String property.
    func isNextWhitespaceBoundary(in text: String) -> Bool {
        guard self > text.startIndex else { return true }
        return text[text.index(before: self)].isWhitespace
    }
}
