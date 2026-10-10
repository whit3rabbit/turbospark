import Foundation

/// Display-only differences between a draft and its rewrite. Text and
/// whitespace are never rewritten by the comparison, and the clipboard always
/// receives the original result string.
///
/// Follows the publisher's display diff (`space/diffmark.py` in
/// sgaofen/humanizer-local-model): words compare whole, Han and kana one
/// character at a time, punctuation on its own, and whitespace not at all.
struct HumanizeDiff: Sendable, Equatable {
    /// New wording, as UTF-16 ranges of the rewrite.
    let inserted: [NSRange]
    /// Wording the rewrite dropped, as UTF-16 ranges of the draft.
    let deleted: [NSRange]
    /// Content weight of the rewrite's new wording, and of the whole rewrite.
    let newWeight: Int
    let totalWeight: Int

    static let empty = HumanizeDiff(inserted: [], deleted: [], newWeight: 0, totalWeight: 0)

    /// Share of the rewrite that is new wording, 0...1.
    var changedShare: Double { totalWeight > 0 ? Double(newWeight) / Double(totalWeight) : 0 }

    /// Compares each rewritten part with the draft text it came from. Parts
    /// are rewritten independently, so pairing them keeps every comparison
    /// small and word-level even on a long document. A part with no segment
    /// (a stopped run's unfinished tail) is not compared at all, rather than
    /// marking the draft it never reached as cut.
    static func compare(source: String, rewrite: String, segments: [HumanizeSegment]) -> HumanizeDiff {
        let original = source as NSString
        let output = rewrite as NSString
        var inserted: [NSRange] = [], deleted: [NSRange] = []
        var newWeight = 0, totalWeight = 0
        for segment in segments
        where NSMaxRange(segment.source) <= original.length && NSMaxRange(segment.rewrite) <= output.length {
            let part = compare(source: original.substring(with: segment.source),
                               rewrite: output.substring(with: segment.rewrite))
            inserted += part.inserted.map { NSRange(location: $0.location + segment.rewrite.location, length: $0.length) }
            deleted += part.deleted.map { NSRange(location: $0.location + segment.source.location, length: $0.length) }
            newWeight += part.newWeight
            totalWeight += part.totalWeight
        }
        return HumanizeDiff(inserted: inserted, deleted: deleted, newWeight: newWeight, totalWeight: totalWeight)
    }

    static func compare(source: String, rewrite: String) -> HumanizeDiff {
        var before = tokens(source)
        var after = tokens(rewrite)
        // Bound work on very large pastes. Paragraph alignment still shows
        // changed passages without running a quadratic word diff on a novel.
        if before.count > 4000 || after.count > 4000 || before.count * after.count > 4_000_000 {
            before = paragraphs(source)
            after = paragraphs(rewrite)
        }
        var oldMarks = [Bool](repeating: false, count: before.count)
        var newMarks = [Bool](repeating: false, count: after.count)
        for op in cleanup(align(before, after), before, after) {
            switch op {
            case .delete(let i): oldMarks[i] = true
            case .insert(let j): newMarks[j] = true
            case .same: break
            }
        }
        quietPunctuation(&oldMarks, before)
        quietPunctuation(&newMarks, after)
        let total = after.reduce(0) { $0 + $1.weight }
        let new = zip(after, newMarks).reduce(0) { $0 + ($1.1 ? $1.0.weight : 0) }
        return HumanizeDiff(inserted: spans(newMarks, after, in: rewrite),
                            deleted: spans(oldMarks, before, in: source),
                            newWeight: new, totalWeight: total)
    }

    private struct Token {
        let range: NSRange
        let key: String
        /// Characters, or 2 for a single Han or kana character, so a changed
        /// Chinese clause weighs about what the same English clause does.
        let weight: Int
        let isPunctuation: Bool
    }

    private enum Op { case same(Int, Int), delete(Int), insert(Int) }

    private static func align(_ before: [Token], _ after: [Token]) -> [Op] {
        var removed = Set<Int>(), added = Set<Int>()
        for change in after.map(\.key).difference(from: before.map(\.key)) {
            switch change {
            case .remove(let offset, _, _): removed.insert(offset)
            case .insert(let offset, _, _): added.insert(offset)
            }
        }
        var ops: [Op] = []
        var i = 0, j = 0
        while i < before.count || j < after.count {
            if i < before.count, removed.contains(i) {
                ops.append(.delete(i)); i += 1
            } else if j < after.count, added.contains(j) {
                ops.append(.insert(j)); j += 1
            } else {
                ops.append(.same(i, j)); i += 1; j += 1
            }
        }
        return ops
    }

    private struct Group {
        var same: Bool
        var ops: [Op] = []
        var kept = 0, deleted = 0, inserted = 0
    }

    /// An unchanged run sitting between two changes, and shorter than half of
    /// the change on each side, counts as changed. Otherwise a deep rewrite is
    /// shredded into confetti by small words ("the", a comma) that happen to
    /// line up.
    private static func cleanup(_ ops: [Op], _ before: [Token], _ after: [Token]) -> [Op] {
        var ops = ops
        for _ in 0..<6 {
            var groups: [Group] = []
            for op in ops {
                let same: Bool
                if case .same = op { same = true } else { same = false }
                if groups.last?.same != same { groups.append(Group(same: same)) }
                groups[groups.count - 1].ops.append(op)
                switch op {
                case .same(let i, _): groups[groups.count - 1].kept += before[i].weight
                case .delete(let i): groups[groups.count - 1].deleted += before[i].weight
                case .insert(let j): groups[groups.count - 1].inserted += after[j].weight
                }
            }
            var changed = false
            if groups.count >= 3 {
                for k in 1..<(groups.count - 1) where groups[k].same {
                    let run = groups[k].kept * 2
                    let p = groups[k - 1], q = groups[k + 1]
                    // The publisher's rule, kept as-is: the larger side of
                    // each neighboring change sets the bar.
                    guard run <= max(p.deleted, p.inserted), run <= max(q.deleted, q.inserted) else { continue }
                    groups[k].ops = groups[k].ops.flatMap { op -> [Op] in
                        guard case .same(let i, let j) = op else { return [op] }
                        return [.delete(i), .insert(j)]
                    }
                    groups[k].same = false
                    changed = true
                }
            }
            ops = groups.flatMap(\.ops)
            if !changed { break }
        }
        return ops
    }

    /// A change that is only one or two punctuation marks is not marked.
    private static func quietPunctuation(_ marks: inout [Bool], _ tokens: [Token]) {
        var i = 0
        while i < marks.count {
            guard marks[i] else { i += 1; continue }
            var j = i
            while j < marks.count, marks[j] { j += 1 }
            if j - i <= 2, tokens[i..<j].allSatisfy(\.isPunctuation) {
                for k in i..<j { marks[k] = false }
            }
            i = j
        }
    }

    /// Marked tokens joined across spaces into one stroke; a line break never
    /// joins.
    private static func spans(_ marks: [Bool], _ tokens: [Token], in text: String) -> [NSRange] {
        let string = text as NSString
        var ranges: [NSRange] = []
        for (token, marked) in zip(tokens, marks) where marked {
            if let last = ranges.last {
                let gap = NSRange(location: NSMaxRange(last), length: token.range.location - NSMaxRange(last))
                if string.substring(with: gap).allSatisfy({ $0.isWhitespace && !$0.isNewline }) {
                    ranges[ranges.count - 1] = NSRange(
                        location: last.location, length: NSMaxRange(token.range) - last.location)
                    continue
                }
            }
            ranges.append(token.range)
        }
        return ranges
    }

    private static func tokens(_ text: String) -> [Token] {
        // Grouped numbers (1,250 / 10:30 / 3.5) stay one unit; Han and kana are
        // individual units; composed Unicode words keep their UTF-16 ranges.
        matches(#"\d+(?:[.,:/]\d+)+|[\p{Han}\p{Hiragana}\p{Katakana}]|[\p{L}\p{M}\p{N}]+(?:['\u2019][\p{L}\p{M}\p{N}]+)*|[^\s]"#,
                in: text)
    }

    private static func paragraphs(_ text: String) -> [Token] {
        matches(#"[^\r\n]+"#, in: text)
    }

    private static let cjk = try? NSRegularExpression(pattern: #"^[\p{Han}\p{Hiragana}\p{Katakana}]$"#)
    private static let punctuation = CharacterSet.punctuationCharacters.union(.symbols)

    private static func matches(_ pattern: String, in text: String) -> [Token] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let string = text as NSString
        return regex.matches(in: text, range: NSRange(location: 0, length: string.length)).map { match in
            let piece = string.substring(with: match.range)
            let single = NSRange(location: 0, length: (piece as NSString).length)
            return Token(
                range: match.range,
                key: normalized(piece),
                weight: cjk?.firstMatch(in: piece, range: single) != nil ? 2 : piece.count,
                isPunctuation: piece.unicodeScalars.allSatisfy(punctuation.contains))
        }
    }

    /// Compatibility forms and curly quotes compare equal to their plain
    /// spellings, so a rewrite that only changed quote style is not marked.
    private static func normalized(_ piece: String) -> String {
        piece.precomposedStringWithCompatibilityMapping
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{2018}", with: "'")
            .replacingOccurrences(of: "\u{201C}", with: "\"")
            .replacingOccurrences(of: "\u{201D}", with: "\"")
    }
}
