import Foundation

/// One reviewable difference: draft text and the rewrite text that replaced
/// it, as UTF-16 ranges. Either side may be empty (pure insertion or cut).
struct HumanizeChange: Sendable, Equatable {
    let source: NSRange
    let rewrite: NSRange
}

/// Display-only differences between a draft and its rewrite. Text and
/// whitespace are never rewritten by the comparison. The clipboard receives
/// the rewrite, or `compose` of it with the changes the user rejected; neither
/// the comparison nor the composition reaches the model.
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
    /// Every place the texts differ, in order, so rejecting all of them gives
    /// back the draft exactly. Unlike `inserted` and `deleted` this includes
    /// spacing and punctuation-only differences, which are not highlighted.
    /// Empty when the texts are the same, or when the invariant check failed
    /// (per-change review is then off and Copy sends the whole rewrite).
    var changes: [HumanizeChange] = []

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
        var changes: [HumanizeChange] = []
        var newWeight = 0, totalWeight = 0
        // Text between parts (the draft's own breaks, copied as written) and
        // outside them (trimmed outer whitespace, a stopped run's unfinished
        // tail against the draft it never reached) becomes a change only
        // where it differs, so rejecting every change is exactly the draft.
        var sourceEnd = 0, rewriteEnd = 0
        // Segments arrive in order; one that does not turns review off rather
        // than splicing out of order. The highlights are unaffected.
        var ordered = true
        for segment in segments
        where NSMaxRange(segment.source) <= original.length && NSMaxRange(segment.rewrite) <= output.length {
            if segment.source.location < sourceEnd || segment.rewrite.location < rewriteEnd { ordered = false }
            if ordered, let gap = change(original, NSRange(location: sourceEnd, length: segment.source.location - sourceEnd),
                                output, NSRange(location: rewriteEnd, length: segment.rewrite.location - rewriteEnd)) {
                changes.append(gap)
            }
            let part = compare(source: original.substring(with: segment.source),
                               rewrite: output.substring(with: segment.rewrite))
            inserted += part.inserted.map { NSRange(location: $0.location + segment.rewrite.location, length: $0.length) }
            deleted += part.deleted.map { NSRange(location: $0.location + segment.source.location, length: $0.length) }
            changes += part.changes.map {
                HumanizeChange(
                    source: NSRange(location: $0.source.location + segment.source.location, length: $0.source.length),
                    rewrite: NSRange(location: $0.rewrite.location + segment.rewrite.location, length: $0.rewrite.length))
            }
            newWeight += part.newWeight
            totalWeight += part.totalWeight
            sourceEnd = NSMaxRange(segment.source)
            rewriteEnd = NSMaxRange(segment.rewrite)
        }
        if ordered, let tail = change(original, NSRange(location: sourceEnd, length: original.length - sourceEnd),
                             output, NSRange(location: rewriteEnd, length: output.length - rewriteEnd)) {
            changes.append(tail)
        }
        return HumanizeDiff(inserted: inserted, deleted: deleted, newWeight: newWeight, totalWeight: totalWeight,
                            changes: ordered ? verified(changes, source: source, rewrite: rewrite) : [])
    }

    /// The copied text: the rewrite, with each rejected change's span put back
    /// to the draft's wording. Text between changes comes from the rewrite.
    /// Nil when the changes are out of order, overlap, or fall outside either
    /// text, rather than guessing at a splice.
    static func compose(source: String, rewrite: String, changes: [HumanizeChange], rejected: Set<Int>) -> String? {
        let original = source as NSString
        let output = rewrite as NSString
        var result = ""
        var sourceEnd = 0, rewriteEnd = 0
        for (index, change) in changes.enumerated() {
            guard change.source.location >= sourceEnd, change.rewrite.location >= rewriteEnd,
                  NSMaxRange(change.source) <= original.length, NSMaxRange(change.rewrite) <= output.length
            else { return nil }
            result += output.substring(with: NSRange(location: rewriteEnd, length: change.rewrite.location - rewriteEnd))
            result += rejected.contains(index)
                ? original.substring(with: change.source) : output.substring(with: change.rewrite)
            sourceEnd = NSMaxRange(change.source)
            rewriteEnd = NSMaxRange(change.rewrite)
        }
        result += output.substring(from: rewriteEnd)
        return result
    }

    /// Keeps the changes only if they reproduce both texts. A failure would
    /// mean Reject All copies something that is not the draft, so review is
    /// turned off instead.
    private static func verified(_ changes: [HumanizeChange], source: String, rewrite: String) -> [HumanizeChange] {
        guard compose(source: source, rewrite: rewrite, changes: changes, rejected: []) == rewrite,
              compose(source: source, rewrite: rewrite, changes: changes, rejected: Set(changes.indices)) == source
        else { return [] }
        return changes
    }

    /// The change between two regions, nil when they are identical. Spacing
    /// both sides share at the edges is left out, so a change covers the
    /// wording and not the spaces around it; a spacing-only difference stays
    /// whole.
    private static func change(_ source: NSString, _ sourceRange: NSRange,
                               _ rewrite: NSString, _ rewriteRange: NSRange) -> HumanizeChange? {
        let before = source.substring(with: sourceRange), after = rewrite.substring(with: rewriteRange)
        guard before != after else { return nil }
        let a = Array(before.utf16), b = Array(after.utf16)
        func space(_ unit: UInt16) -> Bool {
            guard let scalar = Unicode.Scalar(unit) else { return false }
            return CharacterSet.whitespacesAndNewlines.contains(scalar)
        }
        var lead = 0
        while lead < a.count, lead < b.count, a[lead] == b[lead], space(a[lead]) { lead += 1 }
        var trail = 0
        while trail < a.count - lead, trail < b.count - lead,
              a[a.count - 1 - trail] == b[b.count - 1 - trail], space(a[a.count - 1 - trail]) { trail += 1 }
        // A spacing-only difference trimmed to nothing on both sides would
        // vanish, so keep it whole.
        if a.count - lead - trail == 0, b.count - lead - trail == 0 { lead = 0; trail = 0 }
        return HumanizeChange(
            source: NSRange(location: sourceRange.location + lead, length: a.count - lead - trail),
            rewrite: NSRange(location: rewriteRange.location + lead, length: b.count - lead - trail))
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
        let ops = cleanup(align(before, after), before, after)
        for op in ops {
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
        let changes = changes(ops, before, after, source: source as NSString, rewrite: rewrite as NSString)
        return HumanizeDiff(inserted: spans(newMarks, after, in: rewrite),
                            deleted: spans(oldMarks, before, in: source),
                            newWeight: new, totalWeight: total,
                            changes: verified(changes, source: source, rewrite: rewrite))
    }

    /// Splits both texts at tokens the alignment kept and that are spelled
    /// exactly alike. Each region between two such anchors that differs is
    /// one change, so an absorbed short run joins its neighbors (as in the
    /// highlight) and a curly-quote or spacing difference is still covered.
    private static func changes(_ ops: [Op], _ before: [Token], _ after: [Token],
                                source: NSString, rewrite: NSString) -> [HumanizeChange] {
        var result: [HumanizeChange] = []
        var sourceEnd = 0, rewriteEnd = 0
        for case .same(let i, let j) in ops
        where source.substring(with: before[i].range) == rewrite.substring(with: after[j].range) {
            if let found = change(source, NSRange(location: sourceEnd, length: before[i].range.location - sourceEnd),
                                  rewrite, NSRange(location: rewriteEnd, length: after[j].range.location - rewriteEnd)) {
                result.append(found)
            }
            sourceEnd = NSMaxRange(before[i].range)
            rewriteEnd = NSMaxRange(after[j].range)
        }
        if let found = change(source, NSRange(location: sourceEnd, length: source.length - sourceEnd),
                              rewrite, NSRange(location: rewriteEnd, length: rewrite.length - rewriteEnd)) {
            result.append(found)
        }
        return result
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
