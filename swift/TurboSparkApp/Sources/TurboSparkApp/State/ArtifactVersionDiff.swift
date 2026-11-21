import Foundation

/// The line diff behind "Compare with Current" in the artifact panel's
/// history: an earlier version against the file on disk now. Read-only and
/// pure; the panel computes it off the main actor and only draws the rows.
///
/// Matching is the standard library's `difference(from:)` (the same one
/// `FormatterRunner.change` uses), so unchanged lines are a longest common
/// subsequence and every other line is a removal or an addition.
enum ArtifactVersionDiff {
    enum Kind: Equatable, Sendable { case unchanged, removed, added }

    struct Line: Equatable, Sendable {
        var kind: Kind
        /// 1-based line in the earlier version; nil for an added line.
        var oldNumber: Int?
        /// 1-based line in the current file; nil for a removed line.
        var newNumber: Int?
        var text: String
    }

    enum Row: Equatable, Sendable {
        case line(Line)
        /// A run of unchanged lines folded to its count.
        case skipped(Int)
    }

    /// Either side longer than this is not compared: the matcher's cost
    /// grows with lines times differences, and the panel would stall.
    static let maxLines = 5_000
    /// Unchanged lines kept beside each change when a run is folded.
    static let contextLines = 3

    /// Lines as an editor counts them: a trailing newline adds no line, and
    /// CRLF and LF text compare equal. "\r\n" is one Character in Swift, so
    /// it is a separator of its own rather than a CR left on the line.
    static func splitLines(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        var lines = text.split(
            omittingEmptySubsequences: false, whereSeparator: { $0 == "\n" || $0 == "\r\n" }
        ).map { line -> String in
            line.hasSuffix("\r") ? String(line.dropLast()) : String(line)
        }
        if text.hasSuffix("\n") || text.hasSuffix("\r\n") { lines.removeLast() }
        return lines
    }

    /// Every line of both texts in reading order, removals before the
    /// additions that replace them, or nil when either side is over
    /// `maxLines`.
    static func lines(old: String, new: String) -> [Line]? {
        let before = splitLines(old)
        let after = splitLines(new)
        guard before.count <= maxLines, after.count <= maxLines else { return nil }
        var removed = Set<Int>()
        var inserted = Set<Int>()
        for change in after.difference(from: before) {
            switch change {
            case .remove(let offset, _, _): removed.insert(offset)
            case .insert(let offset, _, _): inserted.insert(offset)
            }
        }
        var out: [Line] = []
        out.reserveCapacity(before.count + inserted.count)
        var i = 0
        var j = 0
        while i < before.count || j < after.count {
            if i < before.count, removed.contains(i) {
                out.append(Line(kind: .removed, oldNumber: i + 1, newNumber: nil, text: before[i]))
                i += 1
            } else if j < after.count, inserted.contains(j) {
                out.append(Line(kind: .added, oldNumber: nil, newNumber: j + 1, text: after[j]))
                j += 1
            } else if i < before.count, j < after.count {
                out.append(Line(kind: .unchanged, oldNumber: i + 1, newNumber: j + 1, text: after[j]))
                i += 1
                j += 1
            } else {
                // Unreachable for a consistent difference; stop rather than spin.
                break
            }
        }
        return out
    }

    static func hasChanges(_ lines: [Line]) -> Bool {
        lines.contains { $0.kind != .unchanged }
    }

    static func hasChanges(_ rows: [Row]) -> Bool {
        rows.contains { row in
            if case .line(let line) = row { return line.kind != .unchanged }
            return false
        }
    }

    /// `lines` with each unchanged run folded down to `context` lines beside
    /// the neighboring changes. A run is shown whole when folding would hide
    /// fewer than two lines, since a one-line fold takes the line's own row.
    static func rows(_ lines: [Line], context: Int = contextLines) -> [Row] {
        var rows: [Row] = []
        var index = 0
        while index < lines.count {
            guard lines[index].kind == .unchanged else {
                rows.append(.line(lines[index]))
                index += 1
                continue
            }
            let start = index
            while index < lines.count, lines[index].kind == .unchanged { index += 1 }
            let run = lines[start..<index]
            let keepHead = start > 0 ? context : 0
            let keepTail = index < lines.count ? context : 0
            let hidden = run.count - keepHead - keepTail
            if hidden < 2 {
                rows += run.map(Row.line)
            } else {
                rows += run.prefix(keepHead).map(Row.line)
                rows.append(.skipped(hidden))
                rows += run.suffix(keepTail).map(Row.line)
            }
        }
        return rows
    }
}
