import Foundation

/// An interactive markdown table, the qwen-code `EnhancedMarkdownTable`
/// subset this app ports: column sorting, cell-range copy as TSV/CSV or
/// original markdown, and CSV export. The richer React surface's column
/// reordering, per-type filters, and density modes are NOT ported; the
/// subset is what a native transcript row wants.
///
/// Pure value type: parse, sort and serialize are all testable without a
/// view, and the view holds only the selection and sort-direction state.
struct MarkdownTableGrid: Equatable {
    let header: [String]
    let rows: [[String]]

    var columnCount: Int { header.count }
    var rowCount: Int { rows.count }

    /// Worth the interactive affordance: a one-row table has nothing to
    /// sort and little to copy.
    var isInteractiveCandidate: Bool { rowCount > 1 && columnCount > 0 }

    // MARK: - Parsing

    /// Splits message markdown into prose and table segments, in order, so
    /// a renderer can pass the prose to the markdown view and give each
    /// table the interactive treatment. A table is a header line and a
    /// separator line that are both pipe-delimited, followed by contiguous
    /// pipe-delimited rows.
    static func segments(in markdown: String) -> [Segment] {
        var result: [Segment] = []
        let lines = markdown.components(separatedBy: "\n")
        var prose: [String] = []
        var index = 0
        func flushProse() {
            if !prose.isEmpty {
                result.append(.markdown(prose.joined(separator: "\n")))
                prose = []
            }
        }
        while index < lines.count {
            guard index + 1 < lines.count,
                isPipeRow(lines[index]),
                isSeparatorRow(lines[index + 1])
            else {
                prose.append(lines[index])
                index += 1
                continue
            }
            flushProse()
            var tableLines = [lines[index], lines[index + 1]]
            var cursor = index + 2
            while cursor < lines.count, isPipeRow(lines[cursor]) {
                tableLines.append(lines[cursor])
                cursor += 1
            }
            if let grid = parse(tableLines) {
                result.append(.table(grid))
            } else {
                result.append(.markdown(tableLines.joined(separator: "\n")))
            }
            index = cursor
        }
        flushProse()
        return result
    }

    enum Segment: Equatable {
        case markdown(String)
        case table(MarkdownTableGrid)
    }

    /// Parses pipe rows into a grid. Cells split on unescaped pipes; `\|`
    /// is a literal pipe in a cell. Ragged rows are PADDED to the header
    /// width (a markdown table with a missing trailing cell is still a
    /// table), never truncated -- data a row did carry stays.
    static func parse(_ lines: [String]) -> MarkdownTableGrid? {
        guard lines.count >= 2, isSeparatorRow(lines[1]) else { return nil }
        let header = cells(in: lines[0])
        guard !header.isEmpty else { return nil }
        let rows = lines.dropFirst(2).map { cells(in: $0) }.map { row -> [String] in
            row.count >= header.count ? row : row + Array(repeating: "", count: header.count - row.count)
        }
        return MarkdownTableGrid(header: header, rows: rows)
    }

    private static func isPipeRow(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        return trimmed.hasPrefix("|") && trimmed.hasSuffix("|") && trimmed.contains("|")
    }

    private static func isSeparatorRow(_ line: String) -> Bool {
        let cells = cells(in: line)
        guard !cells.isEmpty else { return false }
        return cells.allSatisfy { cell in
            !cell.isEmpty && cell.allSatisfy { $0 == "-" || $0 == ":" }
        }
    }

    private static func cells(in line: String) -> [String] {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        var inner = Substring(trimmed)
        if inner.hasPrefix("|") { inner = inner.dropFirst() }
        if inner.hasSuffix("|") { inner = inner.dropLast() }
        var cells: [String] = []
        var current = ""
        var escaped = false
        for character in inner {
            if escaped {
                // A backslash escapes only the pipe for cell-splitting
                // purposes; any other character keeps its backslash, which
                // is markdown's own rule.
                if character == "|" {
                    current.append("|")
                } else {
                    current.append("\\")
                    current.append(character)
                }
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == "|" {
                cells.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            } else {
                current.append(character)
            }
        }
        if escaped { current.append("\\") }
        cells.append(current.trimmingCharacters(in: .whitespaces))
        return cells
    }

    // MARK: - Sorting

    enum SortDirection: Equatable {
        case ascending
        case descending

        var toggled: SortDirection { self == .ascending ? .descending : .ascending }
    }

    /// Rows sorted by one column; the header never moves. Numeric-looking
    /// columns compare numerically (so 9 < 10), everything else compares
    /// case-insensitively. An index past the end returns self unchanged.
    func sorted(byColumn column: Int, direction: SortDirection) -> MarkdownTableGrid {
        guard column >= 0, column < columnCount else { return self }
        let sorted = rows.sorted { lhs, rhs in
            let left = lhs[safe: column] ?? ""
            let right = rhs[safe: column] ?? ""
            let order: Bool
            if let leftNumber = Double(left.replacingOccurrences(of: ",", with: "")),
                let rightNumber = Double(right.replacingOccurrences(of: ",", with: "")) {
                order = leftNumber < rightNumber
            } else {
                order = left.localizedCaseInsensitiveCompare(right) == .orderedAscending
            }
            return direction == .ascending ? order : !order
        }
        return MarkdownTableGrid(header: header, rows: sorted)
    }

    // MARK: - Serialization

    /// Tab-separated, what a spreadsheet paste wants. Newlines and tabs
    /// inside a cell are escaped so the grid shape survives the paste.
    func tsv(range: Range<Int>? = nil) -> String {
        delimited("\t", escapeTabNewline: true, rows: range.map { Array(rows[$0]) } ?? rows)
    }

    func csv() -> String {
        delimited(",", escapeTabNewline: false, rows: rows)
    }

    /// The original markdown again; the copy-the-source escape hatch.
    var markdown: String {
        var lines = ["| " + header.joined(separator: " | ") + " |"]
        lines.append("| " + header.map { _ in "---" }.joined(separator: " | ") + " |")
        for row in rows {
            let padded = row + Array(repeating: "", count: max(0, columnCount - row.count))
            lines.append("| " + padded.prefix(columnCount).joined(separator: " | ") + " |")
        }
        return lines.joined(separator: "\n")
    }

    private func delimited(
        _ separator: String, escapeTabNewline: Bool, rows rowsToWrite: [[String]]
    ) -> String {
        func cell(_ value: String) -> String {
            if escapeTabNewline {
                return value
                    .replacingOccurrences(of: "\t", with: "\\t")
                    .replacingOccurrences(of: "\n", with: "\\n")
            }
            let needsQuotes = value.contains(separator) || value.contains("\"")
                || value.contains("\n")
            guard needsQuotes else { return value }
            return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        var lines = [header.map(cell).joined(separator: separator)]
        for row in rowsToWrite {
            let padded = row + Array(repeating: "", count: max(0, columnCount - row.count))
            lines.append(padded.prefix(columnCount).map(cell).joined(separator: separator))
        }
        return lines.joined(separator: "\n")
    }
}
