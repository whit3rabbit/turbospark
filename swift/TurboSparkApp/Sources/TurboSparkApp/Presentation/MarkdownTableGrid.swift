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
    ///
    /// FENCED CODE BLOCKS ARE NEVER TABLES. The scan tracks ```/~~~ fences
    /// and treats their contents as prose: without that, an example table
    /// inside a code block is detected, and the segmentation cuts the fence
    /// in half -- the markdown renderer then receives an unclosed fence as
    /// one segment and a stray ``` as another, which swallows the message's
    /// remaining prose.
    static func segments(in markdown: String) -> [Segment] {
        var result: [Segment] = []
        let lines = markdown.components(separatedBy: "\n")
        var prose: [String] = []
        var index = 0
        /// The fence currently open, if any: its delimiter character and
        /// run length.
        var openFence: (character: Character, length: Int)? = nil
        func flushProse() {
            if !prose.isEmpty {
                result.append(.markdown(prose.joined(separator: "\n")))
                prose = []
            }
        }
        while index < lines.count {
            if let fence = openFence {
                if closesFence(lines[index], fence) {
                    openFence = nil
                }
                prose.append(lines[index])
                index += 1
                continue
            }
            if let fence = openedFence(lines[index]) {
                openFence = fence
                prose.append(lines[index])
                index += 1
                continue
            }
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

    /// The fence a line OPENS: three or more backticks or tildes, the
    /// CommonMark fenced code block. Any info string after the marker is
    /// ignored.
    private static func openedFence(_ line: String) -> (character: Character, length: Int)? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard let first = trimmed.first, first == "`" || first == "~" else { return nil }
        var length = 0
        for character in trimmed {
            guard character == first else { break }
            length += 1
        }
        return length >= 3 ? (first, length) : nil
    }

    /// Whether a line CLOSES the given fence: the same character, at least
    /// as long a run, and nothing else on the line.
    private static func closesFence(
        _ line: String, _ fence: (character: Character, length: Int)
    ) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard let first = trimmed.first, first == fence.character else { return false }
        var length = 0
        for character in trimmed {
            guard character == fence.character else { break }
            length += 1
        }
        return length >= fence.length && length == trimmed.count
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
        let order = sortedRowIndices(byColumn: column, direction: direction)
        return MarkdownTableGrid(header: header, rows: order.map { rows[$0] })
    }

    /// The row permutation `sorted(byColumn:direction:)` applies, so a view
    /// can map a display position back to its original row POSITIONALLY.
    /// Matching rows back by value collides on duplicate rows.
    ///
    /// The numeric decision is made for the WHOLE column, not per pair:
    /// per-pair detection is not a strict weak ordering when a column mixes
    /// numbers with text ("9" < "10" numerically, but "10" < "1a" < "9" as
    /// strings is a cycle). Ties compare equal BOTH ways; the old `!order`
    /// descending arm claimed a < b and b < a at once, and `sorted(by:)`
    /// documents its behavior as unspecified under exactly that.
    func sortedRowIndices(byColumn column: Int, direction: SortDirection) -> [Int] {
        guard column >= 0, column < columnCount else { return Array(rows.indices) }
        let normalized = rows.map { ($0[safe: column] ?? "").replacingOccurrences(of: ",", with: "") }
        let isNumericColumn = normalized.allSatisfy { Double($0) != nil }
        return rows.indices.sorted { lhs, rhs in
            let comparison: ComparisonResult
            if isNumericColumn {
                let left = Double(normalized[lhs]) ?? 0
                let right = Double(normalized[rhs]) ?? 0
                comparison = left < right ? .orderedAscending
                    : (left > right ? .orderedDescending : .orderedSame)
            } else {
                comparison = normalized[lhs].localizedCaseInsensitiveCompare(normalized[rhs])
            }
            switch (comparison, direction) {
            case (.orderedSame, _): return false
            case (.orderedAscending, .ascending), (.orderedDescending, .descending): return true
            default: return false
            }
        }
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

    /// The original markdown again; the copy-the-source escape hatch. A
    /// literal pipe inside a cell is re-escaped, or the consumer (including
    /// this app's own `segments(in:)`) re-splits the cell and the pasted
    /// table comes out wider than the copied one.
    var markdown: String {
        func cell(_ value: String) -> String {
            value.replacingOccurrences(of: "|", with: "\\|")
        }
        var lines = ["| " + header.map(cell).joined(separator: " | ") + " |"]
        lines.append("| " + header.map { _ in "---" }.joined(separator: " | ") + " |")
        for row in rows {
            let padded = row + Array(repeating: "", count: max(0, columnCount - row.count))
            lines.append("| " + padded.prefix(columnCount).map(cell).joined(separator: " | ") + " |")
        }
        return lines.joined(separator: "\n")
    }

    private func delimited(
        _ separator: String, escapeTabNewline: Bool, rows rowsToWrite: [[String]]
    ) -> String {
        func cell(_ value: String) -> String {
            // CSV quoting is structural only: spreadsheets remove it before
            // deciding whether a cell is a formula. Treat assistant-generated
            // headers and values as text before applying CSV/TSV escaping.
            let firstNonWhitespace = value.first { !$0.isWhitespace }
            let safeValue: String
            if let firstNonWhitespace, "=+-@".contains(firstNonWhitespace) {
                safeValue = "'" + value
            } else {
                safeValue = value
            }
            if escapeTabNewline {
                return safeValue
                    .replacingOccurrences(of: "\t", with: "\\t")
                    .replacingOccurrences(of: "\n", with: "\\n")
            }
            let needsQuotes = safeValue.contains(separator) || safeValue.contains("\"")
                || safeValue.contains("\n")
            guard needsQuotes else { return safeValue }
            return "\"" + safeValue.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        var lines = [header.map(cell).joined(separator: separator)]
        for row in rowsToWrite {
            let padded = row + Array(repeating: "", count: max(0, columnCount - row.count))
            lines.append(padded.prefix(columnCount).map(cell).joined(separator: separator))
        }
        return lines.joined(separator: "\n")
    }
}
