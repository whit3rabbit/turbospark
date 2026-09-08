import AppKit
import SwiftUI

/// Assistant-message markdown that renders each pipe table as an
/// interactive grid when the message carries any. Prose segments go to the
/// ordinary markdown renderer, so a table is an upgrade of ONE segment, not
/// a second code path for the whole message (the qwen-code
/// `EnhancedMarkdownTable` subset: sorting, row selection, copy, export).
///
/// The parse runs once per body text; a message without interactive-grade
/// tables takes the plain markdown path untouched.
struct MarkdownContentWithTablesView: View {
    let text: String
    var onPreviewHTML: ((String) -> Void)?

    private var segments: [MarkdownTableGrid.Segment] {
        MarkdownTableGrid.segments(in: text)
    }

    private func isInteractive(_ segment: MarkdownTableGrid.Segment) -> Bool {
        if case .table(let grid) = segment { return grid.isInteractiveCandidate }
        return false
    }

    var body: some View {
        let segments = segments
        if segments.contains(where: isInteractive) {
            VStack(alignment: .leading, spacing: 14) {
                ForEach(Array(segments.enumerated()), id: \.offset) { _, segment in
                    switch segment {
                    case .markdown(let prose):
                        if !prose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            ChatMessageMarkdownView(prose, onPreviewHTML: onPreviewHTML)
                        }
                    case .table(let grid):
                        if grid.isInteractiveCandidate {
                            InteractiveMarkdownTableView(grid: grid)
                        } else {
                            ChatMessageMarkdownView(grid.markdown, onPreviewHTML: onPreviewHTML)
                        }
                    }
                }
            }
        } else {
            ChatMessageMarkdownView(text, onPreviewHTML: onPreviewHTML)
        }
    }
}

/// One interactive table: sortable header, click-to-select rows, copy as
/// TSV/CSV/markdown, and CSV export. Sort and selection are view state on
/// purpose -- a table row has no identity the message model knows about.
struct InteractiveMarkdownTableView: View {
    @Environment(\.appTheme) private var theme
    let grid: MarkdownTableGrid

    @State private var sortedColumn: Int?
    @State private var sortDirection: MarkdownTableGrid.SortDirection = .ascending
    @State private var selectedRowIndices: Set<Int> = []
    @State private var isCopied = false
    @State private var copyResetTask: Task<Void, Never>? = nil

    private var displayGrid: MarkdownTableGrid {
        if let sortedColumn {
            return grid.sorted(byColumn: sortedColumn, direction: sortDirection)
        }
        return grid
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            toolbar
            // The grid: a header row of sort buttons over the data rows.
            // LazyVStack is unnecessary by design -- a table worth
            // interacting with is a table small enough to lay out whole.
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 0) {
                    ForEach(Array(grid.header.enumerated()), id: \.offset) { column, cell in
                        headerCell(column: column, cell: cell)
                    }
                }
                Divider()
                ForEach(Array(displayGrid.rows.enumerated()), id: \.offset) { displayIndex, row in
                    // Selection tracks the ORIGINAL row index, so sorting
                    // does not silently reshuffle what is selected.
                    let originalIndex = grid.rows.firstIndex(of: row)
                    rowView(row: row, isSelected: originalIndex.map(selectedRowIndices.contains) ?? false)
                        .contentShape(Rectangle())
                        .onTapGesture { toggleSelection(originalIndex) }
                    if displayIndex < displayGrid.rows.count - 1 {
                        Divider().opacity(0.5)
                    }
                }
            }
            .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 0.5)
            )
        }
        .markdownMargin(top: 8, bottom: 12)
    }

    private var toolbar: some View {
        HStack(spacing: 8) {
            Text("Table", bundle: .module)
                .themedCode(.tiny, weight: .semibold)
                .foregroundStyle(.tertiary)
            Text("\(grid.rowCount) rows x \(grid.columnCount) columns", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.tertiary)
            Spacer()
            if !selectedRowIndices.isEmpty {
                Text(
                    selectedRowIndices.count == 1
                        ? "1 row selected"
                        : "\(selectedRowIndices.count) rows selected",
                    bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
                Button("Clear") { selectedRowIndices.removeAll() }
                    .buttonStyle(.plain)
                    .themedFont(.tiny)
            }
            Menu {
                Button("Copy selection as TSV") {
                    copyTSV(selectedOnly: true)
                }
                .disabled(selectedRowIndices.isEmpty)
                Button("Copy table as TSV") { copyTSV(selectedOnly: false) }
                Button("Copy table as CSV") { copyCSV() }
                Button("Copy table as Markdown") { copyMarkdown() }
                Divider()
                Button("Export as CSV...") { exportCSV() }
            } label: {
                Label(isCopied ? "Copied" : "Copy", systemImage: isCopied ? "checkmark" : "doc.on.doc")
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(isCopied ? Color.accentColor : Color.secondary)
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .accessibilityLabel("Copy or export this table")
        }
        .padding(.bottom, 4)
    }

    private func headerCell(column: Int, cell: String) -> some View {
        Button {
            if sortedColumn == column {
                sortDirection = sortDirection.toggled
            } else {
                sortedColumn = column
                sortDirection = .ascending
            }
        } label: {
            HStack(spacing: 3) {
                Text(cell)
                    .themedFont(.small, weight: .semibold)
                    .lineLimit(2)
                Image(
                    systemName: sortedColumn == column
                        ? (sortDirection == .ascending ? "chevron.up" : "chevron.down")
                        : "arrow.up.arrow.down")
                    .themedFont(points: 7, weight: .bold)
                    .foregroundStyle(
                        sortedColumn == column ? TurboSparkTheme.accentColor : Color.secondary.opacity(0.5))
                    .accessibilityHidden(true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Sort by \(cell)")
        .accessibilityLabel("Sort table by column \(cell)")
    }

    private func rowView(row: [String], isSelected: Bool) -> some View {
        HStack(spacing: 0) {
            ForEach(Array(row.prefix(grid.columnCount).enumerated()), id: \.offset) { _, cell in
                Text(cell)
                    .themedFont(.small)
                    .lineLimit(3)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
            }
        }
        .background(isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear)
    }

    private func toggleSelection(_ originalIndex: Int?) {
        guard let originalIndex else { return }
        if selectedRowIndices.contains(originalIndex) {
            selectedRowIndices.remove(originalIndex)
        } else {
            selectedRowIndices.insert(originalIndex)
        }
    }

    private func flashCopied() {
        copyResetTask?.cancel()
        isCopied = true
        copyResetTask = Task {
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            isCopied = false
        }
    }

    private func copyTSV(selectedOnly: Bool) {
        let payload = selectedOnly
            ? gridSelectionTSV()
            : displayGrid.tsv()
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(payload, forType: .string)
        flashCopied()
    }

    private func copyCSV() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(displayGrid.csv(), forType: .string)
        flashCopied()
    }

    private func copyMarkdown() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(displayGrid.markdown, forType: .string)
        flashCopied()
    }

    /// TSV of just the selected rows (header included), what the
    /// "selection" copy arm exists for.
    private func gridSelectionTSV() -> String {
        let selected = displayGrid.rows.enumerated()
            .filter { _, row in
                grid.rows.firstIndex(of: row).map(selectedRowIndices.contains) == true
            }
            .map(\.element)
        return MarkdownTableGrid(header: grid.header, rows: selected).tsv()
    }

    private func exportCSV() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.nameFieldStringValue = "table.csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? displayGrid.csv().write(to: url, atomically: true, encoding: .utf8)
        flashCopied()
    }
}
