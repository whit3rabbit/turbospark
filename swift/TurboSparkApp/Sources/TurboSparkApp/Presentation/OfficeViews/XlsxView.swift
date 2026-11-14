import SwiftUI

/// Workbook viewer: sheet tabs along the bottom, a scrollable grid with a
/// pinned column-letter header and pinned frozen rows. Values are the cached
/// ones the file stores, formatted through `OfficeNumberFormat`.
struct XlsxView: View {
    let document: XlsxDocument
    @State private var selected = 0

    private static let defaultColumnWidth = 64.0 + 8
    private static let rowHeight = 22.0
    private static let gutterWidth = 40.0

    var body: some View {
        VStack(spacing: 0) {
            if document.sheets.indices.contains(selected) {
                grid(document.sheets[selected])
                    .id(selected)
            } else {
                Color.clear.frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if document.sheets.count > 1 {
                Divider()
                tabs
            }
        }
    }

    private var tabs: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                ForEach(Array(document.sheets.enumerated()), id: \.offset) { index, sheet in
                    Button {
                        selected = index
                    } label: {
                        Text(verbatim: sheet.name)
                            .lineLimit(1)
                    }
                    .appAction(index == selected ? .secondary : .quiet, size: .small)
                    .accessibilityAddTraits(index == selected ? .isSelected : [])
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
        }
        .accessibilityLabel(Text("Sheets", bundle: .module))
    }

    private func width(of column: Int, in sheet: XlsxSheet) -> CGFloat {
        guard let characters = sheet.columnWidths[column] else { return Self.defaultColumnWidth }
        return CGFloat(characters * 7 + 8)
    }

    private func grid(_ sheet: XlsxSheet) -> some View {
        let pinned = sheet.rows.prefix { $0.number <= sheet.frozenRows }
        let scrolling = sheet.rows.dropFirst(pinned.count)
        return ScrollView([.horizontal, .vertical]) {
            LazyVStack(alignment: .leading, spacing: 0, pinnedViews: [.sectionHeaders]) {
                Section {
                    ForEach(Array(scrolling), id: \.number) { row in
                        rowView(row, sheet: sheet)
                    }
                } header: {
                    VStack(alignment: .leading, spacing: 0) {
                        letterHeader(sheet)
                        ForEach(Array(pinned), id: \.number) { row in
                            rowView(row, sheet: sheet)
                        }
                    }
                    .background(OfficePaper.background)
                }
            }
        }
        .background(OfficePaper.background)
        .foregroundStyle(OfficePaper.ink)
    }

    private func letterHeader(_ sheet: XlsxSheet) -> some View {
        HStack(spacing: 0) {
            Color.clear.frame(width: Self.gutterWidth, height: Self.rowHeight)
                .background(OfficePaper.headerFill)
                .overlay(Rectangle().stroke(OfficePaper.gridLine, lineWidth: 0.5))
            ForEach(0..<max(sheet.columnCount, 1), id: \.self) { column in
                Text(verbatim: Self.columnName(column))
                    .themedFont(.tiny, weight: .medium)
                    .frame(width: width(of: column, in: sheet), height: Self.rowHeight)
                    .background(OfficePaper.headerFill)
                    .overlay(Rectangle().stroke(OfficePaper.gridLine, lineWidth: 0.5))
            }
        }
    }

    private func rowView(_ row: XlsxRow, sheet: XlsxSheet) -> some View {
        // Two cells naming the same reference would trap a plain dictionary
        // initializer; the later one wins.
        let byColumn = Dictionary(row.cells.map { ($0.column, $0) }, uniquingKeysWith: { _, later in later })
        return HStack(spacing: 0) {
            Text(verbatim: String(row.number))
                .themedFont(.tiny)
                .frame(width: Self.gutterWidth, height: Self.rowHeight)
                .background(OfficePaper.headerFill)
                .overlay(Rectangle().stroke(OfficePaper.gridLine, lineWidth: 0.5))
            ForEach(0..<max(sheet.columnCount, 1), id: \.self) { column in
                cellView(byColumn[column], width: width(of: column, in: sheet))
            }
        }
    }

    @ViewBuilder
    private func cellView(_ cell: XlsxCell?, width: CGFloat) -> some View {
        let alignment = cell.map(Self.frameAlignment) ?? .leading
        Text(verbatim: cell?.text ?? "")
            .themedFont(.callout, weight: (cell?.bold ?? false) ? .semibold : nil)
            .italic(cell?.italic ?? false)
            .foregroundStyle(OfficePaper.color(hex: cell?.colorHex) ?? OfficePaper.ink)
            .lineLimit(1)
            .padding(.horizontal, 4)
            .frame(width: width, height: Self.rowHeight, alignment: alignment)
            .background(OfficePaper.color(hex: cell?.fillHex) ?? .clear)
            .overlay(Rectangle().stroke(OfficePaper.gridLine, lineWidth: 0.5))
            .overlay(alignment: .topTrailing) {
                if let comment = cell?.comment {
                    // A note marker; the note text is the tooltip.
                    Triangle()
                        .fill(OfficePaper.noteMarker)
                        .frame(width: 6, height: 6)
                        .accessibilityLabel(Text(verbatim: comment))
                }
            }
            .help(Text(verbatim: cell?.comment ?? ""))
    }

    private static func frameAlignment(_ cell: XlsxCell) -> Alignment {
        switch cell.alignment {
        case .center: return .center
        case .trailing: return .trailing
        case .leading, .justified: return .leading
        case nil: return cell.isNumeric ? .trailing : .leading
        }
    }

    /// 0 to "A", 25 to "Z", 26 to "AA".
    static func columnName(_ index: Int) -> String {
        var number = index
        var name = ""
        repeat {
            name = String(UnicodeScalar(65 + number % 26)!) + name
            number = number / 26 - 1
        } while number >= 0
        return name
    }
}

private struct Triangle: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}
