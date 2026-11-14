import AppKit
import SwiftUI

/// Word document on fixed white paper: headings, runs, lists, tables and
/// embedded rasters. Run SIZES are not honored on purpose; body text follows
/// the user's font setting and headings take a themed step, so the pane stays
/// readable at every text size. Bold, italic, underline, strike and color are
/// honored.
struct DocxView: View {
    let document: DocxDocument

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(document.blocks.enumerated()), id: \.offset) { index, block in
                        blockView(block).id(index)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(20)
            }
            .background(OfficePaper.background)
            .foregroundStyle(OfficePaper.ink)
            .overlay(alignment: .topTrailing) {
                if !document.outline.isEmpty {
                    outlineMenu(proxy)
                        .padding(8)
                }
            }
        }
    }

    private func outlineMenu(_ proxy: ScrollViewProxy) -> some View {
        Menu {
            // A document can hold thousands of headings; the menu shows a
            // bounded prefix so it opens at once.
            ForEach(Array(document.outline.prefix(200).enumerated()), id: \.offset) { _, entry in
                Button {
                    proxy.scrollTo(entry.blockIndex, anchor: .top)
                } label: {
                    Text(verbatim: String(repeating: "  ", count: max(0, entry.level - 1)) + entry.title)
                }
            }
        } label: {
            Label { Text("Outline", bundle: .module) } icon: { Image(systemName: "list.bullet.indent") }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help(Text("Outline", bundle: .module))
    }

    @ViewBuilder
    private func blockView(_ block: DocxBlock) -> some View {
        switch block {
        case .paragraph(let paragraph):
            paragraphView(paragraph)
        case .table(let table):
            tableView(table)
        }
    }

    private func paragraphView(_ paragraph: DocxParagraph) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            if !paragraph.runs.isEmpty || paragraph.marker != nil {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    if let marker = paragraph.marker {
                        Text(verbatim: marker)
                    }
                    Text(attributed(paragraph))
                        .textSelection(.enabled)
                        .multilineTextAlignment(textAlignment(paragraph.alignment))
                        .frame(maxWidth: .infinity, alignment: frameAlignment(paragraph.alignment))
                }
                .padding(.leading, CGFloat(paragraph.listLevel) * 18)
                .themedFont(headingStep(paragraph.headingLevel), weight: paragraph.headingLevel == nil ? nil : .semibold)
            }
            ForEach(Array(paragraph.imageIndices.enumerated()), id: \.offset) { _, index in
                if document.images.indices.contains(index) {
                    DocxImageView(data: document.images[index])
                }
            }
        }
    }

    private func attributed(_ paragraph: DocxParagraph) -> AttributedString {
        var result = AttributedString()
        for run in paragraph.runs {
            var piece = AttributedString(run.text)
            var intent: InlinePresentationIntent = []
            if run.bold { intent.insert(.stronglyEmphasized) }
            if run.italic { intent.insert(.emphasized) }
            if !intent.isEmpty { piece.inlinePresentationIntent = intent }
            if run.underline { piece.underlineStyle = .single }
            if run.strike { piece.strikethroughStyle = .single }
            if let color = OfficePaper.color(hex: run.colorHex) { piece.foregroundColor = color }
            result += piece
        }
        return result
    }

    private func headingStep(_ level: Int?) -> AppFontStep {
        switch level {
        case 1: return .title2
        case 2: return .title3
        case 3: return .large
        default: return .base
        }
    }

    private func textAlignment(_ value: DocxAlignment) -> TextAlignment {
        switch value {
        case .center: return .center
        case .trailing: return .trailing
        default: return .leading
        }
    }

    private func frameAlignment(_ value: DocxAlignment) -> Alignment {
        switch value {
        case .center: return .center
        case .trailing: return .trailing
        default: return .leading
        }
    }

    private func tableView(_ table: DocxTable) -> some View {
        Grid(alignment: .topLeading, horizontalSpacing: 0, verticalSpacing: 0) {
            ForEach(Array(table.rows.enumerated()), id: \.offset) { _, row in
                GridRow {
                    ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                        VStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(cell.paragraphs.enumerated()), id: \.offset) { _, paragraph in
                                Text(attributed(paragraph))
                                    .textSelection(.enabled)
                            }
                        }
                        .themedFont(.callout)
                        .padding(6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(OfficePaper.color(hex: cell.shadingHex) ?? .clear)
                        .overlay(Rectangle().stroke(OfficePaper.gridLine, lineWidth: 1))
                    }
                }
            }
        }
    }
}

/// An embedded raster, decoded off the main thread through the bounded
/// decoder in `OfficePaper`. A refusal shows nothing rather than an error.
private struct DocxImageView: View {
    let data: Data
    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 520, maxHeight: 400, alignment: .leading)
                    .accessibilityHidden(true)
            }
        }
        .task {
            let bytes = data
            image = await Task.detached(priority: .utility) { OfficePaper.decodeImage(bytes) }.value
        }
    }
}
