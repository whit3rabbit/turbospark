import AppKit
import SwiftUI

/// Slide viewer: one slide at a time (so only the visible slide is laid out),
/// scaled to fit, on fixed white paper. Geometry is the slide's own EMU
/// frames; layouts and masters are not read (see `OfficeSlidesModel`).
struct PptxView: View {
    let document: PptxDocument
    @State private var index = 0

    var body: some View {
        VStack(spacing: 0) {
            GeometryReader { proxy in
                let scale = min(
                    (proxy.size.width - 24) / document.width,
                    (proxy.size.height - 24) / document.height)
                if scale > 0, document.slides.indices.contains(index) {
                    slideCanvas(document.slides[index], scale: scale)
                        .frame(width: document.width * scale, height: document.height * scale)
                        .frame(width: proxy.size.width, height: proxy.size.height)
                }
            }
            Divider()
            navigator
        }
    }

    private var navigator: some View {
        HStack(spacing: 8) {
            AppIconButton("chevron.left", label: Text("Previous slide", bundle: .module)) {
                index = max(0, index - 1)
            }
            .disabled(index == 0)
            Text("Slide \(index + 1) of \(document.slides.count)", bundle: .module)
                .themedFont(.small)
                .monospacedDigit()
            AppIconButton("chevron.right", label: Text("Next slide", bundle: .module)) {
                index = min(document.slides.count - 1, index + 1)
            }
            .disabled(index >= document.slides.count - 1)
            Spacer(minLength: 0)
        }
        .appAction(.quiet, size: .small)
        .foregroundStyle(.appSecondary)
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
    }

    /// `scale` is points per EMU.
    private func slideCanvas(_ slide: Slide, scale: CGFloat) -> some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(OfficePaper.background)
                .shadow(radius: 2)
            ForEach(Array(slide.shapes.enumerated()), id: \.offset) { _, shape in
                shapeView(shape, scale: scale)
                    .frame(width: max(shape.width * scale, 1), height: max(shape.height * scale, 1), alignment: .topLeading)
                    .offset(x: shape.x * scale, y: shape.y * scale)
            }
        }
        .clipped()
        .foregroundStyle(OfficePaper.ink)
    }

    @ViewBuilder
    private func shapeView(_ shape: SlideShape, scale: CGFloat) -> some View {
        switch shape.kind {
        case .text:
            textBlock(shape, scale: scale)
        case .rectangle:
            ZStack(alignment: .topLeading) {
                Rectangle().fill(OfficePaper.color(hex: shape.fillHex) ?? .clear)
                Rectangle().stroke(OfficePaper.color(hex: shape.lineHex) ?? .clear, lineWidth: 1)
                textBlock(shape, scale: scale)
            }
        case .ellipse:
            ZStack {
                Ellipse().fill(OfficePaper.color(hex: shape.fillHex) ?? .clear)
                Ellipse().stroke(OfficePaper.color(hex: shape.lineHex) ?? .clear, lineWidth: 1)
                textBlock(shape, scale: scale)
            }
        case .line:
            Path { path in
                path.move(to: .zero)
                path.addLine(to: CGPoint(x: shape.width * scale, y: shape.height * scale))
            }
            .stroke(OfficePaper.color(hex: shape.lineHex) ?? OfficePaper.ink, lineWidth: 1)
        case .picture:
            if let data = shape.imageData, let image = OfficePaper.decodeImage(data) {
                Image(nsImage: image).resizable().scaledToFit()
                    .accessibilityHidden(true)
            } else {
                placeholder(Text("Picture", bundle: .module), symbol: "photo", scale: scale)
            }
        case .chart:
            placeholder(Text("Chart not rendered", bundle: .module), symbol: "chart.bar", scale: scale)
        case .table:
            tableBlock(shape, scale: scale)
        }
    }

    private func textBlock(_ shape: SlideShape, scale: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 2 * scale * 12_700) {
            ForEach(Array(shape.paragraphs.enumerated()), id: \.offset) { _, paragraph in
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    if paragraph.bulleted { Text(verbatim: "\u{2022}") }
                    Text(attributed(paragraph))
                        .multilineTextAlignment(alignment(paragraph.alignment))
                        .frame(maxWidth: .infinity, alignment: frame(paragraph.alignment))
                }
                .themedFont(fitting: points(paragraph, scale: scale))
            }
        }
        .padding(4 * scale * 12_700)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .clipped()
    }

    /// Font points for a paragraph: its largest run (18 pt when unstated)
    /// scaled by points per slide point. Floored so text stays a shape.
    private func points(_ paragraph: SlideParagraph, scale: CGFloat) -> CGFloat {
        let source = paragraph.runs.compactMap(\.size).max() ?? 18
        return max(4, CGFloat(source) * scale * 12_700)
    }

    private func attributed(_ paragraph: SlideParagraph) -> AttributedString {
        var result = AttributedString()
        for run in paragraph.runs {
            var piece = AttributedString(run.text)
            var intent: InlinePresentationIntent = []
            if run.bold { intent.insert(.stronglyEmphasized) }
            if run.italic { intent.insert(.emphasized) }
            if !intent.isEmpty { piece.inlinePresentationIntent = intent }
            if run.underline { piece.underlineStyle = .single }
            if let color = OfficePaper.color(hex: run.colorHex) { piece.foregroundColor = color }
            result += piece
        }
        return result
    }

    private func alignment(_ value: DocxAlignment) -> TextAlignment {
        switch value {
        case .center: return .center
        case .trailing: return .trailing
        default: return .leading
        }
    }

    private func frame(_ value: DocxAlignment) -> Alignment {
        switch value {
        case .center: return .center
        case .trailing: return .trailing
        default: return .leading
        }
    }

    private func placeholder(_ label: Text, symbol: String, scale: CGFloat) -> some View {
        ZStack {
            Rectangle().fill(OfficePaper.placeholderFill)
            VStack(spacing: 2) {
                Image(systemName: symbol).accessibilityHidden(true)
                label
            }
            .themedFont(.tiny)
        }
    }

    private func tableBlock(_ shape: SlideShape, scale: CGFloat) -> some View {
        Grid(alignment: .topLeading, horizontalSpacing: 0, verticalSpacing: 0) {
            ForEach(Array(shape.tableRows.enumerated()), id: \.offset) { _, row in
                GridRow {
                    ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                        Text(verbatim: cell)
                            .themedFont(fitting: max(4, 14 * scale * 12_700))
                            .padding(3)
                            .frame(maxWidth: .infinity, alignment: .topLeading)
                            .overlay(Rectangle().stroke(OfficePaper.gridLine, lineWidth: 0.5))
                    }
                }
            }
        }
    }
}
