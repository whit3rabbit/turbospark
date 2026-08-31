import SwiftUI

/// Left-to-right layout that wraps onto a new line when the next subview does
/// not fit.
///
/// An `HStack` compresses its children when the row overflows, and a capsule
/// pill has no floor on how narrow it will go: the composer footer squeezed
/// "Chat" and "Projects" down to one character per line rather than wrapping.
/// This places every subview at its OWN ideal size and moves to the next line
/// instead, so a control is either on screen at full width or on the row below.
///
/// `sizeThatFits` reports the widest line actually used rather than the whole
/// proposal, so this can sit beside a `Spacer` without claiming the row.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    var lineSpacing: CGFloat = 8
    var alignment: HorizontalAlignment = .leading

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let lines = lineBreaks(maxWidth: proposal.width ?? .infinity, subviews: subviews)
        let width = lines.map(\.width).max() ?? 0
        let height = lines.reduce(0) { $0 + $1.height } +
            lineSpacing * CGFloat(max(lines.count - 1, 0))
        return CGSize(width: width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let lines = lineBreaks(maxWidth: bounds.width, subviews: subviews)
        var y = bounds.minY

        for line in lines {
            var x: CGFloat
            switch alignment {
            case .center:
                x = bounds.minX + (bounds.width - line.width) / 2
            case .trailing:
                x = bounds.maxX - line.width
            default:
                x = bounds.minX
            }

            for index in line.range {
                let size = subviews[index].sizeThatFits(.unspecified)
                // Centre each control on its line so pills of different
                // heights (a 28pt icon button beside a 20pt capsule) sit on a
                // shared baseline instead of hanging off the top.
                subviews[index].place(
                    at: CGPoint(x: x, y: y + (line.height - size.height) / 2),
                    proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += line.height + lineSpacing
        }
    }

    private struct Line {
        var range: Range<Int>
        var width: CGFloat
        var height: CGFloat
    }

    private func lineBreaks(maxWidth: CGFloat, subviews: Subviews) -> [Line] {
        var lines: [Line] = []
        var start = 0
        var width: CGFloat = 0
        var height: CGFloat = 0

        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let advance = width == 0 ? size.width : width + spacing + size.width
            if advance > maxWidth, index > start {
                lines.append(Line(range: start..<index, width: width, height: height))
                start = index
                width = size.width
                height = size.height
                continue
            }
            width = advance
            height = max(height, size.height)
        }

        if start < subviews.endIndex {
            lines.append(Line(range: start..<subviews.endIndex, width: width, height: height))
        }
        return lines
    }
}
