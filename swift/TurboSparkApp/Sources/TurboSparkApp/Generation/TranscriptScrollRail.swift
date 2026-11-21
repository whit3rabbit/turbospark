import Foundation
import SwiftUI

/// Pure rules for the transcript's turn rail: one tick per prompt the human
/// wrote, placed by turn index (not pixel offset, which the lazy transcript
/// does not know without measuring every row).
///
/// Kept free of views so every decision the rail draws is unit-tested.
enum TranscriptScrollRail {
    /// Below this a chat fits on a screen or two and the rail is noise.
    static let minimumTurns = 3
    /// Preview cap in characters. The card is one line, so this only bounds
    /// work on a huge pasted prompt; the text itself truncates at the card.
    static let previewLimit = 80
    /// Hit width of the rail, wider than the ticks so the pointer finds it.
    static let railWidth: CGFloat = 16
    /// Space between the rail and the window edge: macOS draws the overlay
    /// scroller there, and the rail must not sit under it.
    static let edgeInset: CGFloat = 14
    /// Tallest the rail grows, so ticks stay near each other in a tall window.
    static let maxHeight: CGFloat = 320

    struct Item: Equatable, Identifiable {
        /// The prompt row's message id, which the transcript scrolls to.
        let id: UUID
        let index: Int
        /// First line of the prompt, nil for a prompt with no text (an image).
        let preview: String?
    }

    /// One item per human prompt, in transcript order, or none when the chat
    /// has fewer than `minimumTurns`. Synthetic user rows (hook reasons, goal
    /// check-ins) are the app talking, not turns the reader took.
    static func items(in messages: [AppChatMessage]) -> [Item] {
        let prompts = messages.filter { $0.role == .user && !$0.isSynthetic }
        guard prompts.count >= minimumTurns else { return [] }
        return prompts.enumerated().map { offset, message in
            Item(id: message.id, index: offset, preview: preview(of: message.content))
        }
    }

    /// The first non-blank line, trimmed, cut at `limit` characters with an
    /// ellipsis. Nil when the content has no text at all.
    static func preview(of content: String, limit: Int = previewLimit) -> String? {
        var first: String?
        // enumerateLines stops at the first hit, so a long paste is not split whole.
        content.enumerateLines { line, stop in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                first = trimmed
                stop = true
            }
        }
        guard let line = first else { return nil }
        guard line.count > limit else { return line }
        let cut = String(line.prefix(limit)).trimmingCharacters(in: .whitespaces)
        return cut + "\u{2026}"
    }

    /// Where a tick sits along the rail, 0 at the top and 1 at the bottom.
    /// A lone tick sits in the middle. Nil for an index outside the rail.
    static func tickFraction(index: Int, count: Int) -> CGFloat? {
        guard count > 0, (0..<count).contains(index) else { return nil }
        guard count > 1 else { return 0.5 }
        return CGFloat(index) / CGFloat(count - 1)
    }

    /// The tick nearest a pointer at `fraction` of the rail's height, clamped
    /// to the end ticks. Nil when there are no ticks or the input is not a number.
    static func index(atFraction fraction: CGFloat, count: Int) -> Int? {
        guard count > 0, fraction.isFinite else { return nil }
        let clamped = min(max(fraction, 0), 1)
        return Int((clamped * CGFloat(count - 1)).rounded())
    }

    /// Whether the pane leaves a gutter beside the centered reading column
    /// wide enough for the rail, so it never covers transcript text.
    static func fitsBeside(columnWidth: CGFloat, containerWidth: CGFloat) -> Bool {
        let gutter = (containerWidth - columnWidth) / 2
        return gutter >= railWidth + edgeInset * 2
    }

    /// The tick to mark as current: the last turn the reader jumped to, by
    /// the rail or the Previous/Next Turn commands. The transcript does not
    /// measure which row is on screen, so nil means "not known".
    static func currentIndex(of targetID: UUID?, in items: [Item]) -> Int? {
        guard let targetID else { return nil }
        return items.firstIndex { $0.id == targetID }
    }

    /// VoiceOver's adjustable step. Clamps at the ends; with no current turn
    /// it starts from the end the swipe points at, as `jumpTurn` does.
    static func adjusted(current: Int?, step: Int, count: Int) -> Int? {
        guard count > 0 else { return nil }
        guard let current else { return step < 0 ? count - 1 : 0 }
        return min(max(current + step, 0), count - 1)
    }
}

/// The rail itself. It draws ticks and a hover preview and reports a jump;
/// the transcript owns the scroll (and follow-mode) in `onJump`.
struct TranscriptScrollRailView: View {
    let items: [TranscriptScrollRail.Item]
    let currentIndex: Int?
    /// `animated` is false when the window is not key, so a background
    /// window does not run a scroll animation nobody is watching.
    let onJump: (UUID, _ animated: Bool) -> Void

    @Environment(\.appTheme) private var theme
    @Environment(\.controlActiveState) private var controlActiveState
    @State private var hoveredIndex: Int?

    /// Inset at both ends so the first and last ticks are not clipped.
    private let endPadding: CGFloat = 4

    var body: some View {
        GeometryReader { proxy in
            let height = proxy.size.height
            ZStack(alignment: .topTrailing) {
                ForEach(items) { item in
                    tick(for: item.index)
                        .position(x: TranscriptScrollRail.railWidth / 2, y: y(for: item.index, height: height))
                }
            }
            .frame(width: TranscriptScrollRail.railWidth, height: height)
            .contentShape(Rectangle())
            .onContinuousHover { phase in
                switch phase {
                case .active(let location):
                    let index = index(at: location.y, height: height)
                    if index != hoveredIndex { hoveredIndex = index }
                case .ended:
                    hoveredIndex = nil
                }
            }
            .onTapGesture { location in
                guard let index = index(at: location.y, height: height) else { return }
                jump(to: index)
            }
            .overlay(alignment: .topTrailing) {
                if let hoveredIndex, items.indices.contains(hoveredIndex) {
                    previewCard(for: items[hoveredIndex])
                        .alignmentGuide(.trailing) { d in d.width + TranscriptScrollRail.railWidth + 6 }
                        .alignmentGuide(.top) { d in d.height / 2 - y(for: hoveredIndex, height: height) }
                        .allowsHitTesting(false)
                }
            }
        }
        .frame(width: TranscriptScrollRail.railWidth)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Turn navigation", bundle: .module))
        .accessibilityValue(currentValue)
        .accessibilityAdjustableAction { direction in
            let step: Int
            switch direction {
            case .increment: step = 1
            case .decrement: step = -1
            @unknown default: return
            }
            if let next = TranscriptScrollRail.adjusted(current: currentIndex, step: step, count: items.count) {
                jump(to: next)
            }
        }
        .accessibilityActions {
            ForEach(items) { item in
                Button {
                    jump(to: item.index)
                } label: {
                    if let preview = item.preview {
                        Text("Jump to turn \(item.index + 1): \(preview)", bundle: .module)
                    } else {
                        Text("Jump to turn \(item.index + 1)", bundle: .module)
                    }
                }
            }
        }
    }

    private var currentValue: Text {
        guard let currentIndex else { return Text(verbatim: "") }
        return Text("Turn \(currentIndex + 1) of \(items.count)", bundle: .module)
    }

    private func jump(to index: Int) {
        guard items.indices.contains(index) else { return }
        onJump(items[index].id, controlActiveState == .key)
    }

    private func y(for index: Int, height: CGFloat) -> CGFloat {
        let span = max(0, height - endPadding * 2)
        let fraction = TranscriptScrollRail.tickFraction(index: index, count: items.count) ?? 0
        return endPadding + fraction * span
    }

    private func index(at y: CGFloat, height: CGFloat) -> Int? {
        let span = height - endPadding * 2
        guard span > 0 else { return nil }
        return TranscriptScrollRail.index(atFraction: (y - endPadding) / span, count: items.count)
    }

    /// Current is the longest tick and hovered the next, so the state reads
    /// by length as well as color. Hover changes are not animated: the
    /// pointer sweeps many ticks a second and each would start an animation.
    @ViewBuilder
    private func tick(for index: Int) -> some View {
        let isCurrent = index == currentIndex
        let isHovered = index == hoveredIndex
        Capsule()
            .fill(isCurrent ? AnyShapeStyle(.appAccentText)
                : isHovered ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(.appBorder))
            .frame(width: isCurrent ? 14 : isHovered ? 11 : 8, height: 2)
            .frame(width: TranscriptScrollRail.railWidth, alignment: .trailing)
    }

    private func previewCard(for item: TranscriptScrollRail.Item) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Turn \(item.index + 1) of \(items.count)", bundle: .module)
                .themedFont(.micro, weight: .medium)
                .foregroundStyle(.appSecondary)
            if let preview = item.preview {
                Text(verbatim: preview)
                    .themedFont(.small)
                    .foregroundStyle(.appText)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
        }
        .frame(maxWidth: (280 * AppMetrics.textScale(theme)).rounded(), alignment: .leading)
        .fixedSize()
        .appCardSurface(padding: 8)
    }
}
