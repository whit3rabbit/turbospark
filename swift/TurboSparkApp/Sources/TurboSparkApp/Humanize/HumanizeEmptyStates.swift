import SwiftUI

/// A glyph on a soft accent tile: the theme's color at rest, in place of a
/// bare gray symbol.
struct HumanizeGlyphTile: View {
    let systemImage: String
    let size: CGFloat
    @Environment(\.appTheme) private var theme

    var body: some View {
        let side = (size * HumanizeMetrics.textScale(theme)).rounded()
        Image(systemName: systemImage)
            .font(theme.ui(.callout, weight: .semibold))
            .foregroundStyle(.appAccentText)
            .frame(width: side, height: side)
            .background(theme.accent.opacity(0.14),
                        in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius + 2, style: .continuous))
            .accessibilityHidden(true)
    }
}

/// The empty Rewrite card. It teaches the three steps in the labels the
/// buttons already carry, so it adds no sentence to translate.
struct HumanizeRewriteEmptyState: View {
    @Environment(\.appTheme) private var theme

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "text.alignleft")
                .font(theme.ui(.display, weight: .light))
                .foregroundStyle(theme.accent.gradient)
                .padding(18)
                .background(Circle().fill(theme.accent.opacity(0.10)))
                .overlay(Circle().stroke(theme.accent.opacity(0.28), lineWidth: 1))
                .accessibilityHidden(true)
            Text("Your rewrite will appear here.", bundle: .module)
                .font(theme.ui(.title3, weight: .semibold))
                .multilineTextAlignment(.center)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) { steps }
                VStack(spacing: 8) { steps }
            }
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private var steps: some View {
        pill("1", Text("Paste", bundle: .module))
        arrow
        pill("2", Text("Rewrite draft", bundle: .module))
        arrow
        pill("3", Text("Show changes", bundle: .module))
    }

    private var arrow: some View {
        Image(systemName: "arrow.forward").font(theme.ui(.micro, weight: .bold))
            .foregroundStyle(.appSecondary).accessibilityHidden(true)
    }

    private func pill(_ number: String, _ label: Text) -> some View {
        HStack(spacing: 6) {
            Text(verbatim: number).font(theme.ui(.micro, weight: .bold)).monospacedDigit()
                .foregroundStyle(.appAccentText)
            label.font(theme.ui(.small, weight: .medium)).foregroundStyle(.appText).lineLimit(1)
        }
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(theme.accent.opacity(0.10), in: Capsule())
        .overlay(Capsule().stroke(theme.accent.opacity(0.25), lineWidth: 1))
        .fixedSize()
    }
}
