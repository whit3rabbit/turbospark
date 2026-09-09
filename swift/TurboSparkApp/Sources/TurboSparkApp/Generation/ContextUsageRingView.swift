import SwiftUI
import TurboSpark

/// The composer's context ring: a small circular meter that starts empty and
/// fills clockwise as the window fills -- green while comfortable, yellow
/// from 70 percent, red from 90. Clicking it opens the context breakdown
/// popover. Claude Code's indicator, at the scale this app runs at.
///
/// The fraction is the SAME reading the status bar's context meter shows
/// (`estimatedContextTokens` over `resolvedContextTokens`), so the two
/// surfaces cannot disagree. Hidden entirely until an engine session is
/// loaded: without one there is no window to be full.
struct ContextUsageRingView: View {
    @ObservedObject var model: AppModel
    @State private var showsPopover = false
    @State private var isHovered = false

    private var fraction: Double {
        guard model.session != nil else { return 0 }
        let limit = model.resolvedContextTokens
        guard limit > 0 else { return 0 }
        return min(1, Double(model.estimatedContextTokens) / Double(limit))
    }

    private var tint: Color {
        ContextUsageSummary.tint(forFraction: fraction)
    }

    private var percentText: String {
        "\(Int((fraction * 100).rounded()))%"
    }

    var body: some View {
        if model.session != nil {
            Button {
                showsPopover.toggle()
            } label: {
                ring
                    .frame(width: 28, height: 28)
                    .background(
                        Color.primary.opacity(isHovered ? 0.08 : 0),
                        in: Circle()
                    )
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .onHover { isHovered = $0 }
            .help("Context usage: \(percentText) of the model's context window")
            .accessibilityLabel("Context usage")
            .accessibilityValue("\(percentText) of the context window")
            .appPointerCursor()
            .popover(
                isPresented: $showsPopover,
                attachmentAnchor: .point(.top),
                arrowEdge: .top
            ) {
                ContextUsagePopoverView(model: model)
            }
        }
    }

    private var ring: some View {
        ZStack {
            Circle()
                .stroke(Color.primary.opacity(0.15), lineWidth: 2.5)
            Circle()
                .trim(from: 0, to: fraction)
                .stroke(tint, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .frame(width: 13, height: 13)
        .animation(.linear(duration: 0.15), value: fraction)
    }
}

/// The context breakdown popover: the exact headline number, a stacked bar
/// of what the window holds, one row per component, and where auto-compact
/// fires. Claude Code's "Context window" panel minus the plan-limit rows,
/// which are Anthropic-side and have no equivalent here; the compaction
/// trigger is the one real limit this app has.
struct ContextUsagePopoverView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    private var summary: ContextUsageSummary? { model.contextUsageSummary }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            bar
            rows
            Divider()
            footer
        }
        .padding(14)
        .frame(width: 320, alignment: .leading)
        .background(TurboSparkTheme.surfaceColor)
        .onAppear { model.updateTokenEstimate() }
    }

    private var headlineText: String {
        guard let summary else { return "" }
        let used = summary.usedTokens.formatted(.number.notation(.compactName))
        let window = summary.windowTokens.formatted(.number.notation(.compactName))
        return "\(used) / \(window) (\(Int((summary.fraction * 100).rounded()))%)"
    }

    private var header: some View {
        HStack {
            Text("Context window", bundle: .module)
                .font(theme.ui(.callout, weight: .semibold))
            Spacer()
            if summary != nil {
                Text(headlineText)
                    .monospacedDigit()
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// The fill as one stacked capsule per component over the same track
    /// style the status bar's meter uses. Segment widths are fractions of
    /// the WINDOW, so unused space stays visibly empty.
    private var bar: some View {
        Capsule()
            .fill(Color.primary.opacity(0.08))
            .frame(height: 6)
            .overlay(
                GeometryReader { geo in
                    HStack(spacing: 1) {
                        if let summary {
                            ForEach(summary.components) { component in
                                Capsule()
                                    .fill(component.kind.segmentColor)
                                    .frame(
                                        width: max(
                                            2, geo.size.width * Double(component.tokens)
                                                / Double(max(1, summary.windowTokens))))
                            }
                        }
                    }
                }
            )
            .accessibilityHidden(true)
    }

    @ViewBuilder
    private var rows: some View {
        if let summary {
            VStack(alignment: .leading, spacing: 5) {
                ForEach(summary.components) { component in
                    row(
                        label: component.label,
                        tokens: component.tokens,
                        window: summary.windowTokens,
                        color: component.kind.segmentColor)
                }
                row(
                    label: "Free space",
                    tokens: summary.freeTokens,
                    window: summary.windowTokens,
                    color: Color.primary.opacity(0.15))
            }
        } else {
            Text("Measuring...", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.tertiary)
        }
    }

    private func row(label: String, tokens: Int, window: Int, color: Color) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            // Verbatim: the label is built at runtime, not a literal key.
            Text(verbatim: label)
                .font(theme.ui(.tiny))
                .foregroundStyle(.primary)
                .lineLimit(1)
            Spacer(minLength: 8)
            let percent = window > 0 ? Int((Double(tokens) / Double(window) * 100).rounded()) : 0
            Text("\(tokens.formatted(.number.notation(.compactName))) (\(percent)%)")
                .monospacedDigit()
                .font(theme.ui(.tiny))
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var footer: some View {
        if model.isCompacting {
            Text("Compacting conversation...", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(Color.orange)
        } else if let trigger = summary?.compactionTriggerTokens, summary?.autoCompactEnabled == true {
            Text(
                "Auto-compact at ~\(trigger.formatted(.number.notation(.compactName))) tokens",
                bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.tertiary)
        } else {
            Text("Auto-compact off", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.tertiary)
        }
        Text(
            "The total is exact; the rows apportion it and skip chat-template framing.",
            bundle: .module)
            .font(theme.ui(.tiny))
            .foregroundStyle(.tertiary)
            .fixedSize(horizontal: false, vertical: true)
    }
}
