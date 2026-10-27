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
    @Environment(\.appTheme) private var theme
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
                        isHovered ? AnyShapeStyle(.appElevated) : AnyShapeStyle(Color.clear),
                        in: Circle()
                    )
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .onHover { isHovered = $0 }
            .help(Text("Context usage: \(percentText) of the model's context window", bundle: .module))
            .accessibilityLabel(Text("Context usage", bundle: .module))
            .accessibilityValue(Text("\(percentText) of the context window", bundle: .module))
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
                .stroke(theme.border, lineWidth: 2.5)
            Circle()
                .trim(from: 0, to: fraction)
                .stroke(tint, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .frame(width: 13, height: 13)
        .tsAnimation(TSMotion.select, value: fraction)
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
        .frame(minWidth: 280, idealWidth: 320, maxWidth: 400, alignment: .leading)
        .background(.appSurface)
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
                .accessibilityAddTraits(.isHeader)
            Spacer()
            if summary != nil {
                Text(headlineText)
                    .monospacedDigit()
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.appSecondary)
            }
        }
    }

    /// The fill as one stacked capsule per component over the same track
    /// style the status bar's meter uses. Segment widths are fractions of
    /// the WINDOW, so unused space stays visibly empty.
    private var bar: some View {
        Capsule()
            .fill(theme.border.opacity(0.5))
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
                    // Every component label `AppContextUsage` builds is
                    // fixed English and a catalog key.
                    row(
                        label: Text(LocalizedStringKey(component.label), bundle: .module),
                        tokens: component.tokens,
                        window: summary.windowTokens,
                        color: component.kind.segmentColor)
                }
                row(
                    label: Text("Free space", bundle: .module),
                    tokens: summary.freeTokens,
                    window: summary.windowTokens,
                    color: theme.border)
            }
        } else {
            Text("Measuring...", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
        }
    }

    private func row(label: Text, tokens: Int, window: Int, color: Color) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            label
                .font(theme.ui(.tiny))
                .foregroundStyle(.appText)
                .lineLimit(1)
            Spacer(minLength: 8)
            let percent = window > 0 ? Int((Double(tokens) / Double(window) * 100).rounded()) : 0
            // A numeric readout: nothing in it to translate.
            Text(verbatim: "\(tokens.formatted(.number.notation(.compactName))) (\(percent)%)")
                .monospacedDigit()
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
        }
    }

    @ViewBuilder
    private var footer: some View {
        if model.isCompacting {
            Text("Compacting conversation...", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(AppAttention.color)
        } else if let trigger = summary?.compactionTriggerTokens, summary?.autoCompactEnabled == true {
            Text(
                "Auto-compact at ~\(trigger.formatted(.number.notation(.compactName))) tokens",
                bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
        } else {
            Text("Auto-compact off", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
        }
        Text(
            "The total is exact; the rows apportion it and skip chat-template framing.",
            bundle: .module)
            .font(theme.ui(.tiny))
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}
