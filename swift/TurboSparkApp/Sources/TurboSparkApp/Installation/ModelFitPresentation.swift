import SwiftUI
import TurboSpark

/// How a fit verdict is shown, in one place.
///
/// **THIS EXISTS BECAUSE THE MAPPING WAS WRITTEN OUT THREE TIMES.**
/// `ModelHubFilter.fitLabel`, `ModelHardwareFitCardView.verdictPill` and
/// `ModelCardView.verdictBadge` each restated the same five cases with their
/// own strings, and a restatement is correct on the day it is written and
/// silently wrong afterwards (swift Gotcha 22). One of them had drifted into
/// a capability claim: the `.streams` badge advertised "~2-4 GB RAM" in its
/// tooltip, a hardcoded figure for every MoE checkpoint regardless of layer
/// count or expert stride.
///
/// A value type rather than inline view code, so the branches can be tested
/// at all: `ServerStatusRows` is the same argument (swift Gotcha 26), and
/// these decide whether a user is shown a number or told there isn't one.
struct ModelFitPresentation: Equatable {
    /// Full label, for a detail pane.
    let label: String
    /// Short label, for a row badge where width is scarce.
    let compactLabel: String
    /// SF Symbol, or `nil` for a text-only pill.
    let iconName: String?
    /// What this verdict means, in words that do not name a byte count. A
    /// footprint belongs beside its context and slot count, not in a tooltip.
    let help: String

    /// The five verdicts and their tones. Green and blue both RUN: streaming
    /// experts from disk is what this engine is for, so it is a fit and not a
    /// caution.
    enum Tone: Equatable {
        case good, streaming, caution, blocked, unknown
    }
    let tone: Tone

    var color: Color {
        switch tone {
        case .good: return .green
        case .streaming: return .blue
        case .caution: return .orange
        case .blocked: return .red
        case .unknown: return .secondary
        }
    }

    static func of(_ verdict: ModelRecommendation.FitVerdict) -> ModelFitPresentation {
        switch verdict {
        case .resident:
            return .init(
                label: "Full Resident Fit", compactLabel: "Resident",
                iconName: "checkmark.circle.fill",
                help: "Fits entirely in unified memory, install included.",
                tone: .good)
        case .streams:
            return .init(
                label: "Fast MoE Streaming", compactLabel: "Streams",
                iconName: "bolt.fill",
                help: "What the engine allocates fits. Routed experts are read "
                    + "from storage on demand, which is what this engine is built for.",
                tone: .streaming)
        case .tight:
            return .init(
                label: "Tight Memory Headroom", compactLabel: "Tight",
                iconName: "exclamationmark.triangle.fill",
                help: "Runs, with under ten percent to spare. Anything else on "
                    + "the machine competes with it.",
                tone: .caution)
        case .refused:
            return .init(
                label: "Exceeds Memory Limit", compactLabel: "Too Large",
                iconName: "xmark.octagon.fill",
                help: "What the engine allocates does not fit. The KV cache and "
                    + "the expert streamer both allocate up front, so this is a "
                    + "failed load rather than a slow one.",
                tone: .blocked)
        case .unknown:
            return .init(
                label: "Not Sized", compactLabel: "Unsized",
                iconName: "questionmark.circle",
                help: "Nothing has read this checkpoint's header, so its "
                    + "footprint is unknown. Probe it.",
                tone: .unknown)
        }
    }

    /// The short label the hub's filter menu groups by. Kept as its own
    /// spelling because those strings are persisted in a picker selection.
    static func filterLabel(_ verdict: ModelRecommendation.FitVerdict) -> String {
        switch verdict {
        case .resident: return "Fits in memory"
        case .streams: return "Streams"
        case .tight: return "Tight fit"
        case .refused: return "Too large"
        case .unknown: return "Unknown"
        }
    }
}

/// The verdict pill. One view, three call sites.
///
/// **`.unknown` RENDERS.** It used to be `EmptyView()` in both places that
/// drew a pill, which is why an unsized row could sit beside a grid of zeros
/// with nothing saying the zeros were not measurements (swift Gotcha 23).
struct ModelFitVerdictPill: View {
    let verdict: ModelRecommendation.FitVerdict
    var compact: Bool = false

    var body: some View {
        let p = ModelFitPresentation.of(verdict)
        HStack(spacing: compact ? 2 : 3) {
            if let icon = p.iconName {
                Image(systemName: icon)
                    .themedFont(.micro, weight: .bold)
                    .accessibilityHidden(true)
            }
            Text(compact ? p.compactLabel : p.label)
        }
        .themedFont(compact ? .tiny : .small, weight: compact ? .semibold : .bold)
        .lineLimit(1)
        .fixedSize()
        .padding(.horizontal, compact ? 4 : 8)
        .padding(.vertical, compact ? 1.5 : 3)
        .background(p.color.opacity(0.18), in: Capsule())
        .foregroundStyle(p.color)
        .help(p.help)
    }
}
