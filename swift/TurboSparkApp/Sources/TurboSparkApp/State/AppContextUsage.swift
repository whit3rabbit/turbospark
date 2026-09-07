import Foundation
import SwiftUI
import TurboSpark

/// The context-window breakdown behind the composer's context ring: what the
/// next turn's prompt is made of and how much of the window it occupies.
///
/// The EXACT total is the same template-rendered `countTokens(history)` the
/// composer's estimate uses, so the ring, the status bar readout and the
/// popover cannot disagree about how full the window is. The per-component
/// rows are cheaper raw-string counts (`countTokens(in:)`, no template
/// render per row), so they APPORTION that total rather than summing to it:
/// chat-template role framing lands on the conversation as a whole and not
/// on any one row. That asymmetry is deliberate and is why the popover leads
/// with the exact number.
///
/// The assembly itself lives in `AppModel+History.buildEstimateParts`, which
/// is the same function the exact estimate uses -- one builder, two
/// consumers, by the same rule that keeps the estimate and the prompt
/// describing the same conversation.
struct ContextUsagePiece {
    let kind: ContextUsageKind
    let label: String
    let content: String
}

/// One row of the breakdown: a labeled slice of the next prompt and its
/// raw-string token count.
public struct ContextUsageComponent: Identifiable {
    public let kind: ContextUsageKind
    public let label: String
    public let tokens: Int

    public var id: String { label }

    public init(kind: ContextUsageKind, label: String, tokens: Int) {
        self.kind = kind
        self.label = label
        self.tokens = tokens
    }
}

/// The color family of a breakdown row and of its segment in the popover's
/// stacked bar.
public enum ContextUsageKind {
    case system
    case tools
    case memory
    case conversation
    case summary
    case attachments
    case draft

    /// MainActor because the accent is theme-resolved; every reader is a
    /// view body.
    @MainActor
    var segmentColor: Color {
        switch self {
        case .system: return .blue
        case .tools: return .purple
        case .memory: return .teal
        case .conversation: return TurboSparkTheme.accentColor
        case .summary: return .orange
        case .attachments: return .pink
        case .draft: return .yellow
        }
    }
}

/// A published snapshot of what the window holds right now. Nil until the
/// first estimate after a session loads, and never updated mid-turn (the
/// counting dispatches onto the session's serial queue, which a running
/// generation holds).
public struct ContextUsageSummary {
    let components: [ContextUsageComponent]
    /// The exact whole-history count from the template render, when one has
    /// been taken. Every headline number reads this; the rows only apportion.
    let measuredTokens: Int?
    let windowTokens: Int
    /// Where auto-compact fires, in the same tokens the engine itself will
    /// compare against. Nil when the reply reservation leaves nothing usable.
    let compactionTriggerTokens: Int?
    let autoCompactEnabled: Bool

    init(
        components: [ContextUsageComponent],
        measuredTokens: Int?,
        windowTokens: Int,
        reservedForNewTokens: Int,
        autoCompactEnabled: Bool
    ) {
        self.components = components
        self.measuredTokens = measuredTokens
        self.windowTokens = windowTokens
        self.autoCompactEnabled = autoCompactEnabled
        // Ceiling division, so a prompt counted AT this number satisfies
        // `AppChatCompaction.shouldAutoCompact` and one token less does not.
        // The same constants, not a restated fraction, for the same reason
        // the status bar's hint runs through them.
        let usable = windowTokens - max(1, reservedForNewTokens)
        if usable > 0 {
            self.compactionTriggerTokens =
                (usable * AppChatCompaction.triggerNumerator
                    + AppChatCompaction.triggerDenominator - 1)
                / AppChatCompaction.triggerDenominator
        } else {
            self.compactionTriggerTokens = nil
        }
    }

    /// The headline numerator: the exact count when known, the rows' sum
    /// otherwise (an approximation, and only before the first estimate).
    var usedTokens: Int {
        measuredTokens ?? components.reduce(0) { $0 + $1.tokens }
    }

    var freeTokens: Int {
        max(0, windowTokens - usedTokens)
    }

    /// Fraction of the window occupied, clamped into [0, 1]. The ring and
    /// the popover bar both render from this.
    var fraction: Double {
        guard windowTokens > 0 else { return 0 }
        return min(1, Double(usedTokens) / Double(windowTokens))
    }

    /// The ring's tiers: green while comfortable, yellow from 70 percent,
    /// red from 90. Claude Code's thresholds, which is also the point at
    /// which this app's own auto-compact (4/5 of usable) is close enough to
    /// matter.
    static func tint(forFraction fraction: Double) -> Color {
        if fraction >= 0.9 { return .red }
        if fraction >= 0.7 { return .yellow }
        return .green
    }
}

extension AppModel {
    /// Counts the labeled pieces of the assembled prompt and publishes
    /// `contextUsageSummary`.
    ///
    /// Called from `updateTokenEstimate`'s existing debounced task with the
    /// exact count that task already took, so the breakdown and the estimate
    /// are one snapshot, never two. Raw-string counts only: each is one
    /// tokenize with no template render, which keeps ~8 dispatches on the
    /// session's serial queue cheap enough to run on every estimate.
    func countAndStoreContextUsage(
        pieces: [ContextUsagePiece],
        attachmentTokens: Int,
        exactTokens: Int?,
        session: TurboSparkSession
    ) async {
        var components: [ContextUsageComponent] = []
        for piece in pieces where !piece.content.isEmpty {
            let tokens = (try? await session.countTokens(in: piece.content)) ?? 0
            guard tokens > 0 else { continue }
            components.append(
                ContextUsageComponent(kind: piece.kind, label: piece.label, tokens: tokens))
        }
        if attachmentTokens > 0 {
            // Images are priced at the same characters-per-token
            // approximation the headline uses: the exact count cannot see
            // them at all (one template marker per image), so there is
            // nothing exact to be exact against.
            components.append(
                ContextUsageComponent(
                    kind: .attachments, label: "Attachments", tokens: attachmentTokens))
        }
        guard !Task.isCancelled else { return }
        contextUsageSummary = ContextUsageSummary(
            components: components,
            measuredTokens: exactTokens,
            windowTokens: resolvedContextTokens,
            reservedForNewTokens: maxNewTokens,
            autoCompactEnabled: autoCompactEnabled)
    }
}
