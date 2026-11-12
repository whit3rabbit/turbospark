import SwiftUI

/// The run button between the cards, the section's one primary action; a
/// stop target while a rewrite runs. Clear Both sits under it, a quiet chip
/// that empties both cards, so the three clear actions mirror the layout:
/// Clear Draft on the left, Clear Both between, Clear Rewrite on the right.
@MainActor
struct HumanizeRewriteRail: View {
    @ObservedObject var controller: HumanizeController
    let wide: Bool
    let canClearAll: Bool
    let onClearAll: () -> Void

    @Environment(\.appTheme) private var theme

    private var scale: CGFloat { HumanizeMetrics.textScale(theme) }

    var body: some View {
        let busy = controller.isBusy
        let ready = busy || controller.canRun
        let layout = wide ? AnyLayout(VStackLayout(spacing: 8)) : AnyLayout(HStackLayout(spacing: 12))
        layout {
            HumanizeRunButton(
                busy: busy, ready: ready, prefill: busy ? controller.prefillProgress : nil,
                // "forward", not "right": the draft sits on the right in Arabic
                // and Hebrew, and only the forward symbol flips with the layout.
                arrow: wide ? "arrow.forward" : "arrow.down",
                invite: controller.canRun && controller.rewrite.isEmpty
            ) {
                if busy { controller.cancel() } else { controller.run() }
            }
            .disabled(!ready)
            .keyboardShortcut(busy ? KeyEquivalent(".") : .return, modifiers: .command)
            // "Rewrite draft", not "Rewrite": that key is the result card's
            // title and is translated as a noun ("the rewrite"), which named
            // this button "Rewritten text" in several languages.
            .help(busy ? Text("Stop", bundle: .module) : Text("Rewrite draft", bundle: .module))
            .accessibilityLabel(busy ? Text("Stop", bundle: .module) : Text("Rewrite draft", bundle: .module))
            .accessibilityIdentifier("humanize.run")
            VStack(spacing: 4) {
                // The verb key, like the tooltip: "Rewrite" is the result
                // card's noun title. Two centered lines at most, in a bounded
                // width, so a long translation wraps instead of widening the
                // rail between the cards.
                Group {
                    if busy { Text("Stop", bundle: .module) } else { Text("Rewrite draft", bundle: .module) }
                }
                .font(theme.ui(.micro, weight: .semibold))
                .textCase(.uppercase)
                .tracking(1)
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: (96 * scale).rounded())
                if let part = controller.partProgress {
                    ModelLoadProgressBar(
                        fraction: Double(part.current - 1) / Double(part.total), tint: theme.accent)
                        .frame(width: (40 * scale).rounded())
                    Text(verbatim: "\(part.current)/\(part.total)").font(theme.ui(.micro)).monospacedDigit()
                        .lineLimit(1)
                } else {
                    Text(verbatim: busy ? "\u{2318}." : "\u{2318}\u{21A9}").font(theme.ui(.micro))
                        .lineLimit(1)
                }
            }
            .foregroundStyle(.appSecondary)
            .frame(maxWidth: (96 * scale).rounded())
            .accessibilityHidden(true)
            // The label first; a translation too long for the rail keeps just
            // the glyph, with the name as tooltip and VoiceOver label.
            ViewThatFits(in: .horizontal) {
                clearAllChip(iconOnly: false)
                clearAllChip(iconOnly: true)
            }
            .frame(maxWidth: wide ? (96 * scale).rounded() : nil)
            .padding(wide ? Edge.Set.top : Edge.Set.leading, 10)
        }
        .tsAnimation(TSMotion.select, value: busy)
    }

    private func clearAllChip(iconOnly: Bool) -> some View {
        AppChip(Text("Clear Both", bundle: .module), systemImage: "eraser.line.dashed", iconOnly: iconOnly,
                action: onClearAll)
            .disabled(!canClearAll)
            .accessibilityIdentifier("humanize.clearAll")
    }
}

/// What the model is doing while a rewrite runs.
@MainActor
struct HumanizeProgressStatus: View {
    @ObservedObject var controller: HumanizeController

    @Environment(\.appTheme) private var theme

    var body: some View {
        HStack(spacing: 8) {
            ModelStatusDot(state: .opening, accent: theme.accent)
            switch controller.phase {
            case .idle: EmptyView()
            case .loadingModel: Text("Loading the Humanizer model...", bundle: .module)
            case .running:
                if controller.prefillProgress != nil {
                    Text("Reading the text...", bundle: .module)
                } else {
                    Text("Rewriting...", bundle: .module)
                }
            }
            if let part = controller.partProgress {
                Text("Part \(part.current) of \(part.total)", bundle: .module).monospacedDigit()
            }
        }
        .font(theme.ui(.small))
        .foregroundStyle(.appSecondary)
        .lineLimit(1)
    }
}

/// The notes under the cards: errors, the long-draft and cut-off notices,
/// and one line per check finding.
@MainActor
struct HumanizeFeedback: View {
    @ObservedObject var controller: HumanizeController

    @Environment(\.appTheme) private var theme

    var body: some View {
        let plan = controller.currentPlan
        let warnings = HumanizeAuditLines.warnings(for: controller)
        VStack(alignment: .leading, spacing: 8) {
            if let message = controller.errorMessage ?? controller.contextError {
                Label(message, systemImage: "xmark.octagon").foregroundStyle(.appDanger).textSelection(.enabled)
            } else if let plan, !plan.fits {
                Label(HumanizeError.passageTooLong.localizedDescription, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(AppAttention.color)
            }
            if let plan, plan.parts.count > 1 {
                Label { Text("Long draft: rewritten in parts at paragraph breaks.", bundle: .module) }
                    icon: { Image(systemName: "rectangle.split.3x1") }
                    .foregroundStyle(.appSecondary)
            }
            if controller.wasTruncated {
                Label { Text("The rewrite stopped at the length limit and may be cut off.", bundle: .module) }
                    icon: { Image(systemName: "scissors") }
                    .foregroundStyle(AppAttention.color)
            }
            if controller.stopReason == .cancelled {
                Text("Stopped. The rewrite is incomplete.", bundle: .module).foregroundStyle(.appSecondary)
            }
            if !controller.rewrite.isEmpty, !controller.isBusy, controller.draft != controller.sourceDraft {
                Text("The draft has changed. Highlights compare with the version used for this rewrite.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            }
            ForEach(warnings, id: \.self) { line in
                Label(line, systemImage: "exclamationmark.triangle").foregroundStyle(AppAttention.color)
            }
        }
        .font(theme.ui(.small))
        .frame(maxWidth: .infinity, alignment: .leading)
        .tsAnimation(TSMotion.hover, value: warnings)
    }
}

/// Localized sentences for the checks, built from their structured fields
/// (the engine's own `warnings` are English only). Every line is a pointer
/// for the proofreader, never a claim that the rewrite is wrong.
///
/// Resolved through `AppStrings`: these lines sit beside labels that follow
/// the in-app language picker, and a plain bundle lookup follows the system
/// language instead.
enum HumanizeAuditLines {
    /// The model's audit and the user's rules, in that order.
    @MainActor
    static func warnings(for controller: HumanizeController) -> [String] {
        (controller.checks.map(lines(for:)) ?? []) + (controller.ruleFindings.map(lines(for:)) ?? [])
    }

    static func lines(for checks: HumanizeChecks) -> [String] {
        var out: [String] = []
        if !checks.droppedNumbers.isEmpty {
            let numbers = checks.droppedNumbers.joined(separator: ", ")
            out.append(AppStrings.localized("Numbers missing from the rewrite: \(numbers)"))
        }
        if !checks.inventedNumbers.isEmpty {
            let numbers = checks.inventedNumbers.joined(separator: ", ")
            out.append(AppStrings.localized("Numbers not in the original: \(numbers)"))
        }
        if checks.languageDrift {
            out.append(AppStrings.localized("The rewrite looks like a different language than the original."))
        }
        if checks.reuseFlagged {
            out.append(AppStrings.localized("The rewrite copies long phrases from the original."))
        }
        return out
    }

    static func lines(for findings: HumanizeRuleFindings) -> [String] {
        var out: [String] = []
        if !findings.missingProtected.isEmpty {
            let terms = findings.missingProtected.joined(separator: ", ")
            out.append(AppStrings.localized("Protected terms missing from the rewrite: \(terms)"))
        }
        if !findings.avoidFound.isEmpty {
            let terms = findings.avoidFound.joined(separator: ", ")
            out.append(AppStrings.localized("Words to avoid found in the rewrite: \(terms)"))
        }
        return out
    }
}
