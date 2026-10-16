import SwiftUI

/// The live state of a Voice Cloning run, in the content where the take will
/// appear: the stage list, the themed progress bar, elapsed time and Cancel.
/// Cancel carries no shortcut here because the status strip's Cancel owns
/// Cmd+Period for the whole workspace.
@MainActor
struct VoiceCloneProgressCard: View {
    @ObservedObject var controller: AudioWorkspaceController
    let run: AudioSpeechRunProgress

    @Environment(\.appTheme) private var theme

    /// The stages a cloning run shows. Waiting is the status line's job.
    private static let stages: [AudioSpeechRunProgress.Stage] = [.loadingModel, .preparingReference, .generating, .saving]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                TSIdlingSparkView(size: 40).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Generating speech", bundle: .module)
                        .themedFont(.base, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                    Text(verbatim: controller.status)
                        .themedFont(.small).foregroundStyle(.appSecondary)
                        .lineLimit(1).truncationMode(.tail)
                        .contentTransition(.opacity)
                }
                Spacer(minLength: 8)
                Button { controller.cancel() } label: { Text("Cancel", bundle: .module) }
                    .appAction(.secondary, size: .small)
            }
            AppProgressBar(value: controller.progress, label: Text(verbatim: controller.status))
                // A new stage starts its own bar instead of sliding back
                // from the last stage's full fraction.
                .id(run.stage)
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                stageList
                Spacer(minLength: 8)
                readout
            }
        }
        .tsAnimation(TSMotion.hover, value: controller.status)
        .appCardSurface()
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(theme.isHighContrast ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appAccent.opacity(0.4)), lineWidth: 1)
                .allowsHitTesting(false)
        }
        .shadow(color: theme.isHighContrast ? .clear : theme.accent.opacity(0.18), radius: 14, x: 0, y: 4)
        .accessibilityElement(children: .contain)
    }

    private var stageList: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { ForEach(Self.stages, id: \.self) { stageLabel($0) } }
            VStack(alignment: .leading, spacing: 6) { ForEach(Self.stages, id: \.self) { stageLabel($0) } }
        }
    }

    private func stageLabel(_ stage: AudioSpeechRunProgress.Stage) -> some View {
        let state = run.state(of: stage)
        return HStack(spacing: 5) {
            Image(systemName: symbol(state))
                .foregroundStyle(state == .pending ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(.appAccent))
                .contentTransition(.symbolEffect(.replace))
                .tsPulse(active: state == .active)
                .accessibilityHidden(true)
            stageTitle(stage)
                .foregroundStyle(state == .pending ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(.appText))
        }
        .themedFont(.tiny, weight: state == .active ? .semibold : .regular)
        .lineLimit(1)
        .tsAnimation(TSMotion.select, value: state)
        .accessibilityElement(children: .combine)
        .accessibilityValue(stateText(state))
    }

    @ViewBuilder private var readout: some View {
        HStack(spacing: 10) {
            if let progress = controller.progress {
                Text(max(0, min(1, progress)), format: .percent.precision(.fractionLength(0)))
                    .monospacedDigit()
                    .contentTransition(.numericText())
            }
            TimelineView(.periodic(from: .now, by: 1)) { context in
                HStack(spacing: 4) {
                    Text("Elapsed time", bundle: .module)
                    Text(verbatim: audioTime(context.date.timeIntervalSince(run.startedAt))).monospacedDigit()
                }
                .accessibilityElement(children: .combine)
            }
        }
        .themedFont(.tiny).foregroundStyle(.appSecondary)
        .fixedSize()
    }

    private func symbol(_ state: AudioSpeechRunProgress.StageState) -> String {
        switch state {
        case .pending: return "circle"
        case .active: return "circle.dotted.circle"
        case .done: return "checkmark.circle.fill"
        }
    }

    private func stageTitle(_ stage: AudioSpeechRunProgress.Stage) -> Text {
        switch stage {
        case .waiting: Text("Waiting for background work", bundle: .module)
        case .loadingModel: Text("Loading model", bundle: .module)
        case .preparingReference: Text("Preparing reference audio", bundle: .module)
        case .generating: Text("Generating speech", bundle: .module)
        case .saving: Text("Saving audio", bundle: .module)
        }
    }

    private func stateText(_ state: AudioSpeechRunProgress.StageState) -> Text {
        switch state {
        case .pending: Text("Pending", bundle: .module)
        case .active: Text("In progress", bundle: .module)
        case .done: Text("Done", bundle: .module)
        }
    }
}
