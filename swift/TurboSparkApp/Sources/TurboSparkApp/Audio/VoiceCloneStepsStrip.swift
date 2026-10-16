import SwiftUI

/// Where the user is in cloning a voice: one button per step, each with a
/// glyph and a state word so the state never rests on color alone. A step
/// scrolls to its section and moves focus there.
@MainActor
struct VoiceCloneStepsStrip: View {
    let steps: [VoiceCloneStepStatus]
    let select: (VoiceCloneStep) -> Void

    /// The first step still waiting on the user, announced as selected.
    private var current: VoiceCloneStep? {
        steps.first { $0.state != .done }?.step
    }

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 6) {
                ForEach(Array(steps.enumerated()), id: \.element.id) { index, status in
                    if index > 0 {
                        Image(systemName: "chevron.forward")
                            .themedFont(.tiny).foregroundStyle(.appSecondary)
                            .accessibilityHidden(true)
                    }
                    stepButton(index: index, status: status)
                }
                Spacer(minLength: 0)
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 8, alignment: .leading)], alignment: .leading, spacing: 8) {
                ForEach(Array(steps.enumerated()), id: \.element.id) { index, status in
                    stepButton(index: index, status: status)
                }
            }
        }
        .accessibilityElement(children: .contain)
    }

    private func stepButton(index: Int, status: VoiceCloneStepStatus) -> some View {
        VoiceCloneStepButton(number: index + 1, status: status, isCurrent: status.step == current) {
            select(status.step)
        }
    }
}

@MainActor
private struct VoiceCloneStepButton: View {
    let number: Int
    let status: VoiceCloneStepStatus
    let isCurrent: Bool
    let action: () -> Void

    @Environment(\.appTheme) private var theme
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Image(systemName: symbol)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(tint)
                    .contentTransition(.symbolEffect(.replace))
                    .tsPulse(active: status.state == .active)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 0) {
                    title.themedFont(.small, weight: .semibold).foregroundStyle(.appText)
                    stateText.themedFont(.tiny).foregroundStyle(.appSecondary)
                }
                .lineLimit(1)
            }
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(background, in: Capsule())
            .overlay {
                Capsule().stroke(edge, lineWidth: isCurrent || theme.isHighContrast ? 1.5 : 1)
            }
            .contentShape(Capsule())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.97))
        .onHover { hovered = $0 }
        .tsAnimation(TSMotion.select, value: status.state)
        .tsAnimation(TSMotion.hover, value: hovered)
        .fixedSize()
        .help(title)
        .accessibilityLabel(title)
        .accessibilityValue(stateText)
        .accessibilityAddTraits(isCurrent ? .isSelected : [])
    }

    private var title: Text {
        switch status.step {
        case .voice: Text("Voice", bundle: .module)
        case .consent: Text("Consent", bundle: .module)
        case .transcript: Text("Transcript", bundle: .module)
        case .script: Text("Script", bundle: .module)
        case .generate: Text("Generate", bundle: .module)
        }
    }

    private var stateText: Text {
        switch status.state {
        case .todo: Text("To do", bundle: .module)
        case .attention: Text("Needs attention", bundle: .module)
        case .done: Text("Done", bundle: .module)
        case .ready: Text("Ready", bundle: .module)
        case .active: Text("In progress", bundle: .module)
        }
    }

    private var symbol: String {
        switch status.state {
        case .todo: return "\(number).circle"
        case .attention: return "exclamationmark.triangle.fill"
        case .done: return "checkmark.circle.fill"
        case .ready: return "play.circle.fill"
        case .active: return "waveform.circle.fill"
        }
    }

    private var tint: Color {
        switch status.state {
        case .todo: return theme.secondaryText
        case .attention: return .orange
        case .done, .ready, .active: return theme.accent
        }
    }

    private var background: Color {
        switch status.state {
        case .done, .ready, .active: return theme.accent.opacity(hovered ? 0.18 : 0.12)
        case .attention: return Color.orange.opacity(hovered ? 0.16 : 0.10)
        case .todo: return Color.primary.opacity(hovered ? 0.08 : 0.04)
        }
    }

    private var edge: Color {
        if theme.isHighContrast { return isCurrent ? theme.accent : theme.border }
        return isCurrent ? theme.accent.opacity(0.55) : .clear
    }
}
