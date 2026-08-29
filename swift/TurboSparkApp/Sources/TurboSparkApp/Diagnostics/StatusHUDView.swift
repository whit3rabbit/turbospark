import SwiftUI

struct StatusHUDView: View {
    @ObservedObject var model: AppModel
    let isChatSidebarVisible: Bool
    let isInspectorVisible: Bool
    let toggleChatSidebar: () -> Void
    let toggleInspector: () -> Void

    @ScaledMetric private var toggleButtonSize: CGFloat = 28
    @ScaledMetric private var dividerHeight: CGFloat = 16
    @ScaledMetric private var hudMinHeight: CGFloat = 30

    var body: some View {
        strip
            .padding(.top, 10)
            .padding(.horizontal, CGFloat(AppChromeLayout.headerHorizontalPadding))
    }

    private var strip: some View {
        HStack(spacing: 12) {
            chatSidebarToggle
            Divider().frame(height: dividerHeight)
            ModelStatusBadge(model: model)
            Divider().frame(height: dividerHeight)
            PhaseLabel(model: model)
            Spacer(minLength: 12)
            if showsMetrics {
                HUDMetricView(value: rateText, label: "tok/s", animated: !model.isRunning)
                HUDMetricView(value: tokensText, label: "tokens", animated: !model.isRunning)
                HUDMetricView(value: memoryText, label: "memory", animated: !model.isRunning)
            }
            inspectorToggle
        }
        .frame(minHeight: hudMinHeight)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background {
            Capsule()
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    Capsule().stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    private var chatSidebarToggle: some View {
        let presentation = AppSidebarControlPresentation(
            sidebar: .chats,
            isVisible: isChatSidebarVisible)
        return Button(action: toggleChatSidebar) {
            Label(
                presentation.title,
                systemImage: presentation.systemImage)
                .labelStyle(.iconOnly)
                .frame(width: toggleButtonSize, height: toggleButtonSize)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(isChatSidebarVisible ? .primary : .secondary)
        .keyboardShortcut("s", modifiers: [.command, .control])
        .help(presentation.help)
        // .labelStyle(.iconOnly) drops the title from the accessibility label,
        // so VoiceOver would otherwise only hear "button". Explicit labels
        // re-supply the human-readable action.
        .accessibilityLabel(presentation.title)
        .accessibilityHint(presentation.help)
        .accessibilityValue(presentation.accessibilityValue)
    }

    private var inspectorToggle: some View {
        let presentation = AppSidebarControlPresentation(
            sidebar: .inspector,
            isVisible: isInspectorVisible)
        return Button(action: toggleInspector) {
            Label(
                presentation.title,
                systemImage: presentation.systemImage)
                .labelStyle(.iconOnly)
                .frame(width: toggleButtonSize, height: toggleButtonSize)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(isInspectorVisible ? .primary : .secondary)
        .keyboardShortcut("i", modifiers: [.command, .shift])
        .help(presentation.help)
        .accessibilityLabel(presentation.title)
        .accessibilityHint(presentation.help)
        .accessibilityValue(presentation.accessibilityValue)
    }


    private var rateText: String {
        if model.phase == .decode { return MetricFormat.rate(model.liveTokensPerSecond) }
        if let d = model.diagnostics { return MetricFormat.rate(d.tokensPerSecond) }
        return "\u{2014}"
    }

    private var tokensText: String {
        if model.isRunning { return "\(model.liveTokenCount)" }
        if let d = model.diagnostics { return "\(d.generatedTokens)" }
        return "\u{2014}"
    }

    private var memoryText: String {
        MetricFormat.memory(model.currentProcessMemoryBytes)
    }

    private var showsMetrics: Bool {
        model.session != nil || model.isRunning || model.diagnostics != nil
    }
}

private struct PhaseLabel: View {
    @ObservedObject var model: AppModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var statusText: String {
        if model.opening { return "Opening" }
        if model.isInstallingModel { return model.installStageText ?? "Installing" }
        switch model.phase {
        case .prefill:
            return "Prefill \(model.livePrefillDone)/\(max(model.livePrefillTotal, 1))"
        case .decode:
            return "Generating"
        case .idle:
            return model.session != nil ? "Ready" : "Idle"
        }
    }

    var body: some View {
        HStack(spacing: 6) {
            if model.opening {
                ProgressView().controlSize(.mini)
                Text(statusText).foregroundStyle(.secondary)
            } else if model.isInstallingModel {
                ProgressView().controlSize(.mini)
                Text(statusText).foregroundStyle(.secondary)
            } else if model.phase == .prefill {
                if reduceMotion {
                    Circle().fill(TurboSparkTheme.accentColor).frame(width: 7, height: 7)
                } else {
                    PulsingDot()
                }
                Text(statusText)
                    .monospacedDigit()
            } else if model.phase == .decode {
                Circle().fill(TurboSparkTheme.accentColor).frame(width: 7, height: 7)
                Text(statusText).foregroundStyle(.primary)
            } else if model.session != nil {
                Circle().fill(.green).frame(width: 7, height: 7)
                Text(statusText).foregroundStyle(.secondary)
            } else {
                Text(statusText).foregroundStyle(.secondary)
            }
        }
        .font(.caption.weight(.medium))
        .lineLimit(1)
        // Combine the indicator dot/spinner and text into a single accessibility
        // element so VoiceOver announces "Model status: Generating" rather
        // than reading the label and the decorative dot separately.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Model status")
        .accessibilityValue(statusText)
    }
}

private struct PulsingDot: View {
    var body: some View {
        Circle()
            .fill(TurboSparkTheme.accentColor)
            .frame(width: 7, height: 7)
            .phaseAnimator([0.4, 1.0]) { dot, opacity in
                dot.opacity(opacity)
            } animation: { _ in
                .easeInOut(duration: 0.7)
            }
    }
}
