import SwiftUI

/// Flat top bar: sidebar toggles, live machine load, model loader.
///
/// **The Chat/Projects segment is gone from here.** It was the second copy of
/// a control that already lives in `ChatSidebarView.tabSelector`, and the two
/// were never independent -- both wrote `model.setInteractionMode(_:)`. A mode
/// switch you can make in two places one inch apart is not two affordances,
/// it is one affordance the user has to reconcile. The sidebar keeps it,
/// because the thing the mode changes IS the sidebar's contents.
///
/// The freed centre now carries the polled machine load (memory, CPU, GPU
/// wait). That is a deliberate reversal of the note this file used to carry
/// ("Metrics moved to the bottom status strip: a number that updates once per
/// token pulls the eye"), and the reversal holds ONLY for the slow numbers:
/// `ChromeTelemetryView` polls on the same 2-second timer the status strip
/// used, so nothing here moves per token. tok/s and the token count stay out
/// of this bar for exactly the original reason -- see `StatusBarView`, which
/// keeps them.
struct TopBarView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let isChatSidebarVisible: Bool
    let isInspectorVisible: Bool
    let toggleChatSidebar: () -> Void
    let toggleInspector: () -> Void

    @ScaledMetric private var buttonSize: CGFloat = 26
    @State private var confirmingEndGhost = false

    var body: some View {
        HStack(spacing: 8) {
            HStack(spacing: 8) {
                sidebarToggle
                GenerationPhaseIndicator(model: model)
                if let worktree = model.worktree, worktree.isGitRepository {
                    gitPill(worktree: worktree)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Centre. Fixed-size, so the flexible groups either side keep
            // their own widths and the model loader does not slide when a
            // phase indicator appears.
            ChromeTelemetryView(model: model)
                .fixedSize()

            HStack(spacing: 8) {
                ModelLoaderControl(model: model)
                inspectorToggle
                ghostButton
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding(.leading, AppChromeLayout.trafficLightClearance)
        .padding(.trailing, 10)
        .frame(height: AppChromeLayout.topBarHeight)
        .background(TurboSparkTheme.barBackgroundColor)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 0.5)
        }
    }

    private var sidebarToggle: some View {
        let presentation = AppSidebarControlPresentation(
            sidebar: .chats,
            isVisible: isChatSidebarVisible)
        return Button(action: toggleChatSidebar) {
            Image(systemName: presentation.systemImage)
                .font(theme.ui(points: 13, weight: .medium))
                .frame(width: buttonSize, height: buttonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.9))
        .foregroundStyle(isChatSidebarVisible ? Color.primary : Color.secondary)
        // No `.keyboardShortcut` here (U7): the "View" menu's "Toggle Chat
        // Sidebar" command already claims Cmd+Ctrl+S, and SwiftUI does not
        // define which of two identical shortcuts on different controls wins.
        .help(presentation.help)
        .accessibilityLabel(presentation.title)
        .accessibilityHint(presentation.help)
        .accessibilityValue(presentation.accessibilityValue)
    }

    private var inspectorToggle: some View {
        let presentation = AppSidebarControlPresentation(
            sidebar: .inspector,
            isVisible: isInspectorVisible)
        return Button(action: toggleInspector) {
            Image(systemName: presentation.systemImage)
                .font(theme.ui(points: 13, weight: .medium))
                .frame(width: buttonSize, height: buttonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.9))
        .foregroundStyle(isInspectorVisible ? theme.accent : Color.secondary)
        .background {
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(isInspectorVisible ? theme.accent.opacity(0.14) : .clear)
        }
        .animation(TSMotion.hover, value: isInspectorVisible)
        // No `.keyboardShortcut` here (U7): the "View" menu's "Toggle
        // Inspector" command claims Cmd+Shift+I and its handler also
        // dismisses an open attachment preview first.
        .help(presentation.help)
        .accessibilityLabel(presentation.title)
        .accessibilityHint(presentation.help)
        .accessibilityValue(presentation.accessibilityValue)
    }

    /// Ghost Mode: start a temporary chat, or end the one being viewed.
    private var ghostButton: some View {
        Button {
            if model.isInGhostChat {
                if model.ghostChatHasContent {
                    confirmingEndGhost = true
                } else {
                    model.endGhostChat()
                }
            } else {
                model.enterGhostChat()
            }
        } label: {
            Image(systemName: "ghost")
                .font(theme.ui(points: 13, weight: .medium))
                .frame(width: buttonSize, height: buttonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.9))
        .foregroundStyle(model.isInGhostChat ? theme.accent : Color.secondary)
        // `enterGhostChat()`/`endGhostChat()` both no-op under exactly this
        // guard, matching the File menu's "New Temporary Chat".
        .disabled(model.generating || model.submitting || model.pendingToolCall != nil)
        .help(model.isInGhostChat ? "End temporary chat" : "New temporary chat")
        .accessibilityLabel(model.isInGhostChat ? "End temporary chat" : "New temporary chat")
        .accessibilityHint(
            model.isInGhostChat
                ? "Discards the temporary conversation. It is never saved."
                : "Starts a temporary chat that lives only in memory and is never saved.")
        .accessibilityAddTraits(model.isInGhostChat ? [.isButton, .isSelected] : .isButton)
        .confirmationDialog(
            "End temporary chat?", isPresented: $confirmingEndGhost,
            titleVisibility: .visible
        ) {
            Button("Discard Temporary Chat", role: .destructive) {
                model.endGhostChat()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                "This conversation exists only in memory and cannot be recovered "
                    + "once discarded."
            )
        }
    }

    private func gitPill(worktree: WorktreeModel) -> some View {
        Button {
            NotificationCenter.default.post(name: .toggleInspector, object: nil)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "point.topleft.down.to.point.bottomright.curvepath")
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(theme.accent)

                Text(worktree.currentBranch)
                    .font(theme.code(.small, weight: .semibold))
                    .foregroundStyle(.primary)

                if worktree.totalAdditions > 0 || worktree.totalDeletions > 0 {
                    HStack(spacing: 2) {
                        if worktree.totalAdditions > 0 {
                            Text("+\(worktree.totalAdditions)", bundle: .module)
                                .themedFont(.tiny, weight: .semibold).monospacedDigit()
                                .foregroundStyle(.green)
                        }
                        if worktree.totalDeletions > 0 {
                            Text("-\(worktree.totalDeletions)", bundle: .module)
                                .themedFont(.tiny, weight: .semibold).monospacedDigit()
                                .foregroundStyle(.red)
                        }
                    }
                }
            }
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(TurboSparkTheme.surfaceColor, in: Capsule())
            .overlay(Capsule().stroke(Color.primary.opacity(0.08), lineWidth: 0.5))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.97))
        .help("Git: \(worktree.currentBranch) (click to open changes pane)")
        .accessibilityLabel("Git branch \(worktree.currentBranch)")
    }
}

/// Memory, CPU and GPU wait, in the top bar's centre.
///
/// Polled on a 2-second timer -- the same cadence `StatusBarView` used for
/// these three -- so this bar never redraws per token. The readouts are the
/// same accessors, moved rather than restated: `currentProcessMemoryBytes`,
/// `currentProcessCPUUsage`, and `diagnostics?.phases?.gpuWaitMs`.
///
/// **GPU wait, not "GPU %".** There is no GPU-utilisation figure anywhere in
/// this app's telemetry, and inventing a percentage from `gpuWaitMs` would be
/// a number arrived at by ignorance (`swift/CLAUDE.md`'s rule about the
/// unsized model's "16 slots"). GPU wait is what the engine actually
/// measures, so GPU wait is what this shows -- and it shows an em dash until a
/// run has produced phase counters, rather than a zero.
@MainActor
struct ChromeTelemetryView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @State private var memoryBytes: UInt64?
    @State private var cpuPercent: Double?

    private let poll = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 11) {
            memoryCell
            divider
            cpuCell
            divider
            gpuCell
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5)
        }
        .onAppear(perform: refresh)
        .onReceive(poll) { _ in refresh() }
    }

    private var divider: some View {
        Rectangle()
            .fill(TurboSparkTheme.hairlineColor)
            .frame(width: 1, height: 13)
    }

    private var memoryCell: some View {
        HStack(spacing: 6) {
            Image(systemName: "memorychip")
                .font(theme.ui(points: 10))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            Text(MetricFormat.memory(memoryBytes))
                .font(theme.code(points: 11, weight: .medium))
                .monospacedDigit()
                .foregroundStyle(.primary)
            if let fraction = memoryFraction {
                TSMeterBar(fraction: fraction, tint: memoryTint(fraction))
                    .frame(width: 34)
            }
        }
        .help(memoryHelp)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Process memory")
        .accessibilityValue(memoryAccessibilityValue)
    }

    private var cpuCell: some View {
        HStack(spacing: 6) {
            Image(systemName: "cpu")
                .font(theme.ui(points: 10))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            Text(cpuText)
                .font(theme.code(points: 11, weight: .medium))
                .monospacedDigit()
                .foregroundStyle(.primary)
            Text("CPU", bundle: .module)
                .font(theme.ui(points: 11))
                .foregroundStyle(.tertiary)
        }
        .help("Current CPU utilization of the TurboSpark process")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Process CPU usage")
        .accessibilityValue(cpuText)
    }

    private var gpuCell: some View {
        HStack(spacing: 6) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(theme.ui(points: 10))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            Text(gpuWaitText)
                .font(theme.code(points: 11, weight: .medium))
                .monospacedDigit()
                .foregroundStyle(.primary)
            Text("GPU wait", bundle: .module)
                .font(theme.ui(points: 11))
                .foregroundStyle(.tertiary)
        }
        .help(
            "Metal GPU wait per pass from the last run's phase counters. "
                + "This app measures no GPU utilization figure, so none is shown."
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("GPU wait per pass")
        .accessibilityValue(gpuWaitText)
    }

    // MARK: - Values

    private var memoryFraction: Double? {
        guard let memoryBytes,
              let physical = model.telemetry?.physicalMemoryBytes,
              physical > 0
        else { return nil }
        return min(1.0, Double(memoryBytes) / Double(physical))
    }

    private func memoryTint(_ fraction: Double) -> Color {
        if fraction > 0.85 { return .red }
        if fraction > 0.65 { return .orange }
        return theme.accent
    }

    private var memoryHelp: String {
        var text = "Physical footprint of this process (phys_footprint)."
        if let physical = model.telemetry?.physicalMemoryBytes {
            text += " Machine has \(MetricFormat.memory(physical))."
        }
        return text
    }

    private var memoryAccessibilityValue: String {
        guard let fraction = memoryFraction else { return MetricFormat.memory(memoryBytes) }
        return "\(MetricFormat.memory(memoryBytes)), \(MetricFormat.percent(fraction * 100)) of system memory"
    }

    private var cpuText: String {
        guard let cpuPercent else { return "\u{2014}" }
        return MetricFormat.percent(cpuPercent)
    }

    /// An em dash until a run has produced phase counters. `calls > 0` is the
    /// same guard `RunnerDiagnosticsSection` uses before showing this figure.
    private var gpuWaitText: String {
        guard let phases = model.diagnostics?.phases, phases.calls > 0 else { return "\u{2014}" }
        return String(format: "%.1f ms", phases.gpuWaitMs)
    }

    private func refresh() {
        memoryBytes = model.currentProcessMemoryBytes
        cpuPercent = model.currentProcessCPUUsage
        model.refreshTelemetry()
    }
}

/// Two-tone horizontal meter.
///
/// Declared here rather than reused: `StatusBarView`'s `MeterBar` is
/// `private`, and widening it to share would couple the top bar to a file it
/// otherwise has nothing to do with. Six lines is cheaper than that coupling.
private struct TSMeterBar: View {
    let fraction: Double
    let tint: Color

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.1))
                Capsule()
                    .fill(tint)
                    .frame(width: max(2, geometry.size.width * fraction))
            }
        }
        .frame(height: 4)
        .accessibilityHidden(true)
    }
}


/// Compact prefill/decode indicator shown to the right of the model loader.
struct GenerationPhaseIndicator: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    /// App preference over system value; see `ToastOverlayView`.
    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        Group {
            if let statusText {
                HStack(spacing: 5) {
                    if model.opening || model.isInstallingModel {
                        ProgressView().controlSize(.mini).scaleEffect(0.7)
                    } else if model.phase == .prefill, !reduceMotion {
                        PulsingIndicatorDot()
                    } else {
                        Circle()
                            .fill(TurboSparkTheme.accentColor)
                            .frame(width: 6, height: 6)
                    }
                    Text(statusText)
                        .font(theme.ui(points: 11, weight: .medium))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(TurboSparkTheme.surfaceColor, in: Capsule())
                .help(statusText)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Model status")
                .accessibilityValue(statusText)
            }
        }
    }

    /// Nil while idle: a bar that shows "Ready" forever is chrome that never
    /// changes, and the status strip already carries the loaded/not signal.
    private var statusText: String? {
        if model.opening { return "Loading model" }
        if model.isInstallingModel { return model.installStageText ?? "Installing" }
        switch model.phase {
        case .prefill:
            return "Prefill \(model.livePrefillDone)/\(max(model.livePrefillTotal, 1))"
        case .decode:
            return "Generating"
        case .idle:
            return nil
        }
    }
}

private struct PulsingIndicatorDot: View {
    var body: some View {
        Circle()
            .fill(TurboSparkTheme.accentColor)
            .frame(width: 6, height: 6)
            .phaseAnimator([0.4, 1.0]) { dot, opacity in
                dot.opacity(opacity)
            } animation: { _ in
                .easeInOut(duration: 0.7)
            }
    }
}
