import SwiftUI

/// Flat top bar carrying the sidebar toggles, the model loader and the title.
///
/// This replaces the floating capsule HUD. Metrics moved to the bottom status
/// strip: a number that updates once per token pulls the eye, and next to the
/// model name it competes with the control the user actually reaches for.
struct TopBarView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let isChatSidebarVisible: Bool
    let isInspectorVisible: Bool
    let toggleChatSidebar: () -> Void
    let toggleInspector: () -> Void

    @ScaledMetric private var buttonSize: CGFloat = 26

    var body: some View {
        // The model loader sits beside the settings toggle it feeds (the
        // Model Settings panel), trailing-anchored together. The phase
        // indicator stays on the LEADING side rather than between them: it
        // appears and disappears every turn, and sitting between the loader
        // and a trailing-pinned toggle would make the loader itself slide
        // sideways each time it did.
        HStack(spacing: 8) {
            sidebarToggle
            GenerationPhaseIndicator(model: model)
            Spacer(minLength: 8)
            ModelLoaderControl(model: model)
            inspectorToggle
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
        .buttonStyle(.plain)
        .foregroundStyle(isChatSidebarVisible ? Color.primary : Color.secondary)
        // No `.keyboardShortcut` here (U7): the "View" menu's "Toggle Chat
        // Sidebar" command already claims Cmd+Ctrl+S, and SwiftUI does not
        // define which of two identical shortcuts on different controls
        // wins -- a second declaration here is at best redundant and at
        // worst a dead or double-firing shortcut. The menu is the one
        // source of truth for this key.
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
        .buttonStyle(.plain)
        .foregroundStyle(isInspectorVisible ? Color.primary : Color.secondary)
        // No `.keyboardShortcut` here (U7): the "View" menu's "Toggle
        // Inspector" command claims the same Cmd+Shift+I and, unlike this
        // button's plain `toggleInspector` closure, its handler
        // (RootView's `.toggleInspector` notification receiver) also
        // dismisses an open attachment preview first. Two controls
        // registering the identical shortcut with DIFFERENT behavior is
        // the double-toggle/dead-shortcut bug; the menu is the one that
        // does the right thing, so it is the one that keeps the key.
        .help(presentation.help)
        .accessibilityLabel(presentation.title)
        .accessibilityHint(presentation.help)
        .accessibilityValue(presentation.accessibilityValue)
    }
}

/// Compact prefill/decode indicator shown to the right of the model loader.
struct GenerationPhaseIndicator: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

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
