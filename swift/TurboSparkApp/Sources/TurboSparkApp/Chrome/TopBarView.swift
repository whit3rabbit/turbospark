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
    @State private var confirmingEndGhost = false

    var body: some View {
        // The model loader sits beside the settings toggle it feeds (the
        // Model Settings panel), trailing-anchored together. The phase
        // indicator stays on the LEADING side rather than between them: it
        // appears and disappears every turn, and sitting between the loader
        // and a trailing-pinned toggle would make the loader itself slide
        // sideways each time it did.
        HStack(spacing: 8) {
            HStack(spacing: 8) {
                sidebarToggle
                GenerationPhaseIndicator(model: model)
                if let worktree = model.worktree, worktree.isGitRepository {
                    gitPill(worktree: worktree)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 8) {
                ModelLoaderControl(model: model)
                inspectorToggle
                ghostButton
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
        }
        // Overlaid rather than a third HStack member: the mode segment only
        // appears for the Chat section, and a THIRD flow element would
        // change how much width the flexible leading/trailing groups above
        // get on every section switch, sliding the sidebar toggle and the
        // model loader sideways -- the exact hazard the top bar's
        // left/right split exists to avoid.
        .overlay {
            if model.activeSection == .chat {
                PromptInteractionModeSegment(model: model)
            }
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

    /// Ghost Mode: start a temporary chat, or end the one being viewed.
    ///
    /// Occupies the top-right corner itself, trailing the inspector toggle.
    /// Ending a conversation that has content asks first, because the loss
    /// is the feature's whole point and an accidental click must not spend
    /// it.
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
        .buttonStyle(.plain)
        .foregroundStyle(model.isInGhostChat ? TurboSparkTheme.accentColor : Color.secondary)
        // `enterGhostChat()`/`endGhostChat()` both no-op under exactly this
        // guard, matching the File menu's "New Temporary Chat" (`.disabled(
        // model.isRunning)`); without it the button looks live during a turn
        // and silently does nothing when clicked.
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
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(TurboSparkTheme.accentColor)

                Text(worktree.currentBranch)
                    .font(theme.code(.small, weight: .semibold))
                    .foregroundStyle(.primary)

                if worktree.totalAdditions > 0 || worktree.totalDeletions > 0 {
                    HStack(spacing: 2) {
                        if worktree.totalAdditions > 0 {
                            Text("+\(worktree.totalAdditions)")
                                .font(.caption2.monospacedDigit().weight(.semibold))
                                .foregroundStyle(.green)
                        }
                        if worktree.totalDeletions > 0 {
                            Text("-\(worktree.totalDeletions)")
                                .font(.caption2.monospacedDigit().weight(.semibold))
                                .foregroundStyle(.red)
                        }
                    }
                }
            }
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(TurboSparkTheme.surfaceColor, in: Capsule())
            .overlay(
                Capsule()
                    .stroke(Color.primary.opacity(0.08), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .help("Git: \(worktree.currentBranch) (click to open changes pane)")
        .accessibilityLabel("Git branch \(worktree.currentBranch)")
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
