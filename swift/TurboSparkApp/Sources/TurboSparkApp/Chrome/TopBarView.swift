import SwiftUI
import Combine

/// Flat top bar: sidebar toggles, live machine load.
///
/// **The model loader is no longer here.** It moved under the prompt
/// composer, where "which model, is it loaded" reads as the chat's own
/// footer rather than as window chrome. This band now answers only what
/// the machine is doing. The trailing group keeps its
/// `frame(maxWidth: .infinity)` even though one control is left in it:
/// that frame is half of what holds the fixed-size centre still.
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
    /// False in rail-only sections: there is no panel to show or hide, and
    /// a toggle that does nothing there is a control the user cannot read.
    var showsSidebarToggle = true
    let toggleInspector: () -> Void
    var canPinSummary = false
    var isSummaryVisible = false
    var toggleSummary: () -> Void = {}
    @State private var showsSummaryPopover = false
    @State private var barWidth: CGFloat = 0

    @ScaledMetric private var buttonSize: CGFloat = 26

    var body: some View {
        HStack(spacing: 8) {
            HStack(spacing: 8) {
                if showsSidebarToggle {
                    sidebarToggle
                }
                AudioRecordingBadge(controller: model.audioWorkspace, openWorkspace: { model.activeSection = .audio })
                GenerationPhaseIndicator(model: model)
                if let worktree = model.worktree, worktree.isGitRepository {
                    gitPill(worktree: worktree)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Centre. Fixed-size, so the flexible groups either side keep
            // their own widths and the model loader does not slide when a
            // phase indicator appears.
            if barWidth >= 760 {
                ChromeTelemetryView(model: model).fixedSize()
            }

            HStack(spacing: 8) {
                if model.activeSection == .chat {
                    ChatShareButton(model: model)
                    if ProjectChatSummary.isAvailable(
                        projectID: model.selectedChat.projectID, isChat: true,
                        hasTranscript: model.hasOutputTranscript) {
                        Button {
                            if canPinSummary { toggleSummary() }
                            else { showsSummaryPopover.toggle() }
                        } label: {
                            Image(systemName: "list.bullet.rectangle")
                                .font(theme.ui(.callout))
                                .foregroundStyle(canPinSummary && isSummaryVisible ? theme.accent : theme.secondaryText)
                        }
                        .buttonStyle(.plain)
                        .help(Text("Chat summary", bundle: .module))
                        .accessibilityLabel(Text("Chat summary", bundle: .module))
                        .popover(isPresented: $showsSummaryPopover) {
                            ProjectChatSummaryView(model: model)
                                .id(model.selectedChatID)
                                .frame(width: ProjectChatSummary.width, height: 480)
                                .appThemed()
                        }
                        .onChange(of: model.openArtifactID) { _, value in
                            if value != nil { showsSummaryPopover = false }
                        }
                        .onChange(of: canPinSummary) { _, value in
                            if value { showsSummaryPopover = false }
                        }
                    }
                }
                unloadButton
                if model.browserSettings.enabled && !model.activeSection.hidesChatPanes {
                    browserToggle
                }
                if !model.activeSection.hidesChatPanes {
                    inspectorToggle
                }
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding(.leading, AppChromeLayout.trafficLightClearance)
        .padding(.trailing, 10)
        .frame(height: AppChromeLayout.topBarHeight)
        .background(.appSurface)
        .background {
            GeometryReader { geometry in
                Color.clear
                    .onAppear { barWidth = geometry.size.width }
                    .onChange(of: geometry.size.width) { _, width in barWidth = width }
            }
        }
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(.appBorder)
                .frame(height: 0.5)
        }
    }

    private var sidebarToggle: some View {
        let presentation = AppSidebarControlPresentation(
            sidebar: .chats,
            isVisible: isChatSidebarVisible)
        return Button(action: toggleChatSidebar) {
            Image(systemName: presentation.systemImage)
                .font(theme.ui(.callout, weight: .medium))
                .frame(width: buttonSize, height: buttonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.9))
        .foregroundStyle(isChatSidebarVisible ? theme.foreground : theme.secondaryText)
        // No `.keyboardShortcut` here (U7): the "View" menu's "Toggle
        // Sidebar" command already claims Cmd+Ctrl+S, and SwiftUI does not
        // define which of two identical shortcuts on different controls wins.
        .help(helpText(presentation))
        .accessibilityLabel(titleText(presentation))
        .accessibilityHint(helpText(presentation))
        .accessibilityValue(Text(LocalizedStringKey(presentation.accessibilityValue), bundle: .module))
    }

    private var inspectorToggle: some View {
        let presentation = AppSidebarControlPresentation(
            sidebar: .inspector,
            isVisible: isInspectorVisible)
        return Button(action: toggleInspector) {
            Image(systemName: presentation.systemImage)
                .font(theme.ui(.callout, weight: .medium))
                .frame(width: buttonSize, height: buttonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.9))
        .foregroundStyle(isInspectorVisible ? theme.accent : theme.secondaryText)
        .background {
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(isInspectorVisible ? theme.accent.opacity(0.14) : .clear)
        }
        .tsAnimation(TSMotion.hover, value: isInspectorVisible)
        // No `.keyboardShortcut` here (U7): the "View" menu's "Toggle
        // Inspector" command claims Cmd+Shift+I and its handler also
        // dismisses an open attachment preview first.
        .help(helpText(presentation))
        .accessibilityLabel(titleText(presentation))
        .accessibilityHint(helpText(presentation))
        .accessibilityValue(Text(LocalizedStringKey(presentation.accessibilityValue), bundle: .module))
    }

    /// `AppSidebarControlPresentation` carries English `String`s, which the
    /// `String` overloads of `.help` and `.accessibilityLabel` would show
    /// untranslated. Its title and value are catalog keys; its help is the
    /// title plus a shortcut suffix, which has nothing to translate.
    private func titleText(_ presentation: AppSidebarControlPresentation) -> Text {
        Text(LocalizedStringKey(presentation.title), bundle: .module)
    }

    private func helpText(_ presentation: AppSidebarControlPresentation) -> Text {
        guard presentation.help.hasPrefix(presentation.title) else {
            return Text(LocalizedStringKey(presentation.help), bundle: .module)
        }
        return titleText(presentation)
            + Text(verbatim: String(presentation.help.dropFirst(presentation.title.count)))
    }

    private var browserToggle: some View {
        Button {
            model.setBrowserPaneOpen(!model.browserPaneIsOpen)
        } label: {
            Image(systemName: "globe")
                .font(theme.ui(.callout, weight: .medium))
                .frame(width: buttonSize, height: buttonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.9))
        .foregroundStyle(model.browserPaneIsOpen ? theme.accent : theme.secondaryText)
        .background {
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(model.browserPaneIsOpen ? theme.accent.opacity(0.14) : .clear)
        }
        .tsAnimation(TSMotion.hover, value: model.browserPaneIsOpen)
        .help(Text("Browser", bundle: .module))
        .accessibilityLabel(Text("Browser", bundle: .module))
        .accessibilityAddTraits(model.browserPaneIsOpen ? .isSelected : [])
    }

    private func gitPill(worktree: WorktreeModel) -> some View {
        Button {
            NotificationCenter.default.post(name: .toggleInspector, object: nil)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "point.topleft.down.to.point.bottomright.curvepath")
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(theme.accent)

                Text(verbatim: worktree.currentBranch)
                    .font(theme.code(.small, weight: .semibold))
                    .foregroundStyle(.appText)

                if worktree.totalAdditions > 0 || worktree.totalDeletions > 0 {
                    HStack(spacing: 2) {
                        if worktree.totalAdditions > 0 {
                            Text(verbatim: "+\(worktree.totalAdditions)")
                                .themedFont(.tiny, weight: .semibold).monospacedDigit()
                                .foregroundStyle(.green)
                        }
                        if worktree.totalDeletions > 0 {
                            Text(verbatim: "-\(worktree.totalDeletions)")
                                .themedFont(.tiny, weight: .semibold).monospacedDigit()
                                .foregroundStyle(.red)
                        }
                    }
                }
            }
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(TurboSparkTheme.surfaceColor, in: Capsule())
            .overlay(Capsule().stroke(.appText.opacity(0.08), lineWidth: 0.5))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.97))
        .help(Text("Git: \(worktree.currentBranch) (click to open changes pane)", bundle: .module))
        .accessibilityLabel(Text("Git branch \(worktree.currentBranch)", bundle: .module))
    }

    @ViewBuilder
    private var unloadButton: some View {
        let canUnloadText = model.canUnloadModel
        let canUnloadImage = model.canUnloadImageModel
        let canUnloadAudio = model.audioWorkspace.canUnloadModel
        let count = (canUnloadText ? 1 : 0) + (canUnloadImage ? 1 : 0) + (canUnloadAudio ? 1 : 0)
        if count > 1 {
            Menu {
                if canUnloadText {
                    Button {
                        model.unloadModel()
                    } label: {
                        Label {
                            Text("Unload Model", bundle: .module)
                        } icon: {
                            Image(systemName: "text.bubble")
                        }
                    }
                }
                if canUnloadImage {
                    Button {
                        model.unloadImageModel()
                    } label: {
                        Label {
                            Text("Unload", bundle: .module)
                        } icon: {
                            Image(systemName: "photo")
                        }
                    }
                }
                if canUnloadAudio {
                    Button {
                        model.audioWorkspace.unloadModel()
                    } label: {
                        Label {
                            Text("Unload", bundle: .module)
                        } icon: {
                            Image(systemName: "waveform")
                        }
                    }
                }
            } label: {
                Image(systemName: "eject.fill")
                    .font(theme.ui(.callout, weight: .medium))
                    .frame(width: buttonSize, height: buttonSize)
                    .contentShape(Rectangle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .buttonStyle(TSPressScaleStyle(scale: 0.9))
            .foregroundStyle(theme.accent)
            .help(Text("Unload Model", bundle: .module))
            .accessibilityLabel(Text("Unload Model", bundle: .module))
        } else {
            Button {
                model.unloadActiveOrAnyModel()
            } label: {
                Image(systemName: "eject.fill")
                    .font(theme.ui(.callout, weight: .medium))
                    .frame(width: buttonSize, height: buttonSize)
                    .contentShape(Rectangle())
            }
            .buttonStyle(TSPressScaleStyle(scale: 0.9))
            .foregroundStyle(model.canUnloadAnyModel ? theme.accent : theme.secondaryText.opacity(0.35))
            .disabled(!model.canUnloadAnyModel)
            .help(Text("Unload Model", bundle: .module))
            .accessibilityLabel(Text("Unload Model", bundle: .module))
        }
    }
}
