import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The window shell: icon rail, top bar, working panes, status strip.
///
/// The chrome is banded rather than floating. Sections live in the rail so the
/// chat sidebar can be hidden without stranding navigation, and every live
/// metric lives in the bottom strip so nothing next to the model loader
/// updates once per token.
@MainActor
struct RootView: View {
    @ObservedObject var model: AppModel
    @State private var conversationChromeHeight: CGFloat = 0
    @AppStorage("TurboSpark.chatSidebarVisible")
    private var isChatSidebarVisible = true
    @AppStorage("TurboSpark.inspectorVisible")
    private var isInspectorVisible = false
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    @State private var isChatSearchPresented = false

    private var effectiveReduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        HStack(spacing: 0) {
            NavigationRailView(model: model)
                .zIndex(10)

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(width: AppChromeLayout.dividerWidth)

            VStack(spacing: 0) {
                TopBarView(
                    model: model,
                    isChatSidebarVisible: isChatSidebarVisible,
                    isInspectorVisible: isInspectorVisible,
                    toggleChatSidebar: { isChatSidebarVisible.toggle() },
                    toggleInspector: { isInspectorVisible.toggle() })

                workingArea

                StatusBarView(model: model)
            }
        }
        .frame(
            minWidth: AppChromeLayout.minimumWindowWidth(
                isChatSidebarVisible: isChatSidebarVisible && showsChatSidebar,
                rightColumn: rightColumnClaimant),
            minHeight: AppChromeLayout.minimumHeight)
        .clipped()
        .background(Color(nsColor: .windowBackgroundColor))
        .appThemed()
        .animation(effectiveReduceMotion ? nil : .smooth(duration: 0.2), value: isChatSidebarVisible)
        .animation(effectiveReduceMotion ? nil : .smooth(duration: 0.2), value: isInspectorVisible)
        .animation(effectiveReduceMotion ? nil : .smooth(duration: 0.2), value: model.previewAttachmentID)
        .animation(effectiveReduceMotion ? nil : .smooth(duration: 0.2), value: model.openArtifactID)
        .animation(effectiveReduceMotion ? nil : .smooth(duration: 0.2), value: model.htmlPreviewID)
        .overlay(alignment: .top) {
            ToastOverlayView(model: model)
                .padding(.top, AppChromeLayout.topBarHeight + 10)
        }
        .overlay {
            // The Search Chats palette. Sits above every pane so Cmd+K
            // works from any section.
            if isChatSearchPresented {
                ChatSearchOverlayView(model: model, isPresented: $isChatSearchPresented)
            }
        }
        .overlay {
            // The unsloth-compatible alternate chords (Cmd+Shift+O, Cmd+B,
            // Ctrl+1..5). Mounted at the window root like the palette above
            // so they work from any section, including while the prompt
            // editor is first responder.
            AlternateShortcutBridge(model: model, toggleSidebar: { isChatSidebarVisible.toggle() })
        }
        .sheet(isPresented: Binding(
            get: { !model.pendingMcpApprovals.isEmpty },
            // Dismissing the sheet DEFERS rather than rejects: undecided
            // names are simply absent from the registries, so the next
            // project selection detects them again. Only an explicit
            // Reject records the decision.
            set: { presented in
                if !presented { model.pendingMcpApprovals.removeAll() }
            })) {
            ProjectMcpApprovalSheet(model: model)
        }
        .onReceive(NotificationCenter.default.publisher(for: .toggleChatSidebar)) { _ in
            isChatSidebarVisible.toggle()
        }
        .onReceive(NotificationCenter.default.publisher(for: .showChatSearch)) { _ in
            isChatSearchPresented.toggle()
        }
        .onReceive(NotificationCenter.default.publisher(for: .toggleInspector)) { _ in
            // A preview pane owns the right column while it is open, so the
            // same key has to be able to close it: otherwise the shortcut
            // silently toggles a pane the user cannot see. All three
            // preview claimants close; the inspector is only reached when
            // none of them does.
            if rightColumnClaimant.isPreviewPane {
                model.dismissArtifact()
                model.dismissHTMLPreview()
                model.dismissPreview()
            } else {
                isInspectorVisible.toggle()
            }
        }
        .onAppear {
            // The delegate cannot reach the `@StateObject`, and it is the one
            // quit hook that survives the window closing first.
            AppShutdownCoordinator.shared.onTerminate = { [weak model] in
                model?.shutdown()
            }
        }
        .onChange(of: model.isModelAvailable) { wasAvailable, isAvailable in
            // Surface Model Settings the moment a load completes, rather than
            // leaving a newly-loaded model's options a click away behind a
            // panel that starts hidden.
            if !wasAvailable, isAvailable {
                isInspectorVisible = true
            }
        }
    }

    /// The chat list is only meaningful beside a conversation.
    private var showsChatSidebar: Bool {
        model.activeSection == .chat
    }

    private var workingArea: some View {
        HStack(spacing: 0) {
            if isChatSidebarVisible && showsChatSidebar {
                ChatSidebarView(model: model)
                    .frame(width: AppChromeLayout.chatSidebarWidth)
                    .frame(maxHeight: .infinity)
                    .background(TurboSparkTheme.sidebarBackgroundColor)
                    .clipped()
                    .layoutPriority(1)
                    .zIndex(1)
                    .transition(effectiveReduceMotion ? .opacity : .move(edge: .leading).combined(with: .opacity))

                verticalHairline
            }

            primaryContent
                .frame(
                    minWidth: AppChromeLayout.primaryMinimumWidth,
                    maxWidth: .infinity,
                    maxHeight: .infinity)
                .clipped()
                .layoutPriority(0)

            rightColumn
        }
        .frame(maxHeight: .infinity)
    }

    /// Who owns the right column right now, resolved in ONE place.
    ///
    /// This used to be an `if/else if` chain reading two AppModel properties
    /// directly; the third claimant (`htmlPreview`) is what made the chain a
    /// decision worth naming (`AppRightColumnClaimant`).
    private var rightColumnClaimant: AppRightColumnClaimant {
        AppRightColumnClaimant.resolve(
            openArtifactID: model.openArtifactID,
            htmlPreviewID: model.htmlPreviewID,
            previewAttachmentID: model.previewAttachmentID,
            isInspectorVisible: isInspectorVisible)
    }

    @ViewBuilder
    private var rightColumn: some View {
        switch rightColumnClaimant {
        case .none:
            EmptyView()
        case .artifact(let id):
            verticalHairline
            rightPane(width: AppChromeLayout.artifactPanelWidth) {
                ArtifactPanelView(model: model, source: .artifact(id))
            }
        case .htmlPreview:
            verticalHairline
            rightPane(width: AppChromeLayout.artifactPanelWidth) {
                ArtifactPanelView(model: model, source: .inlinePreview)
            }
        case .filePreview:
            // The claimant is keyed on the id; the attachment lookup can
            // still miss (a detached draft). Missing falls through to the
            // inspector exactly as the old `if let` chain did.
            if let attachment = model.previewAttachment {
                verticalHairline
                rightPane(width: AppChromeLayout.inspectorWidth) {
                    FilePreviewView(model: model, attachment: attachment)
                }
            } else if isInspectorVisible {
                inspectorColumn
            }
        case .inspector:
            inspectorColumn
        }
    }

    /// Shared chrome for every right-column pane: hairline-adjacent, fixed
    /// width, trailing slide-in.
    private func rightPane<W: View>(
        width: CGFloat,
        @ViewBuilder content: () -> W
    ) -> some View {
        content()
            .frame(width: width)
            .frame(maxHeight: .infinity)
            .clipped()
            .layoutPriority(1)
            .zIndex(1)
            .transition(effectiveReduceMotion ? .opacity : .move(edge: .trailing).combined(with: .opacity))
    }

    private var inspectorColumn: some View {
        let isExpandedWorktree = (model.interactionMode == .projects && model.worktree?.isExpandedSplitMode == true)
        let currentWidth = AppChromeLayout.inspectorWidth(isExpanded: isExpandedWorktree)

        return Group {
            if model.interactionMode == .projects, let worktree = model.worktree {
                WorktreeView(model: model, worktree: worktree)
            } else {
                InspectorView(model: model)
            }
        }
        .frame(width: currentWidth)
        .frame(maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
        .clipped()
        .layoutPriority(1)
        .zIndex(1)
        .transition(effectiveReduceMotion ? .opacity : .move(edge: .trailing).combined(with: .opacity))
    }

    private var verticalHairline: some View {
        Rectangle()
            .fill(TurboSparkTheme.hairlineColor)
            .frame(width: AppChromeLayout.dividerWidth)
            .zIndex(1)
    }

    @ViewBuilder
    private var primaryContent: some View {
        switch model.activeSection {
        case .modelManager:
            ModelManagerView(model: model)
        case .modelHub:
            ModelHubView(model: model)
        case .server:
            ServerPaneView(model: model)
        case .files:
            FilesSectionView(model: model)
        case .chat:
            if model.requiresModelInstallation && !model.isInstallingModel {
                ModelInstallView(model: model)
            } else {
                conversationView
            }
        }
    }

    private var conversationView: some View {
        GeometryReader { _ in
            if model.hasOutputTranscript {
                ZStack(alignment: .bottom) {
                    OutputPaneView(model: model)
                        .padding(.bottom, conversationChromeHeight)

                    conversationChrome
                        .background {
                            GeometryReader { chromeGeometry in
                                Color.clear.preference(
                                    key: ConversationChromeHeightKey.self,
                                    value: chromeGeometry.size.height)
                            }
                        }
                }
                .onPreferenceChange(ConversationChromeHeightKey.self) { height in
                    guard height > 0 else { return }
                    var transaction = Transaction()
                    transaction.disablesAnimations = true
                    withTransaction(transaction) {
                        conversationChromeHeight = height
                    }
                }
            } else {
                OutputPaneView(model: model)
            }
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 8) {
            ErrorBanner(model: model)
            PromptComposerView(model: model)
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 12)
    }
}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
