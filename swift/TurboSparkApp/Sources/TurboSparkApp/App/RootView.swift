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
                isInspectorVisible: isInspectorVisible || model.previewAttachment != nil),
            minHeight: AppChromeLayout.minimumHeight)
        .clipped()
        .background(Color(nsColor: .windowBackgroundColor))
        .appThemed()
        .animation(effectiveReduceMotion ? nil : .smooth(duration: 0.2), value: isChatSidebarVisible)
        .animation(effectiveReduceMotion ? nil : .smooth(duration: 0.2), value: isInspectorVisible)
        .animation(effectiveReduceMotion ? nil : .smooth(duration: 0.2), value: model.previewAttachmentID)
        .overlay(alignment: .top) {
            ToastOverlayView(model: model)
                .padding(.top, AppChromeLayout.topBarHeight + 10)
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
        .onReceive(NotificationCenter.default.publisher(for: .toggleInspector)) { _ in
            // The preview owns the right column while it is open, so the same
            // key has to be able to close it: otherwise the shortcut silently
            // toggles a pane the user cannot see.
            if model.previewAttachment != nil {
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

    @ViewBuilder
    private var rightColumn: some View {
        if let attachment = model.previewAttachment {
            verticalHairline

            FilePreviewView(model: model, attachment: attachment)
                .frame(width: AppChromeLayout.inspectorWidth)
                .frame(maxHeight: .infinity)
                .clipped()
                .layoutPriority(1)
                .zIndex(1)
                .transition(effectiveReduceMotion ? .opacity : .move(edge: .trailing).combined(with: .opacity))
        } else if isInspectorVisible {
            verticalHairline

            let isExpandedWorktree = (model.interactionMode == .projects && model.worktree?.isExpandedSplitMode == true)
            let currentWidth = AppChromeLayout.inspectorWidth(isExpanded: isExpandedWorktree)

            Group {
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
