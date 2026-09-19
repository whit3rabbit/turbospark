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
    // The storage KEY is unchanged so an existing preference carries over;
    // only the name is, because the flag stopped meaning "shown" when the
    // rail and the chat sidebar became one column (`AppSidebarView`). True is
    // still the roomy state either way.
    @AppStorage("TurboSpark.chatSidebarVisible")
    private var isSidebarExpanded = true
    @AppStorage("TurboSpark.inspectorVisible")
    private var isInspectorVisible = false
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    @State private var isChatSearchPresented = false
    @State private var isSummaryVisible = true
    @State private var workingWidth: CGFloat = 0
    @AppStorage("TurboSpark.imageModelRecommendationSeen")
    private var imageModelRecommendationSeen = false
    @State private var showingImageModelRecommendation = false

    private var canPinSummary: Bool { ProjectChatSummary.canPin(availableWidth: workingWidth) }
    private var hasProjectSummary: Bool {
        ProjectChatSummary.isAvailable(
            projectID: model.selectedChat.projectID, isChat: model.activeSection == .chat,
            hasTranscript: model.hasOutputTranscript)
    }

    private var effectiveReduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        HStack(spacing: 0) {
            AppSidebarView(model: model, isExpanded: isSidebarExpanded)
                .zIndex(10)

            Rectangle()
                .fill(.appBorder)
                .frame(width: AppChromeLayout.dividerWidth)

            VStack(spacing: 0) {
                TopBarView(
                    model: model,
                    isChatSidebarVisible: isSidebarExpanded,
                    isInspectorVisible: isInspectorVisible,
                    toggleChatSidebar: { isSidebarExpanded.toggle() },
                    toggleInspector: toggleModelSettings,
                    canPinSummary: canPinSummary,
                    isSummaryVisible: rightColumnClaimant == .projectSummary,
                    toggleSummary: {
                        if rightColumnClaimant == .projectSummary {
                            isSummaryVisible = false
                        } else {
                            isSummaryVisible = true
                            isInspectorVisible = false
                            model.dismissArtifact()
                            model.dismissHTMLPreview()
                            model.dismissPreview()
                        }
                    })

                workingArea

                StatusBarView(model: model)
            }
        }
        .frame(
            minWidth: AppChromeLayout.minimumWindowWidth(
                // A function of the toggle ALONE. This used to be
                // conjoined with `activeSection == .chat`, which is now
                // `AppSidebarView`'s business and was never this one's: the
                // column is present in every section, so anding the section in
                // here would let the window shrink under its own sidebar in
                // Files and Server.
                isSidebarExpanded: isSidebarExpanded,
                rightColumn: (model.activeSection == .images || rightColumnClaimant == .projectSummary) ? .none : rightColumnClaimant),
            minHeight: AppChromeLayout.minimumHeight)
        .clipped()
        .background(.appPage)
        .appThemed()
        .animation(effectiveReduceMotion ? nil : .smooth(duration: 0.2), value: isSidebarExpanded)
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
            AlternateShortcutBridge(model: model, toggleSidebar: { isSidebarExpanded.toggle() })
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
        .sheet(isPresented: $showingImageModelRecommendation) {
            ImageModelRecommendationSheet(model: model)
        }
        .onReceive(NotificationCenter.default.publisher(for: .toggleChatSidebar)) { _ in
            isSidebarExpanded.toggle()
        }
        .onReceive(NotificationCenter.default.publisher(for: .showChatSearch)) { _ in
            isChatSearchPresented.toggle()
        }
        .onReceive(NotificationCenter.default.publisher(for: .toggleInspector)) { _ in
            guard model.activeSection != .images else { return }
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
        .onAppear { [weak model] in
            // The delegate cannot reach the `@StateObject`, and it is the one
            // quit hook that survives the window closing first.
            AppShutdownCoordinator.shared.onTerminate = { [weak model] in
                model?.shutdown()
            }
            presentImageModelRecommendationIfNeeded()
        }
        .onChange(of: model.hasOutputTranscript, initial: true) { _, hasTranscript in
            if hasTranscript, model.activeSection == .chat { isInspectorVisible = false }
        }
        .onChange(of: model.selectedChatID) { _, _ in
            if model.hasOutputTranscript, model.activeSection == .chat { isInspectorVisible = false }
        }
        .onChange(of: model.isModelAvailable) { wasAvailable, isAvailable in
            // Loading during a conversation must not steal its reading space.
            if !wasAvailable, isAvailable, !model.hasOutputTranscript, model.activeSection != .images {
                isInspectorVisible = true
            }
        }
    }

    private func toggleModelSettings() {
        guard model.activeSection != .images else { return }
        // Preview panes have priority, so dismiss them before opening settings.
        if !isInspectorVisible || rightColumnClaimant.isPreviewPane {
            model.dismissArtifact()
            model.dismissHTMLPreview()
            model.dismissPreview()
            isInspectorVisible = true
        } else {
            isInspectorVisible = false
        }
    }

    private func presentImageModelRecommendationIfNeeded() {
        guard !imageModelRecommendationSeen,
            !model.hasInstalledZImageModel,
            !model.recommendedZImageSources.isEmpty
        else { return }
        imageModelRecommendationSeen = true
        showingImageModelRecommendation = true
    }

    private var workingArea: some View {
        HStack(spacing: 0) {
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
        .background {
            GeometryReader { geometry in
                Color.clear
                    .onAppear { workingWidth = geometry.size.width }
                    .onChange(of: geometry.size.width) { _, width in workingWidth = width }
            }
        }
    }

    /// Who owns the right column right now, resolved in ONE place.
    ///
    /// This used to be an `if/else if` chain reading two AppModel properties
    /// directly; the third claimant (`htmlPreview`) is what made the chain a
    /// decision worth naming (`AppRightColumnClaimant`).
    private var rightColumnClaimant: AppRightColumnClaimant {
        guard model.activeSection != .images else { return .none }
        return AppRightColumnClaimant.resolve(
            openArtifactID: model.openArtifactID,
            htmlPreviewID: model.htmlPreviewID,
            previewAttachmentID: model.previewAttachmentID,
            isInspectorVisible: isInspectorVisible,
            showProjectSummary: hasProjectSummary && canPinSummary && isSummaryVisible)
    }

    @ViewBuilder
    private var rightColumn: some View {
        if model.activeSection == .images {
            EmptyView()
        } else {
            switch rightColumnClaimant {
            case .none:
                EmptyView()
            case .projectSummary:
                rightPane(width: ProjectChatSummary.width) {
                    ProjectChatSummaryView(model: model).id(model.selectedChatID)
                }
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
                    verticalHairline
                    inspectorColumn
                }
            case .inspector:
                verticalHairline
                inspectorColumn
            }
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
            if model.activeSection == .server {
                ServerInspectorView(model: model)
            } else if model.interactionMode == .projects, let worktree = model.worktree {
                WorktreeView(model: model, worktree: worktree)
            } else {
                InspectorView(model: model)
            }
        }
        .frame(width: currentWidth)
        .frame(maxHeight: .infinity)
        .background(.appPage)
        .clipped()
        .layoutPriority(1)
        .zIndex(1)
        .transition(effectiveReduceMotion ? .opacity : .move(edge: .trailing).combined(with: .opacity))
    }

    private var verticalHairline: some View {
        Rectangle()
            .fill(.appBorder)
            .frame(width: AppChromeLayout.dividerWidth)
            .zIndex(1)
    }

    @ViewBuilder
    private var primaryContent: some View {
        switch model.activeSection {
        case .images:
            ImagesSectionView(model: model)
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
            } else if model.splitPaneChats.isEmpty {
                conversationView
            } else {
                // The qwen-code split view: secondary chats beside the main
                // conversation, each pane independently resizable.
                HSplitView {
                    conversationView
                        .frame(minWidth: 360)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    SplitChatPanesView(model: model)
                }
            }
        }
    }

    private var conversationView: some View {
        ConversationPaneView(model: model)
    }
}
