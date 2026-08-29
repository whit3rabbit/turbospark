import AppKit
import SwiftUI

struct RootView: View {
    @ObservedObject var model: AppModel
    @State private var conversationChromeHeight: CGFloat = 0
    @State private var showingCatalogSheet = false
    @AppStorage("TurboSpark.chatSidebarVisible")
    private var isChatSidebarVisible = true
    @AppStorage("TurboSpark.inspectorVisible")
    private var isInspectorVisible = true
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        HStack(spacing: 0) {
            if isChatSidebarVisible {
                ChatSidebarView(model: model)
                    .frame(width: AppChromeLayout.chatSidebarWidth)
                    .frame(maxHeight: .infinity)
                    .background(TurboSparkTheme.sidebarBackgroundColor)
                    .clipped()
                    .layoutPriority(1)
                    .zIndex(1)
                    .transition(reduceMotion ? .opacity : .move(edge: .leading).combined(with: .opacity))

                Divider()
                    .zIndex(1)
            }

            primaryContent
                .frame(
                    minWidth: AppChromeLayout.primaryMinimumWidth,
                    maxWidth: .infinity,
                    maxHeight: .infinity)
                .clipped()
                .layoutPriority(0)

            if isInspectorVisible {
                Divider()
                    .zIndex(1)

                InspectorView(model: model)
                    .frame(width: AppChromeLayout.inspectorWidth)
                    .frame(maxHeight: .infinity)
                    .background(Color(nsColor: .windowBackgroundColor))
                    .clipped()
                    .layoutPriority(1)
                    .zIndex(1)
                    .transition(reduceMotion ? .opacity : .move(edge: .trailing).combined(with: .opacity))
            }
        }
        .frame(
            minWidth: AppChromeLayout.minimumWindowWidth(
                isChatSidebarVisible: isChatSidebarVisible,
                isInspectorVisible: isInspectorVisible),
            minHeight: AppChromeLayout.minimumHeight)
        .clipped()
        .background(
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(nsColor: .windowBackgroundColor).opacity(0.95),
                ],
                startPoint: .top,
                endPoint: .bottom)
        )
        .tint(TurboSparkTheme.accentColor)
        .animation(reduceMotion ? nil : .smooth(duration: 0.22), value: isChatSidebarVisible)
        .animation(reduceMotion ? nil : .smooth(duration: 0.22), value: isInspectorVisible)
        .sheet(isPresented: $showingCatalogSheet) {
            CatalogSheet(model: model)
        }
        .overlay(alignment: .top) {
            ToastOverlayView(model: model)
                .padding(.top, 60)
        }
        .onReceive(NotificationCenter.default.publisher(for: .toggleChatSidebar)) { _ in
            isChatSidebarVisible.toggle()
        }
        .onReceive(NotificationCenter.default.publisher(for: .toggleInspector)) { _ in
            isInspectorVisible.toggle()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
            model.unloadModel()
            model.persistChats()
            model.persistSettings()
        }
    }



    private var primaryContent: some View {
        Group {
            switch model.activeSection {
            case .modelHub:
                ModelHubView(model: model)
            case .chat:
                if model.requiresModelInstallation && !model.isInstallingModel {
                    ModelInstallView(model: model)
                } else {
                    conversationView
                }
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            StatusHUDView(
                model: model,
                isChatSidebarVisible: isChatSidebarVisible,
                isInspectorVisible: isInspectorVisible,
                toggleChatSidebar: { isChatSidebarVisible.toggle() },
                toggleInspector: { isInspectorVisible.toggle() })
        }
    }

    private var conversationView: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                if model.hasOutputTranscript {
                    OutputPaneView(model: model)
                        .padding(.bottom, conversationChromeHeight)
                } else if conversationChromeHeight > 0 {
                    OutputPaneView(model: model)
                        .frame(height: max(0, geometry.size.height - conversationChromeHeight))
                        .frame(maxHeight: .infinity, alignment: .top)
                } else {
                    OutputPaneView(model: model)
                }

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
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 10) {
            ErrorBanner(model: model)
            PromptComposerView(model: model)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
    }
}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
