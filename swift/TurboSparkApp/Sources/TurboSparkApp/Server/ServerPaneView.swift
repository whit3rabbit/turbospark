import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45), and `apiTab` reads the model.
/// Keep operations and evidence visible before, during and after a server run.
@MainActor
struct ServerPaneView: View {
    @ObservedObject var model: AppModel
    @State private var tab = 0
    @AppStorage("TurboSpark.server.consoleHeight") private var consoleHeight = 300.0
    @State private var dragHeight: Double?

    /// Chosen in the panel beside the rail (`APISidebarPanel`), or the rail's
    /// API menu while that panel is hidden. The pane has no selector of its own.
    private var apiTab: APIWorkspaceTab { model.apiWorkspaceTab }

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        switch apiTab {
                        case .text:
                            ServerHeaderBandView(model: model)
                            ServerConnectionView(model: model)
                            ServerLoadedModelsView(model: model)
                            ServerLiveChartsView(model: model)
                            DisclosureGroup {
                                ServerChartsView(model: model).padding(.top, 12)
                            } label: { Text("Traffic", bundle: .module) }
                            DisclosureGroup {
                                ServerConnectCardView(model: model).padding(.top, 12)
                            } label: { Text("Connect an app", bundle: .module) }
                        case .image:
                            ImageAPIPaneView(model: model)
                        case .audio:
                            AudioAPIPaneView(model: model)
                        case .embeddings:
                            EmbeddingsAPIPaneView(model: model)
                        case .typeSafe:
                            TypeSafeAPIPaneView(model: model)
                        }
                        if apiTab != .text {
                            DisclosureGroup {
                                ServerChartsView(model: model).padding(.top, 12)
                            } label: { Text("Traffic", bundle: .module) }
                        }
                    }
                    .padding(20)
                }
                .frame(maxHeight: .infinity)
                Capsule().fill(.appBorder)
                    .frame(width: 44, height: 4)
                    .frame(maxWidth: .infinity)
                    .frame(height: 12)
                    .contentShape(Rectangle())
                    .gesture(DragGesture().onChanged { value in
                        if dragHeight == nil { dragHeight = consoleHeight }
                        consoleHeight = min(max((dragHeight ?? 300) - value.translation.height, 200), geometry.size.height * 0.65)
                    }.onEnded { _ in dragHeight = nil })
                    .accessibilityLabel(Text("Console", bundle: .module))
                    .accessibilityAdjustableAction { direction in
                        consoleHeight = min(max(consoleHeight + (direction == .increment ? 40 : -40), 200), geometry.size.height * 0.65)
                    }

                VStack(spacing: 0) {
                    Picker(selection: $tab) {
                        Text("Console", bundle: .module).tag(0)
                        Text("Live text", bundle: .module).tag(1)
                    } label: { Text("Activity", bundle: .module) }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 300)
                    .padding(10)
                    if tab == 0 {
                        ServerConsoleView(model: model)
                    } else {
                        ServerTextPreviewView(model: model)
                    }
                }
                .frame(height: min(max(consoleHeight, 200), geometry.size.height * 0.65))
            }
        }
        .background(.appPage)
        .onAppear {
            if model.server != nil { model.startServerPolling() }
        }
    }
}
