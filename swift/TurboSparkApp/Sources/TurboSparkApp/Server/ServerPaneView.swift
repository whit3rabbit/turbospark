import SwiftUI

/// The Server section: start a server, see what is talking to it, and point
/// a tool at it.
///
/// **PROGRESSIVE DISCLOSURE, ONE PANE, AND THE ORDER IS THE ARGUMENT.**
/// Closed, this is a start button, an address and one card explaining how to
/// connect -- which is the whole feature for somebody who wants their editor
/// to talk to a local model. The model table, the charts and the console
/// appear once a server is running, because before that they would all be
/// empty boxes. Auth, the pinned port and the raw endpoint list live behind
/// `Advanced`, which remembers whether it was open.
///
/// The alternative shapes were a Simple/Developer mode switch (a persisted
/// mode a user forgets they set, and two layouts to keep in step) and one
/// dense pane for everyone (fastest to build, and the thing that makes a
/// first open intimidating). This is neither.
struct ServerPaneView: View {
    @ObservedObject var model: AppModel

    @AppStorage("TurboSpark.server.showAdvanced")
    private var showAdvanced = false
    @AppStorage("TurboSpark.server.showConnect")
    private var showConnect = true
    @AppStorage("TurboSpark.server.consoleHeight")
    private var consoleHeight: Double = 220

    private var isRunning: Bool { model.server != nil }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    ServerHeaderBandView(model: model)

                    if isRunning {
                        ServerLoadedModelsView(model: model)
                        ServerChartsView(model: model)
                    }

                    DisclosureGroup(isExpanded: $showConnect) {
                        ServerConnectCardView(model: model)
                            .padding(.top, 10)
                    } label: {
                        sectionLabel("Connect an app", systemImage: "link")
                    }

                    DisclosureGroup(isExpanded: $showAdvanced) {
                        ServerAdvancedSettingsView(model: model)
                            .padding(.top, 10)
                    } label: {
                        sectionLabel("Advanced", systemImage: "slider.horizontal.3")
                    }
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            if isRunning {
                Divider()
                ServerConsoleView(model: model)
                    .frame(height: consoleHeight)
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        // The pane is not the only thing that can stop a server (unloading a
        // model from Chat can), so the timer follows the SERVER rather than
        // this view's lifetime. Appearing here only picks polling back up if
        // something started a server while this view was off screen.
        .onAppear {
            if isRunning { model.startServerPolling() }
        }
    }

    private func sectionLabel(_ title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .themedFont(.small, weight: .semibold)
            .foregroundStyle(.secondary)
    }
}
