import SwiftUI

struct ServerMenuDashboardView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Circle().fill(model.server == nil ? Color.secondary : ((model.serverInfo?.models.isEmpty ?? true) ? Color.orange : Color.green))
                    .frame(width: 8, height: 8)
                Text("Server", bundle: .module).themedFont(.base, weight: .semibold)
                Spacer()
                Menu {
                    ServerMenuBarView(model: model)
                } label: {
                    Label { Text("More", bundle: .module) } icon: { Image(systemName: "ellipsis.circle") }
                        .labelStyle(.iconOnly)
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help(Text("More", bundle: .module))
            }
            if let info = model.serverInfo {
                Text(verbatim: info.baseURL?.absoluteString ?? "\(info.host):\(info.port)")
                    .themedCode(.small).textSelection(.enabled)
                ServerLiveChartsView(model: model, compact: true)
            }
            HStack {
                // Same roles as the API pane's status band: Start is the
                // primary action, Stop is secondary.
                if model.server == nil {
                    Button { model.startServer() } label: { Text("Start Server", bundle: .module) }
                        .appAction(.primary, size: APIControlSize.panel)
                        .disabled(model.serverBusy)
                } else {
                    Button { model.stopServer() } label: { Text("Stop Server", bundle: .module) }
                        .appAction(.secondary, size: APIControlSize.panel)
                        .disabled(model.serverBusy)
                }
                Spacer()
                Button { model.showMainWindow(navigatingTo: .server) } label: {
                    Text("Admin Panel", bundle: .module)
                }
                .appAction(.secondary, size: APIControlSize.panel)
            }
        }
        .padding(16)
        .frame(width: 380)
        .background(.appPage)
    }
}
