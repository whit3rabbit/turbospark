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
                } label: { Image(systemName: "ellipsis.circle") }
                .menuStyle(.borderlessButton)
                .fixedSize()
            }
            if let info = model.serverInfo {
                Text(info.baseURL?.absoluteString ?? "\(info.host):\(info.port)")
                    .themedCode(.small).textSelection(.enabled)
                ServerLiveChartsView(model: model, compact: true)
            }
            HStack {
                Button {
                    model.server == nil ? model.startServer() : model.stopServer()
                } label: {
                    if model.server == nil { Text("Start Server", bundle: .module) }
                    else { Text("Stop Server", bundle: .module) }
                }
                .disabled(model.serverBusy)
                Spacer()
                Button { model.showMainWindow(navigatingTo: .server) } label: {
                    Text("Admin Panel", bundle: .module)
                }
            }
        }
        .padding(16)
        .frame(width: 380)
        .background(.appPage)
    }
}
