import SwiftUI

struct ServerConnectionView: View {
    @ObservedObject var model: AppModel
    @State private var port = ""
    @State private var portError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading) {
                    Text("IP address", bundle: .module).themedFont(.small, weight: .medium)
                    TextField("127.0.0.1", text: $model.serverHost)
                        .textFieldStyle(.roundedBorder)
                        .themedCode(.small)
                        .onChange(of: model.serverHost) { _, _ in model.persistSettingsDebounced() }
                }
                VStack(alignment: .leading) {
                    Text("Port", bundle: .module).themedFont(.small, weight: .medium)
                    TextField("auto", text: $port)
                        .textFieldStyle(.roundedBorder)
                        .themedCode(.small)
                        .frame(width: 100)
                        .onChange(of: port) { _, value in
                            switch ServerPortInput.parse(value) {
                            case .success(let port):
                                portError = nil
                                model.serverPortIsValid = true
                                model.serverPinnedPort = port
                                model.persistSettingsDebounced()
                            case .failure(let error):
                                portError = error.message
                                model.serverPortIsValid = false
                            }
                        }
                }
            }
            .disabled(model.server != nil || model.serverBusy)
            if let portError { Text(portError).foregroundStyle(.red).themedFont(.small) }
            Text("Stop the server to edit. Leave the port blank for automatic assignment.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
        }
        .onAppear { syncPort() }
        .onChange(of: model.serverPinnedPort) { _, _ in syncPort() }
    }

    private func syncPort() { port = model.serverPinnedPort == 0 ? "" : String(model.serverPinnedPort) }
}
