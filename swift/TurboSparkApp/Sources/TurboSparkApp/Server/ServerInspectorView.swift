import AppKit
import SwiftUI

struct ServerInspectorView: View {
    @ObservedObject var model: AppModel
    @State private var name = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Label { Text("Server", bundle: .module) } icon: { Image(systemName: "server.rack") }
                .themedFont(.callout, weight: .semibold)
                .padding(16)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    favorites
                    Divider()
                    Toggle(isOn: $model.serverCaptureText) {
                        Text("Capture text previews", bundle: .module)
                    }
                    .disabled(model.server != nil || model.serverBusy)
                    Text("Optional raw HTTP text. Kept in memory, truncated, cleared on restart.", bundle: .module)
                        .themedFont(.small).foregroundStyle(.appSecondary)
                    Button { copyDiagnostics() } label: {
                        Label { Text("Copy diagnostics", bundle: .module) } icon: { Image(systemName: "doc.on.doc") }
                    }
                    Divider()
                    Picker(selection: $model.maxContextTokens) {
                        ForEach(AppContextLengthOption.allCases) { option in
                            Text(option.menuLabel).tag(option.tokens)
                        }
                        if !AppContextLengthOption.allCases.contains(where: { $0.tokens == model.maxContextTokens }) {
                            Text(model.maxContextTokens.formatted()).tag(model.maxContextTokens)
                        }
                    } label: { Text("Context Window", bundle: .module) }
                    .disabled(model.serverBusy)
                    .onChange(of: model.maxContextTokens) { _, _ in model.persistSettingsDebounced() }
                    Text("Not applied to the loaded model", bundle: .module)
                        .themedFont(.small).foregroundStyle(.appSecondary)
                    ServerAdvancedSettingsView(model: model)
                }
                .padding(16)
            }
        }
        .background(.appPage)
    }

    private var favorites: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Profile", bundle: .module).themedFont(.base, weight: .semibold)
            TextField(text: $name) { Text("Profile name", bundle: .module) }
                .textFieldStyle(.roundedBorder)
            Button {
                model.saveServerFavorite(name: name)
                name = ""
            } label: { Text("Save Preset", bundle: .module) }
            .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            ForEach(model.serverFavorites) { favorite in
                VStack(alignment: .leading, spacing: 6) {
                    Text(favorite.name).themedFont(.small, weight: .medium)
                    Text(verbatim: "\(favorite.host):\(favorite.port) | \(favorite.modelPaths.count) models")
                        .themedCode(.small).foregroundStyle(.appSecondary)
                    HStack {
                        Button { model.loadServerFavorite(favorite) } label: { Text("Load", bundle: .module) }
                            .disabled(model.server != nil || model.serverBusy)
                        Spacer()
                        Button {
                            model.serverFavorites.removeAll { $0.id == favorite.id }
                            model.persistSettingsDebounced()
                        } label: { Text("Delete", bundle: .module) }
                    }
                }
                .padding(10)
                .background(.appSurface, in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private func copyDiagnostics() {
        let info = model.serverInfo
        let payload: [String: Any] = [
            "host": info?.host ?? model.serverHost,
            "port": info?.port ?? model.serverPinnedPort,
            "running": model.server != nil,
            "authEnabled": info?.authEnabled ?? false,
            "models": info?.models ?? [],
            "requests": model.serverMetrics.totalRequests,
            "errors": model.serverMetrics.totalErrors,
            "droppedEvents": model.serverMetrics.droppedEvents,
            "receivedBodyBytes": info?.traffic?.receivedBytes as Any? ?? NSNull(),
            "sentBodyBytes": info?.traffic?.sentBytes as Any? ?? NSNull(),
            "appAndServerMemoryBytes": model.serverLive.points.last?.memoryBytes as Any? ?? NSNull(),
            "physicalMemoryBytes": model.telemetry?.physicalMemoryBytes as Any? ?? NSNull(),
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]) else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(String(decoding: data, as: UTF8.self), forType: .string)
    }
}
