import Foundation
import OpenKind
import SwiftUI

/// The TypeSafe mode: the same band, models panel and test layout as the
/// other API modes, but for OpenKind, which is a SEPARATE loopback daemon
/// with its own port and its own Keychain key. The band says so in its own
/// words so nobody mistakes this Start for the TurboSpark server's.
@MainActor
struct TypeSafeAPIPaneView: View {
    @ObservedObject var model: AppModel

    private var isRunning: Bool { model.typeSafeServer != nil }

    private var trimmedKey: String {
        model.typeSafeAPIKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            band
            modelsPanel
            TypeSafePlaygroundView(model: model)
        }
        .task(id: model.typeSafeServer != nil) {
            while !Task.isCancelled && model.typeSafeServer != nil {
                await model.refreshTypeSafeModels()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    private var bandColor: Color {
        guard isRunning else { return .secondary }
        return model.typeSafeError == nil ? .green : .orange
    }

    private var band: some View {
        APIStatusBand(
            color: bandColor,
            title: isRunning
                ? Text("TypeSafe service running", bundle: .module)
                : Text("TypeSafe service stopped", bundle: .module),
            subtitle: Text("OpenKind runs as a separate loopback service with its own API key.", bundle: .module),
            isRunning: isRunning,
            busy: model.typeSafeBusy,
            startLabel: Text("Start TypeSafe", bundle: .module),
            stopLabel: Text("Stop TypeSafe", bundle: .module),
            startDisabled: model.typeSafeBusy || model.typeSafeBaseURL == nil || trimmedKey.isEmpty,
            stopDisabled: model.typeSafeBusy,
            start: { Task { await model.startTypeSafeServer() } },
            stop: { Task { await model.stopTypeSafeServer() } }
        ) {
            HStack(spacing: 6) {
                Text(verbatim: "127.0.0.1:")
                    .themedCode(.small)
                TextField(text: $model.typeSafePort, prompt: Text("Port", bundle: .module)) {
                    Text("Port", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 70, idealWidth: 80, maxWidth: 100)
                .disabled(isRunning)
                if let url = model.typeSafeBaseURL, isRunning {
                    Text(verbatim: url.absoluteString)
                        .themedCode(.small)
                        .textSelection(.enabled)
                    APICopyButton(label: Text("Copy the address", bundle: .module)) { url.absoluteString }
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    SecureField(text: $model.typeSafeAPIKeyInput, prompt: Text("TypeSafe API key", bundle: .module)) {
                        Text("TypeSafe API key", bundle: .module)
                    }
                    .textFieldStyle(.roundedBorder)
                    .disabled(isRunning)

                    AppIconButton("wand.and.stars", label: Text("Generate a random API key", bundle: .module)) {
                        model.typeSafeAPIKeyInput = TypeSafeKeychain.generateKey()
                    }
                    .buttonStyle(.borderless)
                    .disabled(isRunning)

                    APICopyButton(label: Text("Copy API key to clipboard", bundle: .module)) {
                        trimmedKey.isEmpty ? nil : trimmedKey
                    }
                }
                // Opt-in: nothing is written to the Keychain, and macOS is not
                // asked for access, until this is on.
                Toggle(isOn: Binding(
                    get: { model.typeSafeKeyInKeychain },
                    set: { model.setTypeSafeKeyInKeychain($0) })
                ) {
                    Text("Remember this key in the Keychain", bundle: .module)
                }
                .toggleStyle(.checkbox)
                .disabled(isRunning)
                // Two servers, two keys: saying where this one lives stops a
                // person pasting the TurboSpark key here, or this one there.
                if model.typeSafeKeyInKeychain {
                    Text("Stored in the Keychain separately from the TurboSpark server key.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                } else {
                    Text("A new key is generated each time the app starts and is not saved.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }

            APIStatusLine(label: Text("Health", bundle: .module), value: APIStatusText.text(model.typeSafeHealth))
            if let error = model.typeSafeError {
                Text(verbatim: error).foregroundStyle(.red).textSelection(.enabled)
            }
        }
    }

    private var modelsPanel: some View {
        APIPanel(title: Text("Models", bundle: .module)) {
            Button {
                Task { await model.refreshTypeSafeModels() }
            } label: {
                Label { Text("Refresh", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
            }
            .appAction(.secondary, size: APIControlSize.panel)
            .disabled(!isRunning || model.typeSafeBusy)
        } content: {
            if model.typeSafeModels.isEmpty {
                Text("No local model status available.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            } else {
                APIRowList {
                    ForEach(model.typeSafeModels) { item in
                        modelRow(item)
                        if item.id != model.typeSafeModels.last?.id { Divider() }
                    }
                }
            }
        }
    }

    private func modelRow(_ item: LocalModel) -> some View {
        HStack(spacing: 12) {
            Circle().fill(item.loaded ? Color.green : Color.secondary).frame(width: 7, height: 7)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: item.name).themedCode(.base).textSelection(.enabled)
                (Text(verbatim: item.source + ", ")
                    + (item.loaded ? Text("Loaded", bundle: .module) : Text("Not loaded", bundle: .module)))
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            Spacer(minLength: 8)
            if item.manageable {
                if item.loaded {
                    Button {
                        Task { await model.setTypeSafeModel(item.name, loaded: false) }
                    } label: { Text("Unload", bundle: .module) }
                    .appAction(.secondary, size: APIControlSize.row)
                    .disabled(model.typeSafeBusy)
                } else {
                    Button {
                        Task { await model.setTypeSafeModel(item.name, loaded: true) }
                    } label: { Text("Load", bundle: .module) }
                    .appAction(.primary, size: APIControlSize.row)
                    .disabled(model.typeSafeBusy)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
    }
}
