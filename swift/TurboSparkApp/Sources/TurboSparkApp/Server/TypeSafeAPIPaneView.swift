import Foundation
import SwiftUI

@MainActor
struct TypeSafeAPIPaneView: View {
    @ObservedObject var model: AppModel
    @State private var copiedKey = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 12) {
                Text("TypeSafe API", bundle: .module).themedFont(.title3, weight: .semibold)
                Text("OpenKind runs as a separate loopback service with its own API key.", bundle: .module)
                    .foregroundStyle(.secondary)
                HStack {
                    Text(verbatim: "127.0.0.1:")
                    TextField("Port", text: $model.typeSafePort)
                        .frame(width: 80)
                        .disabled(model.typeSafeServer != nil)
                        .accessibilityLabel(Text("Port", bundle: .module))
                    if let url = model.typeSafeBaseURL, model.typeSafeServer != nil {
                        Text(url.absoluteString).textSelection(.enabled)
                    }
                }
                HStack(spacing: 6) {
                    SecureField("TypeSafe API key", text: $model.typeSafeAPIKeyInput)
                        .disabled(model.typeSafeServer != nil)
                        .accessibilityLabel(Text("TypeSafe API key", bundle: .module))

                    Button {
                        model.typeSafeAPIKeyInput = TypeSafeKeychain.generateKey()
                    } label: {
                        Image(systemName: "wand.and.stars")
                    }
                    .buttonStyle(.borderless)
                    .disabled(model.typeSafeServer != nil)
                    .help("Generate a random key")
                    .accessibilityLabel("Generate a random API key")

                    Button {
                        let key = model.typeSafeAPIKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !key.isEmpty else { return }
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(key, forType: .string)
                        copiedKey = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                            copiedKey = false
                        }
                    } label: {
                        Image(systemName: copiedKey ? "checkmark" : "doc.on.doc")
                    }
                    .buttonStyle(.borderless)
                    .help("Copy API key to clipboard")
                    .accessibilityLabel("Copy API key to clipboard")
                }
                HStack {
                    Button {
                        Task {
                            if model.typeSafeServer == nil { await model.startTypeSafeServer() }
                            else { await model.stopTypeSafeServer() }
                        }
                    } label: {
                        if model.typeSafeServer == nil { Text("Start", bundle: .module) }
                        else { Text("Stop", bundle: .module) }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(
                        model.typeSafeBusy
                        || (model.typeSafeServer == nil
                            && (model.typeSafeBaseURL == nil
                                || model.typeSafeAPIKeyInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))
                    )
                    Button {
                        Task { await model.refreshTypeSafeModels() }
                    } label: { Text("Refresh", bundle: .module) }
                    .disabled(model.typeSafeServer == nil || model.typeSafeBusy)
                    if model.typeSafeBusy { ProgressView().controlSize(.small) }
                    Text(verbatim: "Health: \(model.typeSafeHealth)")
                }
                if let error = model.typeSafeError { Text(error).foregroundStyle(.red).textSelection(.enabled) }
            }
            .padding(16)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))

            TypeSafePlaygroundView(model: model)

            VStack(alignment: .leading, spacing: 10) {
                Text("Models", bundle: .module).themedFont(.title3, weight: .semibold)
                if model.typeSafeModels.isEmpty {
                    Text("No local model status available.", bundle: .module).foregroundStyle(.secondary)
                }
                ForEach(model.typeSafeModels) { item in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(item.name)
                            Text(verbatim: "\(item.source), \(item.loaded ? "loaded" : "unloaded")")
                                .themedFont(.small).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if item.manageable {
                            Button {
                                Task { await model.setTypeSafeModel(item.name, loaded: !item.loaded) }
                            } label: {
                                if item.loaded { Text("Unload", bundle: .module) }
                                else { Text("Load", bundle: .module) }
                            }
                            .disabled(model.typeSafeBusy)
                        }
                    }
                }
            }
            .padding(16)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))

        }
        .task(id: model.typeSafeServer != nil) {
            while !Task.isCancelled && model.typeSafeServer != nil {
                await model.refreshTypeSafeModels()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

}
