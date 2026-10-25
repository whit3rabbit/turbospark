import SwiftUI
import TurboSpark

/// The API Embeddings tab: choose and attach an embedding model to the
/// shared TurboSpark listener, see what is served, and send a test request.
///
/// The model chosen here is the server's one embedding setting: it is
/// persisted and attached again after every start. Attach runs off the main
/// thread because a first Core ML load can take tens of seconds.
@MainActor
struct EmbeddingsAPIPaneView: View {
    @ObservedObject var model: AppModel

    private var trimmedSelection: String {
        model.serverEmbeddingModelInput.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var block: EmbeddingAttachBlock? {
        AppModel.embeddingAttachBlock(
            serverRunning: model.server != nil,
            selection: model.serverEmbeddingModelInput,
            probe: model.embeddingProbes[trimmedSelection],
            busy: model.serverBusy,
            attached: model.serverEmbeddingModelIDs)
    }

    private var exampleBody: String {
        let value: [String: Any] = [
            "model": model.serverEmbeddingModelIDs.first ?? "<attach an embedding model>",
            "input": ["The quick brown fox"]
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        else { return "" }
        return String(decoding: data, as: UTF8.self)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            ServerHeaderBandView(model: model)
            servingPanel
            EmbeddingsAPITestPanel(model: model)
        }
    }

    private var servingPanel: some View {
        APIPanel(
            title: Text("Embeddings API", bundle: .module),
            subtitle: Text("The same address and API key as Text. The model chosen here is attached again each time the server starts.", bundle: .module)
        ) {
            Button {
                model.attachEmbeddingModelToServer(model.serverEmbeddingModelInput)
            } label: { Text("Attach", bundle: .module) }
            .appAction(.primary, size: APIControlSize.panel)
            .disabled(block != nil)
            .help(blockReason ?? Text("Attach", bundle: .module))
        } content: {
            EmbeddingModelPicker(
                model: model, selection: $model.serverEmbeddingModelInput,
                disabled: model.serverEmbeddingAttachInFlight)
                .onChange(of: model.serverEmbeddingModelInput) { _, _ in
                    model.persistSettingsDebounced()
                }
            if let blockReason, !model.serverEmbeddingAttachInFlight {
                blockReason
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if model.serverEmbeddingAttachInFlight {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                .accessibilityElement(children: .combine)
            }
            APIRowList {
                if model.serverEmbeddingModelIDs.isEmpty {
                    APINothingServedRow()
                } else {
                    ForEach(model.serverEmbeddingModelIDs, id: \.self) { id in
                        APIServedModelRow(
                            name: id,
                            detachHelp: Text("Stop serving this and release it from memory.", bundle: .module),
                            detachDisabled: model.serverEmbeddingAttachInFlight,
                            detach: { model.detachModelFromServer(id: id) }
                        ) {
                            Text("Embedding model", bundle: .module)
                        }
                    }
                }
            }
            if let info = model.serverInfo?.baseURL {
                APIRequestPreview(method: "POST", url: info.absoluteString + "/v1/embeddings", example: exampleBody)
            }
        }
    }

    /// Why Attach is disabled, in words beside it (DESIGN: a disabled
    /// control says why). Nil when Attach is available.
    private var blockReason: Text? {
        switch block {
        case .none: return nil
        case .serverStopped:
            return Text("Start the server to attach an embedding model.", bundle: .module)
        case .noSelection:
            return Text("Choose an embedding model to attach.", bundle: .module)
        case .busy:
            return Text("Loading", bundle: .module)
        case .alreadyAttached:
            return Text("This embedding model is already attached.", bundle: .module)
        case .refused(let reason):
            return Text(verbatim: reason)
        }
    }
}
