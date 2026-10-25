import SwiftUI
import TurboSpark

/// Sends one `POST /v1/embeddings` to the running server with its API key,
/// the request an outside client would send, and shows the vector's width,
/// its first values and the wall time.
@MainActor
struct EmbeddingsAPITestPanel: View {
    @ObservedObject var model: AppModel
    @State private var input = "The quick brown fox jumps over the lazy dog."
    @State private var testing = false
    @State private var elapsed: Double?
    @State private var vector: [Double]?
    @State private var errorMessage: String?

    private var attachedID: String? { model.serverEmbeddingModelIDs.first }

    var body: some View {
        APIPanel(title: Text("Test embedding request", bundle: .module)) {
            TextEditor(text: $input)
                .themedFont(.base)
                .appEditorSurface(minHeight: 60)
                .accessibilityLabel(Text("Text to embed", bundle: .module))
            HStack(spacing: AppMetrics.actionSpacing) {
                Button {
                    Task { await runTest() }
                } label: { Text("Send", bundle: .module) }
                .appAction(.primary, size: APIControlSize.panel)
                .disabled(testing || attachedID == nil || model.serverEmbeddingAttachInFlight
                          || input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help(attachedID == nil
                      ? Text("Attach an embedding model first.", bundle: .module)
                      : Text("Send", bundle: .module))
                APIRunIndicator(running: testing, elapsed: elapsed)
            }
            if attachedID == nil {
                Text("Attach an embedding model first.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            if let errorMessage {
                Label {
                    Text(verbatim: errorMessage).textSelection(.enabled)
                } icon: {
                    Image(systemName: "exclamationmark.octagon")
                }
                .themedFont(.small)
                .foregroundStyle(.red)
            }
            if let vector {
                Text("\(vector.count) dimensions", bundle: .module)
                    .themedFont(.small, weight: .medium)
                Text(verbatim: vector.prefix(6).map { String(format: "%.5f", $0) }.joined(separator: ", ") + ", ...")
                    .themedCode(.small)
                    .foregroundStyle(.appSecondary)
                    .textSelection(.enabled)
            }
        }
    }

    private func runTest() async {
        guard let modelID = attachedID, let baseURL = model.serverInfo?.baseURL else { return }
        testing = true
        vector = nil
        errorMessage = nil
        elapsed = nil
        let started = Date()
        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("v1/embeddings"))
            request.httpMethod = "POST"
            // A first request to a Core ML model can compile it.
            request.timeoutInterval = 300
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            if let key = AppModel.serverAPIKey(from: model.serverAPIKeyInput) {
                request.setValue("Bearer \(key)", forHTTPHeaderField: "authorization")
            }
            request.httpBody = try JSONSerialization.data(withJSONObject: [
                "model": modelID, "input": input
            ])
            let (data, response) = try await URLSession.shared.data(for: request)
            // Error bodies reach the user verbatim, as in the other panes.
            let document = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
                  let item = (document?["data"] as? [[String: Any]])?.first,
                  let values = item["embedding"] as? [Double] else {
                throw NSError(domain: "Embeddings API", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: String(data: data.prefix(2048), encoding: .utf8) ?? ""
                ])
            }
            vector = values
            elapsed = Date().timeIntervalSince(started)
        } catch {
            errorMessage = error.localizedDescription
        }
        testing = false
    }
}
