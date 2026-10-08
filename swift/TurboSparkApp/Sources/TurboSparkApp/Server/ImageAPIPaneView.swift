import AppKit
import SwiftUI
import TurboSpark

@MainActor
struct ImageAPIPaneView: View {
    @ObservedObject var model: AppModel
    @State private var selectedAlias = ""
    @State private var prompt = "A red fox in a snowy forest, watercolor illustration"
    @State private var size = "1024x1024"
    @State private var testing = false
    @State private var result: Data?
    @State private var errorMessage: String?
    @State private var elapsed: Double?
    @State private var resultSeed: UInt64?
    @State private var resultPrompt = ""
    @State private var resultSize = "1024x1024"
    @State private var saved = false

    private var supported: [ImageInstalledModel] {
        model.imageModels.filter { AppModel.supportsMLXImageModel(modelID: $0.modelID) }
    }

    private var selected: ImageInstalledModel? {
        supported.first { $0.alias == selectedAlias } ?? supported.first
    }

    private var exampleBody: String {
        let value: [String: Any] = [
            "model": model.serverImageAttachedModel?.alias ?? "<attach an image model>",
            "prompt": "a red fox in a snowy forest", "size": "1024x1024", "n": 1
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        else { return "" }
        return String(decoding: data, as: UTF8.self)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            ServerHeaderBandView(model: model)
            VStack(alignment: .leading, spacing: 12) {
                Text("Image API", bundle: .module).themedFont(.title3, weight: .semibold)
                Text("The same address and API key as Text. One installed MLX image model can be attached at a time.", bundle: .module)
                    .foregroundStyle(.secondary)
                HStack {
                    Picker("Image model", selection: $selectedAlias) {
                        ForEach(supported) { item in
                            Text(item.alias).tag(item.alias)
                        }
                    }
                    .frame(maxWidth: 360)
                    Button {
                        if let selected { model.attachImageModelToServer(selected) }
                    } label: { Text("Attach", bundle: .module) }
                    .disabled(model.server == nil || selected == nil || model.serverImageAttachedModel != nil)
                    Button { model.detachImageModelFromServer() } label: {
                        Text("Detach", bundle: .module)
                    }
                        .disabled(model.serverImageAttachedModel == nil)
                }
                if supported.isEmpty {
                    Text("Install a supported Z-Image or Qwen-Image model in Image Generation first.", bundle: .module)
                        .foregroundStyle(.secondary)
                }
                if let info = model.serverInfo?.baseURL {
                    Text(verbatim: "POST " + info.absoluteString + "/v1/images/generations")
                        .themedCode(.callout)
                        .textSelection(.enabled)
                    Text(exampleBody)
                        .themedCode(.small)
                        .textSelection(.enabled)
                }
                Text(verbatim: "Attached: \(model.serverImageAttachedModel?.alias ?? "none")")
                Text(verbatim: "Progress: \(model.serverImageProgress)")
                    .accessibilityLabel(Text(verbatim: "Image progress: \(model.serverImageProgress)"))
            }
            .padding(16)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))

            VStack(alignment: .leading, spacing: 12) {
                Text("Test image request", bundle: .module).themedFont(.title3, weight: .semibold)
                TextEditor(text: $prompt)
                    .frame(height: 74)
                    .border(.appBorder)
                    .accessibilityLabel(Text("Image prompt", bundle: .module))
                Picker("Size", selection: $size) {
                    Text(verbatim: "512x512").tag("512x512")
                    Text(verbatim: "768x768").tag("768x768")
                    Text(verbatim: "1024x1024").tag("1024x1024")
                }
                .frame(maxWidth: 260)
                HStack {
                    Button {
                        Task { await runTest() }
                    } label: { Text("Generate", bundle: .module) }
                    .buttonStyle(.borderedProminent)
                    .disabled(testing || model.serverImageAttachedModel == nil || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    if testing { ProgressView().controlSize(.small) }
                    if let elapsed { Text(String(format: "%.1f s", elapsed)) }
                }
                if let errorMessage { Text(errorMessage).foregroundStyle(.red).textSelection(.enabled) }
                if let result, let image = NSImage(data: result) {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 420)
                        .accessibilityLabel(Text("Generated image preview", bundle: .module))
                    Button {
                        saved = model.saveAPIImageToGallery(
                            png: result, prompt: resultPrompt, seed: resultSeed ?? 0,
                            width: UInt32(resultSize.split(separator: "x").first.flatMap { Int($0) } ?? 1024),
                            height: UInt32(resultSize.split(separator: "x").last.flatMap { Int($0) } ?? 1024))
                    } label: { Text("Save to gallery", bundle: .module) }
                    .disabled(saved)
                }
            }
            .padding(16)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
        }
        .onAppear { if selectedAlias.isEmpty { selectedAlias = supported.first?.alias ?? "" } }
        .onChange(of: supported) { _, items in
            if !items.contains(where: { $0.alias == selectedAlias }) { selectedAlias = items.first?.alias ?? "" }
        }
    }

    private func runTest() async {
        guard let modelID = model.serverImageAttachedModel?.alias,
              let baseURL = model.serverInfo?.baseURL else { return }
        testing = true
        result = nil
        saved = false
        errorMessage = nil
        elapsed = nil
        let started = Date()
        let submittedPrompt = prompt
        let submittedSize = size
        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("v1/images/generations"))
            request.httpMethod = "POST"
            request.timeoutInterval = 1_200
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            if let key = AppModel.serverAPIKey(from: model.serverAPIKeyInput) {
                request.setValue("Bearer \(key)", forHTTPHeaderField: "authorization")
            }
            request.httpBody = try JSONSerialization.data(withJSONObject: [
                "model": modelID, "prompt": submittedPrompt, "size": submittedSize, "n": 1,
                "response_format": "b64_json"
            ])
            let (data, response) = try await URLSession.shared.data(for: request)
            // Parse leniently: error bodies are often plain text (extractor
            // rejections, proxy pages) and must reach the user verbatim.
            let document = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
                  let item = (document?["data"] as? [[String: Any]])?.first,
                  let encoded = item["b64_json"] as? String,
                  let png = Data(base64Encoded: encoded) else {
                throw NSError(domain: "Image API", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: String(data: data.prefix(2048), encoding: .utf8) ?? "Invalid image response"
                ])
            }
            result = png
            resultPrompt = submittedPrompt
            resultSize = submittedSize
            resultSeed = http.value(forHTTPHeaderField: "x-turbospark-seed").flatMap(UInt64.init)
            elapsed = Date().timeIntervalSince(started)
        } catch {
            errorMessage = error.localizedDescription
        }
        testing = false
    }
}
