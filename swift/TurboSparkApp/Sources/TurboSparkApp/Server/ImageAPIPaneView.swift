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
            servingPanel
            testPanel
        }
        .onAppear { if selectedAlias.isEmpty { selectedAlias = supported.first?.alias ?? "" } }
        .onChange(of: supported) { _, items in
            if !items.contains(where: { $0.alias == selectedAlias }) { selectedAlias = items.first?.alias ?? "" }
        }
    }

    private var servingPanel: some View {
        APIPanel(
            title: Text("Image API", bundle: .module),
            subtitle: Text("The same address and API key as Text. One installed MLX image model can be attached at a time.", bundle: .module)
        ) {
            Picker(selection: $selectedAlias) {
                ForEach(supported) { item in
                    Text(verbatim: item.alias).tag(item.alias)
                }
            } label: { Text("Image model", bundle: .module) }
            .frame(minWidth: 180, idealWidth: 280, maxWidth: 360)
            Button {
                if let selected { model.attachImageModelToServer(selected) }
            } label: { Text("Attach", bundle: .module) }
            .appAction(.primary, size: APIControlSize.panel)
            .disabled(model.server == nil || selected == nil || model.serverImageAttachedModel != nil)
        } content: {
            if supported.isEmpty {
                Text("Install a supported Z-Image or Qwen-Image model in Image Generation first.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            APIRowList {
                if let attached = model.serverImageAttachedModel {
                    APIServedModelRow(
                        name: attached.alias,
                        detachHelp: Text("Stop serving this and release it from memory.", bundle: .module),
                        detach: { model.detachImageModelFromServer() }
                    ) {
                        Text("Image model", bundle: .module)
                    }
                } else {
                    APINothingServedRow()
                }
            }
            APIStatusLine(label: Text("Progress", bundle: .module), value: APIStatusText.text(model.serverImageProgress))
            if let info = model.serverInfo?.baseURL {
                APIRequestPreview(method: "POST", url: info.absoluteString + "/v1/images/generations", example: exampleBody)
            }
        }
    }

    private var testPanel: some View {
        APIPanel(title: Text("Test image request", bundle: .module)) {
            TextEditor(text: $prompt)
                .themedFont(.base)
                .frame(height: 74)
                .appEditorSurface()
                .accessibilityLabel(Text("Image prompt", bundle: .module))
            Picker(selection: $size) {
                Text(verbatim: "512x512").tag("512x512")
                Text(verbatim: "768x768").tag("768x768")
                Text(verbatim: "1024x1024").tag("1024x1024")
            } label: { Text("Size", bundle: .module) }
            .frame(minWidth: 160, idealWidth: 220, maxWidth: 260)
            HStack(spacing: AppMetrics.actionSpacing) {
                Button {
                    Task { await runTest() }
                } label: { Text("Generate", bundle: .module) }
                .appAction(.primary, size: APIControlSize.panel)
                .disabled(testing || model.serverImageAttachedModel == nil || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                APIRunIndicator(running: testing, elapsed: elapsed)
            }
            if let errorMessage {
                Text(verbatim: errorMessage).foregroundStyle(.red).textSelection(.enabled)
            }
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
                .appAction(.secondary, size: APIControlSize.panel)
                .disabled(saved)
            }
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
                    NSLocalizedDescriptionKey: String(data: data.prefix(2048), encoding: .utf8) ?? AppStrings.localized("Invalid image response")
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
