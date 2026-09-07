import AppKit
import SwiftUI

/// How to point a tool at this server, with the live port and key already in
/// the snippet.
///
/// **THIS IS THE HOBBYIST FRONT DOOR, AND IT IS WHY A LIST OF ROUTES IS NOT
/// ENOUGH.** A person whose goal is "make my editor talk to this" does not
/// want to know that `POST /v1/chat/completions` exists; they want the two
/// environment variables that make it work. The port is OS-assigned, so it
/// is not something they could have known without reading it off this pane,
/// and a snippet they have to edit is the step people get wrong.
struct ServerConnectCardView: View {
    @ObservedObject var model: AppModel
    @State private var selection: String = "claude-code"
    @State private var copiedID: String?

    private var snippets: [ServerConnectSnippet] {
        ServerConnectRecipes.snippets(
            baseURL: model.serverInfo?.baseURL?.absoluteString ?? "http://127.0.0.1:<port>",
            modelID: model.serverInfo?.models.first ?? "",
            apiKey: AppModel.serverAPIKey(from: model.serverAPIKeyInput))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if model.server == nil {
                Text("Start the server and these fill in with its real address.", bundle: .module)
                    .themedFont(points: 11)
                    .foregroundStyle(.secondary)
            }

            Picker("", selection: $selection) {
                ForEach(snippets) { snippet in
                    Text(snippet.title).tag(snippet.id)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if let snippet = snippets.first(where: { $0.id == selection }) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(snippet.note)
                        .themedFont(points: 11)
                        .foregroundStyle(.secondary)

                    ZStack(alignment: .topTrailing) {
                        ScrollView(.horizontal, showsIndicators: false) {
                            Text(snippet.body)
                                .themedCode(points: 11)
                                .textSelection(.enabled)
                                .padding(12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }

                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(snippet.body, forType: .string)
                            copiedID = snippet.id
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                copiedID = nil
                            }
                        } label: {
                            Image(systemName: copiedID == snippet.id ? "checkmark" : "doc.on.doc")
                        }
                        .buttonStyle(.borderless)
                        .padding(8)
                        .help("Copy code snippet to clipboard")
                        .accessibilityLabel("Copy code snippet")
                        .accessibilityValue(copiedID == snippet.id ? "Copied" : "")
                    }
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color(nsColor: .textBackgroundColor).opacity(0.5)))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5))
                }
            }

            if AppModel.serverAPIKey(from: model.serverAPIKeyInput) == nil {
                // Most clients require SOME key even when nothing checks it,
                // so the snippets send a dummy rather than omitting the
                // header -- which produces a confusing client-side failure
                // instead of a working call.
                Text("No key is set, so the snippets send a placeholder that is never checked.", bundle: .module)
                    .themedFont(points: 10)
                    .foregroundStyle(.secondary)
            }

            ServerEndpointListView()
        }
    }
}

/// The routes, grouped by the API they belong to.
///
/// **ONLY IMPLEMENTED ROUTES, AND `/v1/embeddings` IS THE ABSENCE WORTH
/// NOTICING.** This engine has no embedding path at all, so a row for it
/// would be a capability claim a user could go and build against.
private struct ServerEndpointListView: View {
    @State private var family: ServerAPIFamily = .openAI

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider().padding(.vertical, 4)

            HStack {
                Text("Endpoints", bundle: .module)
                    .themedFont(points: 11, weight: .semibold)
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("", selection: $family) {
                    ForEach(ServerAPIFamily.allCases) { option in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 300)
            }

            Text(family.blurb)
                .themedFont(points: 10)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                ForEach(ServerEndpointCatalog.endpoints(for: family)) { endpoint in
                    HStack(spacing: 8) {
                        Text(endpoint.method)
                            .themedCode(points: 9, weight: .semibold)
                            .foregroundStyle(.secondary)
                            .frame(width: 34, alignment: .leading)
                        Text(endpoint.path)
                            .themedCode(points: 11)
                            .textSelection(.enabled)
                        if endpoint.streams {
                            Text(family == .ollama ? "NDJSON" : "SSE")
                                .themedFont(points: 8, weight: .medium)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Color.secondary.opacity(0.15), in: Capsule())
                                .help(
                                    family == .ollama
                                        ? "One JSON object per line, no sentinel. Ollama's framing."
                                        : "Server-sent events.")
                        }
                        Spacer(minLength: 8)
                        Text(endpoint.summary)
                            .themedFont(points: 10)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
            }
        }
    }
}
