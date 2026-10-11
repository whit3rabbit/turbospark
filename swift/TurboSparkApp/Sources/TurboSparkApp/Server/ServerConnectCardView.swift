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
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }

            // Snippet titles, notes, family names and route summaries are
            // fixed English in ServerEndpointCatalog (pure, and tested
            // against the router), so they are looked up in the catalog here.
            Picker(selection: $selection) {
                ForEach(snippets) { snippet in
                    Text(LocalizedStringKey(snippet.title), bundle: .module).tag(snippet.id)
                }
            } label: { Text("Client", bundle: .module) }
            .pickerStyle(.segmented)
            .labelsHidden()

            if let snippet = snippets.first(where: { $0.id == selection }) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(LocalizedStringKey(snippet.note), bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)

                    ZStack(alignment: .topTrailing) {
                        ScrollView(.horizontal, showsIndicators: false) {
                            Text(verbatim: snippet.body)
                                .themedCode(.callout)
                                .textSelection(.enabled)
                                .padding(4)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }

                        APICopyButton(label: Text("Copy code snippet to clipboard", bundle: .module)) { snippet.body }
                            .padding(4)
                    }
                    // The same surface as the request and response editors
                    // in the Image, Audio and TypeSafe modes.
                    .appEditorSurface()
                }
            }

            if AppModel.serverAPIKey(from: model.serverAPIKeyInput) == nil {
                // Most clients require SOME key even when nothing checks it,
                // so the snippets send a dummy rather than omitting the
                // header -- which produces a confusing client-side failure
                // instead of a working call.
                Text("No key is set, so the snippets send a placeholder that is never checked.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }

            ServerEndpointListView()
        }
    }
}

/// The routes, grouped by the API they belong to.
///
/// **ONLY IMPLEMENTED ROUTES, AND THE CATALOG IS THE SOURCE.** A test keeps
/// `ServerEndpointCatalog` in step with the router's own route table, so a
/// row shown here is one the server actually answers. `/v1/embeddings`,
/// `/api/embeddings` and `/api/embed` answer 503 until an embedding model
/// is attached under Advanced, which is what that field's help text says.
private struct ServerEndpointListView: View {
    @State private var family: ServerAPIFamily = .openAI

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider().padding(.vertical, 4)

            HStack {
                Text("Endpoints", bundle: .module)
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                Spacer()
                Picker(selection: $family) {
                    ForEach(ServerAPIFamily.allCases) { option in
                        Text(LocalizedStringKey(option.title), bundle: .module).tag(option)
                    }
                } label: { Text("API family", bundle: .module) }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(minWidth: 220, idealWidth: 300, maxWidth: 360)
            }

            Text(LocalizedStringKey(family.blurb), bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)

            VStack(alignment: .leading, spacing: 4) {
                ForEach(ServerEndpointCatalog.endpoints(for: family)) { endpoint in
                    HStack(spacing: 8) {
                        Text(verbatim: endpoint.method)
                            .themedCode(.small, weight: .semibold)
                            .foregroundStyle(.appSecondary)
                            .frame(width: 34, alignment: .leading)
                        Text(verbatim: endpoint.path)
                            .themedCode(.callout)
                            .textSelection(.enabled)
                        if endpoint.streams {
                            Text(verbatim: family == .ollama ? "NDJSON" : "SSE")
                                .themedFont(.micro, weight: .medium)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Color.secondary.opacity(0.15), in: Capsule())
                                .help(
                                    family == .ollama
                                        ? Text("One JSON object per line, no sentinel. Ollama's framing.", bundle: .module)
                                        : Text("Server-sent events.", bundle: .module))
                        }
                        Spacer(minLength: 8)
                        Text(LocalizedStringKey(endpoint.summary), bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
            }
        }
    }
}
