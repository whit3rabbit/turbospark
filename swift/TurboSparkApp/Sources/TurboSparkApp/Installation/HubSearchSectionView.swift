import SwiftUI
import TurboSpark

/// The "On Hugging Face" section under the Discover list.
///
/// Local matches stay on top and this sits below them, so one search box
/// answers both "is it in the catalog" and "is it on Hugging Face" without a
/// second place to type. It does nothing until asked: the idle state is a row
/// that says what pressing it sends.
@MainActor
struct HubSearchSectionView: View {
    @ObservedObject var controller: HubSearchController
    /// The trimmed text in the search box.
    let query: String
    /// `owner/name` when the text is shaped like a repository (or a pasted
    /// Hugging Face URL), so it can be checked directly instead of searched.
    let directRepo: String?
    @Binding var selectedRepo: String?
    let onSearch: () -> Void
    let onSettings: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            AppSectionHeader(title: Text("On Hugging Face", bundle: .module))
                .padding(.top, 10)
            content
        }
        // A format or sort change re-asks, but only once there is an answer to
        // change: it must not send a query nobody submitted.
        .onChange(of: controller.format) { reSearchIfAnswered() }
        .onChange(of: controller.sort) { reSearchIfAnswered() }
    }

    @ViewBuilder private var content: some View {
        switch controller.phase {
        case .idle:
            idleRows
        case .searching:
            searchingRow
        case .loaded(let rows, let fromCache):
            filterRow
            if rows.isEmpty {
                noResults
            } else {
                resultRows(rows, fromCache: fromCache)
            }
        case .failed(let status, let message, _):
            failure(status: status, message: message)
        }
    }

    // MARK: - States

    private var idleRows: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button(action: onSearch) {
                Label {
                    Text("Search Hugging Face for \u{201C}\(query)\u{201D}", bundle: .module)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                } icon: {
                    Image(systemName: "magnifyingglass")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .appAction(.secondary, size: .small)
            .help(Text("Sends this search to huggingface.co. Press Return in the search box to do the same.", bundle: .module))

            if let directRepo {
                Button { selectedRepo = directRepo } label: {
                    Label {
                        Text("Check \(directRepo) directly", bundle: .module)
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                    } icon: {
                        Image(systemName: "checkmark.shield")
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .appAction(.quiet, size: .small)
                .help(Text("Reads this repository's headers to see whether it runs here, without downloading it.", bundle: .module))
            }
        }
    }

    private var searchingRow: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text("Searching Hugging Face...", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .combine)
    }

    private var noResults: some View {
        AppEmptyState(
            systemImage: "magnifyingglass",
            title: Text("No text models found", bundle: .module),
            message: Text("Nothing on Hugging Face matched this search. Try fewer words or another format.", bundle: .module)
        ) {
            if controller.format != .any {
                Button { controller.format = .any } label: { Text("Any format", bundle: .module) }
                    .appAction(.secondary, size: .small)
            }
        }
    }

    private func resultRows(_ rows: [HubSearchResult], fromCache: Bool) -> some View {
        LazyVStack(spacing: 3) {
            ForEach(rows) { row in
                HubResultRowView(
                    result: row,
                    isSelected: selectedRepo?.caseInsensitiveCompare(row.repoId) == .orderedSame,
                    onSelect: { selectedRepo = row.repoId })
            }
            if fromCache {
                Text("Showing results saved in the last 15 minutes.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 4)
            }
        }
    }

    private var filterRow: some View {
        FlowLayout(spacing: 6, lineSpacing: 4) {
            ModelFilterMenu(
                name: Text("Result format", bundle: .module),
                value: formatText(controller.format)
            ) {
                Picker(selection: $controller.format) {
                    ForEach(HubSearchFormat.allCases, id: \.self) { format in
                        formatText(format).tag(format)
                    }
                } label: { Text("Result format", bundle: .module) }
                .pickerStyle(.inline)
            }
            ModelFilterMenu(
                name: Text("Sort Hugging Face results", bundle: .module),
                value: sortText(controller.sort)
            ) {
                Picker(selection: $controller.sort) {
                    ForEach(HubSearchSort.allCases, id: \.self) { sort in
                        sortText(sort).tag(sort)
                    }
                } label: { Text("Sort Hugging Face results", bundle: .module) }
                .pickerStyle(.inline)
            }
        }
    }

    private func failure(status: HubSearchStatus, message: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label {
                failureText(status: status, message: message)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: status == .offline ? "wifi.slash" : "exclamationmark.triangle.fill")
            }
            .themedFont(.small)
            .foregroundStyle(.appWarning)

            HStack(spacing: 8) {
                Button(action: onSearch) { Text("Try again", bundle: .module) }
                    .appAction(.secondary, size: .small)
                if status == .needsAuth {
                    Button(action: onSettings) { Text("Open Settings", bundle: .module) }
                        .appAction(.quiet, size: .small)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appWarningTint, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .accessibilityElement(children: .contain)
    }

    /// Each status says what to DO. The engine's own sentence is kept for the
    /// generic case, where it is the only information there is.
    private func failureText(status: HubSearchStatus, message: String) -> Text {
        switch status {
        case .offline:
            return Text("You appear to be offline. Check your connection and try again.", bundle: .module)
        case .rateLimited:
            // Not the exact wait the Hub named: a count of seconds would need
            // plural forms in 21 languages for a figure that is usually
            // under a minute.
            return Text("Hugging Face asked us to slow down. Try again in a minute.", bundle: .module)
        case .needsAuth:
            return Text("Hugging Face rejected the request. Check your token in Settings.", bundle: .module)
        case .ok, .error:
            return Text(verbatim: message)
        }
    }

    // MARK: - Helpers

    private func reSearchIfAnswered() {
        guard case .loaded = controller.phase else { return }
        onSearch()
    }

    private func formatText(_ format: HubSearchFormat) -> Text {
        switch format {
        case .any: return Text("Any format", bundle: .module)
        case .gguf: return Text(verbatim: "GGUF")
        case .mlx: return Text(verbatim: "MLX")
        }
    }

    private func sortText(_ sort: HubSearchSort) -> Text {
        switch sort {
        case .downloads: return Text("Most downloads", bundle: .module)
        case .trending: return Text("Trending", bundle: .module)
        case .likes: return Text("Most liked", bundle: .module)
        }
    }
}
