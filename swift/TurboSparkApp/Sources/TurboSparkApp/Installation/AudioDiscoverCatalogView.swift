import SwiftUI
import TurboSpark

/// Audio recommendations in Discover. It reuses the installed-audio rows and
/// detail lifecycle so downloads stay on the verified audio catalog path.
@MainActor
struct AudioDiscoverCatalogView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    @Binding var filter: ModelHubFilter
    @State private var selectedAlias: String?
    @Environment(\.appTheme) private var theme

    private var entries: [AudioDiscoverCatalog.Entry] {
        AudioDiscoverCatalog.entries(
            profiles: controller.profiles,
            searchText: filter.searchText,
            sort: filter.sort,
            canRun: { AudioRuntimeProvider($0).canRun($0) })
    }

    private var selectedEntry: AudioDiscoverCatalog.Entry? {
        if let selectedAlias, let match = entries.first(where: { $0.id == selectedAlias }) { return match }
        return entries.first
    }

    var body: some View {
        let scale = AppMetrics.textScale(theme)
        VStack(spacing: 0) {
            header
            Divider()
            HStack(spacing: 0) {
                list
                    .frame(minWidth: 260 * scale, idealWidth: 320 * scale, maxWidth: 380 * scale)
                    .frame(maxHeight: .infinity)
                Rectangle().fill(.appBorder).frame(width: AppChromeLayout.dividerWidth)
                detail
                    .frame(minWidth: 280 * scale, maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task {
            if controller.profiles.isEmpty && !controller.isRefreshingModels { controller.refreshModels() }
        }
    }

    private var header: some View {
        AppAdaptiveHeader {
            ModelWorkspaceTitle(
                systemImage: "waveform",
                tint: .teal,
                title: Text("Audio Models", bundle: .module),
                badge: Text("Recommended", bundle: .module),
                summary: Text("Show the pinned audio models this Mac can download", bundle: .module))
        } actions: {
            Button { controller.refreshModels() } label: {
                Label { Text("Rescan", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
            }
            .appAction(.secondary)
            .disabled(controller.isRefreshingModels)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 4) {
                if entries.isEmpty {
                    emptyList
                } else {
                    ForEach(AudioDiscoverCatalog.grouped(entries), id: \.group) { section in
                        Section {
                            ForEach(section.entries) { entry in
                                AudioInstalledRow(
                                    model: model,
                                    controller: controller,
                                    entry: entry.installedEntry,
                                    isSelected: selectedEntry?.id == entry.id) { selectedAlias = entry.id }
                            }
                        } header: {
                            HStack {
                                Text(LocalizedStringKey(section.group.titleKey), bundle: .module)
                                    .themedFont(.small, weight: .bold).foregroundStyle(.appSecondary)
                                    .accessibilityAddTraits(.isHeader)
                                Spacer()
                                Text(verbatim: "\(section.entries.count)")
                                    .themedFont(.tiny).monospacedDigit().foregroundStyle(.appSecondary)
                            }
                            .padding(.horizontal, 8).padding(.top, 10).padding(.bottom, 2)
                        }
                    }
                }
            }
            .padding(8)
        }
    }

    @ViewBuilder private var emptyList: some View {
        if controller.isRefreshingModels {
            ProgressView().controlSize(.small).frame(maxWidth: .infinity).padding(.top, 24)
        } else if let error = controller.modelCatalogError {
            AppEmptyState(systemImage: "exclamationmark.triangle", title: Text("Audio Models", bundle: .module),
                          message: Text(verbatim: error)) {
                Button { controller.refreshModels() } label: { Text("Retry", bundle: .module) }
                    .appAction(.secondary, size: .small)
            }
        } else {
            AppEmptyState(systemImage: "waveform", title: Text("No recommended models", bundle: .module))
        }
    }

    @ViewBuilder private var detail: some View {
        if let entry = selectedEntry {
            AudioInstalledModelDetailView(model: model, controller: controller, entry: entry.installedEntry)
                .id(entry.id)
        } else {
            AppEmptyState(systemImage: "waveform", title: Text("No Model Selected", bundle: .module))
        }
    }
}
