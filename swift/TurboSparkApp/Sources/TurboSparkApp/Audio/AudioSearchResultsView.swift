import SwiftUI

/// Library search results: each recording once, best first, with the lanes
/// that matched it and up to three timestamped hits per lane. A lane is
/// named in text beside its glyph, never by color alone.
@MainActor
struct AudioSearchResultsView: View {
    @ObservedObject var controller: AudioWorkspaceController
    @ObservedObject var search: AudioLibrarySearch

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let reason = search.semanticUnavailableReason {
                Label { Text(reason).fixedSize(horizontal: false, vertical: true) } icon: {
                    Image(systemName: "info.circle").accessibilityHidden(true)
                }
                .themedFont(.small).foregroundStyle(.appSecondary)
            } else if search.isIndexing {
                Label { Text("Indexing recordings for meaning and sound search", bundle: .module) } icon: {
                    ProgressView().controlSize(.mini)
                }
                .themedFont(.small).foregroundStyle(.appSecondary)
            }
            if search.results.isEmpty {
                Text("No matches", bundle: .module).foregroundStyle(.appSecondary).padding(.vertical, 12)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(search.results) { result in
                            if let item = controller.items.first(where: { $0.id == result.itemID }) { row(result, item) }
                        }
                    }
                }
                if search.results.contains(where: { $0.lanes.contains(.sound) }) {
                    Text("Sound matches are experimental.", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Search results", bundle: .module))
    }

    private func row(_ result: AudioSearchResult, _ item: AudioLibraryItem) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Button { controller.select(item) } label: {
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.title).themedFont(.base, weight: .medium).lineLimit(2)
                    Text(item.createdAt, format: .dateTime.month(.abbreviated).day()).themedFont(.small).foregroundStyle(.appSecondary)
                }.frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(controller.selectedID == item.id ? .isSelected : [])
            HStack(spacing: 8) {
                ForEach(result.lanes) { lane in
                    Label { Text(LocalizedStringKey(lane.titleKey), bundle: .module) } icon: { Image(systemName: lane.symbol) }
                        .themedFont(.small).foregroundStyle(.appSecondary)
                }
            }
            ForEach(result.hits) { hit in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    if let start = hit.start {
                        Button { controller.select(item); controller.play(item, at: start) } label: {
                            Text("Play from \(audioTime(start))", bundle: .module).monospacedDigit()
                        }
                        .appAction(.quiet, size: .small)
                        .disabled(controller.hasRecordingActivity || !item.hasPlayableAudio)
                    }
                    if !hit.snippet.isEmpty {
                        Text(verbatim: hit.snippet).themedFont(.small).lineLimit(2).foregroundStyle(.appSecondary)
                    }
                }
            }
        }
        .padding(12)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
        .overlay { RoundedRectangle(cornerRadius: 10).stroke(controller.selectedID == item.id ? .appAccent : .appBorder, lineWidth: controller.selectedID == item.id ? 2 : 1) }
    }
}
