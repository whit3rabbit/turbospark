import SwiftUI

/// Library items grouped by day, with each row's actions: favorite, rename,
/// collection, duplicate and delete. The Audio library and each
/// Speech-to-Text mode list their own items with it. The caller scrolls it.
@MainActor
struct AudioLibraryList: View {
    @ObservedObject var controller: AudioWorkspaceController
    let items: [AudioLibraryItem]
    /// The row drawn as selected. A Speech-to-Text mode passes a transcript's
    /// source, so the row stays lit while its transcript shows.
    var selectedID: UUID?
    @State private var editingItem: AudioLibraryItem?
    @State private var editingCollection = false
    @State private var editedName = ""
    @State private var deletingItem: AudioLibraryItem?

    init(controller: AudioWorkspaceController, items: [AudioLibraryItem], selectedID: UUID? = nil) {
        self.controller = controller
        self.items = items
        self.selectedID = selectedID
    }

    private var highlightedID: UUID? { selectedID ?? controller.selectedID }

    var body: some View {
        LazyVStack(alignment: .leading, spacing: 8) {
            ForEach(AudioLibraryOrganizer.grouped(items), id: \.group) { section in
                Text(LocalizedStringKey(section.group.titleKey), bundle: .module)
                    .themedFont(.small, weight: .semibold).foregroundStyle(.appSecondary)
                    .accessibilityAddTraits(.isHeader)
                    .padding(.top, section.group == .today ? 0 : 6)
                ForEach(section.items) { item in row(item) }
            }
        }
        .sheet(item: $editingItem) { item in
            AudioNameSheet(title: editingCollection ? "Collection" : "Rename", value: $editedName,
                           allowEmpty: editingCollection) {
                if editingCollection { controller.organize(item, collection: editedName) }
                else { controller.rename(item, title: editedName) }
                editingItem = nil
            }
        }
        .modifier(AudioDeleteConfirmation(controller: controller, item: $deletingItem))
    }

    private func row(_ item: AudioLibraryItem) -> some View {
        let isSelected = highlightedID == item.id
        return HStack(alignment: .top, spacing: 8) {
            Button { controller.select(item) } label: {
                VStack(alignment: .leading, spacing: 6) {
                    Text(item.title).themedFont(.base, weight: .medium).lineLimit(2)
                    HStack {
                        Text(item.createdAt, format: .dateTime.month(.abbreviated).day())
                        if item.hasPlayableAudio { Text(audioTime(item.duration)).monospacedDigit() }
                    }.themedFont(.small).foregroundStyle(.appSecondary)
                    if !item.collection.isEmpty { Text(item.collection).themedFont(.small).foregroundStyle(.appSecondary) }
                    if !item.tagList.isEmpty {
                        Label { Text(verbatim: item.tagList.joined(separator: ", ")).lineLimit(1) } icon: { Image(systemName: "tag") }
                            .themedFont(.small).foregroundStyle(.appSecondary)
                    }
                    if item.status != .completed {
                        Label { Text(LocalizedStringKey(item.status.audioTitle), bundle: .module) }
                            icon: { Image(systemName: item.status.audioSymbol) }
                            .themedFont(.small)
                    }
                }.frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(isSelected ? .isSelected : [])
            .accessibilityAction(named: Text("Favorite", bundle: .module)) { controller.toggleFavorite(item) }
            .accessibilityAction(named: Text("Rename", bundle: .module)) { beginEditing(item, collection: false) }
            .accessibilityAction(named: Text("Delete", bundle: .module)) { deletingItem = item }
            VStack(spacing: 8) {
                Button { controller.toggleFavorite(item) } label: {
                    Image(systemName: item.favorite ? "star.fill" : "star")
                }.buttonStyle(.plain)
                    .accessibilityLabel(Text(item.favorite ? "Remove favorite" : "Favorite", bundle: .module))
                Menu {
                    Button { beginEditing(item, collection: false) } label: { Text("Rename", bundle: .module) }
                    Button { beginEditing(item, collection: true) } label: { Text("Collection", bundle: .module) }
                    Button { controller.select(item); controller.duplicateSelected() } label: {
                        Text("Duplicate with changes", bundle: .module)
                    }
                    Divider()
                    Button(role: .destructive) { deletingItem = item } label: { Text("Delete", bundle: .module) }
                } label: { Image(systemName: "ellipsis") }
                .menuStyle(.borderlessButton).fixedSize()
                .accessibilityLabel(Text("Audio actions", bundle: .module))
                .disabled(controller.hasRecordingActivity || controller.isBusy)
                AudioDeleteButton(controller: controller, item: item, compact: true)
            }
        }
        .padding(12)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
        .overlay { RoundedRectangle(cornerRadius: 10).stroke(isSelected ? .appAccent : .appBorder, lineWidth: isSelected ? 2 : 1) }
    }

    private func beginEditing(_ item: AudioLibraryItem, collection: Bool) {
        editingCollection = collection
        editedName = collection ? item.collection : item.title
        editingItem = item
    }
}

extension AudioLibraryItem.Status {
    var audioTitle: String {
        switch self {
        case .draft: return "Draft"
        case .recording: return "Recording"
        case .processing: return "Processing"
        case .completed: return "Completed"
        case .interrupted: return "Interrupted"
        case .failed: return "Failed"
        }
    }
    var audioSymbol: String {
        switch self {
        case .draft: return "pencil"
        case .recording: return "record.circle"
        case .processing: return "hourglass"
        case .completed: return "checkmark.circle"
        case .interrupted: return "pause.circle"
        case .failed: return "exclamationmark.triangle"
        }
    }
}
