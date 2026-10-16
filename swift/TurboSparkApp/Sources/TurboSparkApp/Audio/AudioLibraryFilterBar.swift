import SwiftUI

extension AudioWorkspaceController {
    var recordingPreset: AudioRecordingPreset? { preferences.recordingPreset.flatMap(AudioRecordingPreset.init(rawValue:)) }

    /// The preset a new take records as: the chosen one, unless its sources
    /// were changed by hand since, which makes the take a custom one.
    var effectiveRecordingPreset: AudioRecordingPreset? {
        guard let preset = recordingPreset, preset.matches(microphone: captureMicrophone, systemAudio: captureSystemAudio) else { return nil }
        return preset
    }

    func applyRecordingPreset(_ preset: AudioRecordingPreset) {
        guard !hasRecordingActivity else { return }
        preferences.recordingPreset = preset.rawValue
        restoreRecordingPreset()
        flushPreferences()
    }

    /// Puts the saved preset's sources and live text back, at launch and
    /// when one is picked. Sources are not saved on their own.
    func restoreRecordingPreset() {
        guard let preset = recordingPreset else { return }
        captureMicrophone = preset.capturesMicrophone
        captureSystemAudio = preset.capturesSystemAudio
        liveTranscription = preset.liveText
    }

    /// The preset's collection, named in the current language the first time
    /// and kept from then on.
    func collectionName(for preset: AudioRecordingPreset) -> String {
        if let saved = preferences.presetCollections?[preset.rawValue] { return saved }
        let name = preset.defaultCollection
        var names = preferences.presetCollections ?? [:]
        names[preset.rawValue] = name
        preferences.presetCollections = names
        flushPreferences()
        return name
    }

    func setTags(_ item: AudioLibraryItem, _ tags: [String]) { update(item.id) { $0.setTags(tags) } }
    func addTag(_ item: AudioLibraryItem, _ tag: String) { setTags(item, item.tagList + [tag]) }
    func removeTag(_ item: AudioLibraryItem, _ tag: String) { setTags(item, item.tagList.filter { $0 != tag }) }

    var libraryCollections: [AudioFacet] { AudioLibraryOrganizer.collections(items.filter { !$0.isEmptyGeneratedAttempt }) }
    var libraryTags: [AudioFacet] { AudioLibraryOrganizer.tags(items.filter { !$0.isEmptyGeneratedAttempt }) }
}

/// The Record page's choice of what a take is for.
@MainActor
struct AudioRecordingPresetPicker: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        Picker(selection: Binding(
            get: { controller.effectiveRecordingPreset },
            set: { if let preset = $0 { controller.applyRecordingPreset(preset) } }
        )) {
            ForEach(AudioRecordingPreset.allCases) { preset in
                Label { Text(LocalizedStringKey(preset.titleKey), bundle: .module) } icon: { Image(systemName: preset.symbol) }
                    .tag(Optional(preset))
            }
        } label: { Text("Recording type", bundle: .module) }
        .pickerStyle(.segmented)
        .disabled(controller.hasRecordingActivity)
        .help(Text("Sets the sources and live text, and files new takes into a matching collection.", bundle: .module))
    }
}

/// What the library shows: a smart filter, a collection, and a tag.
@MainActor
struct AudioLibraryFilterBar: View {
    @ObservedObject var controller: AudioWorkspaceController
    @Binding var filter: AudioLibraryFilter
    @Binding var collection: String?
    @Binding var tag: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker(selection: $filter) {
                ForEach(AudioLibraryFilter.allCases) { option in
                    Text(LocalizedStringKey(option.titleKey), bundle: .module).tag(option)
                }
            } label: { Text("Show", bundle: .module) }
            HStack(spacing: 8) {
                facetMenu(title: Text("Collection", bundle: .module), systemImage: "folder",
                          facets: controller.libraryCollections, selection: $collection)
                facetMenu(title: Text("Tag", bundle: .module), systemImage: "tag",
                          facets: controller.libraryTags, selection: $tag)
            }
            if collection != nil || tag != nil {
                HStack(spacing: 6) {
                    if let collection {
                        AppChip(Text("Collection: \(collection)", bundle: .module), systemImage: "xmark", isOn: true) { self.collection = nil }
                            .accessibilityHint(Text("Removes this filter", bundle: .module))
                    }
                    if let tag {
                        AppChip(Text("Tag: \(tag)", bundle: .module), systemImage: "xmark", isOn: true) { self.tag = nil }
                            .accessibilityHint(Text("Removes this filter", bundle: .module))
                    }
                }
            }
        }
    }

    private func facetMenu(title: Text, systemImage: String, facets: [AudioFacet], selection: Binding<String?>) -> some View {
        Menu {
            Button { selection.wrappedValue = nil } label: { Text("Any", bundle: .module) }
            if !facets.isEmpty { Divider() }
            ForEach(facets) { facet in
                Button { selection.wrappedValue = facet.name } label: {
                    Text(verbatim: "\(facet.name) (\(facet.count))")
                }
            }
        } label: { Label { title } icon: { Image(systemName: systemImage) } }
        .menuStyle(.borderlessButton).fixedSize()
        .disabled(facets.isEmpty)
        .help(facets.isEmpty ? Text("Nothing in the library has one yet.", bundle: .module) : title)
    }
}

/// An item's tags, removable one by one, and a field that adds one. Shows
/// the tags already used elsewhere as suggestions.
@MainActor
struct AudioTagEditor: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem
    @State private var draft = ""

    private var suggestions: [String] {
        let have = Set(item.tagList.map { $0.lowercased() })
        let typed = draft.trimmingCharacters(in: .whitespaces).lowercased()
        return controller.libraryTags.map(\.name)
            .filter { !have.contains($0.lowercased()) && (typed.isEmpty || $0.lowercased().hasPrefix(typed)) }
            .prefix(5).map { $0 }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                ForEach(item.tagList, id: \.self) { tag in
                    AppChip(Text(verbatim: tag), systemImage: "xmark", isOn: true) { controller.removeTag(item, tag) }
                        .accessibilityLabel(Text("Remove tag \(tag)", bundle: .module))
                }
                TextField(text: $draft) { Text("Add tag", bundle: .module) }
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 160)
                    .onSubmit(add)
                    .disabled(item.tagList.count >= AudioLibraryOrganizer.maxTags)
            }
            if !draft.isEmpty && !suggestions.isEmpty {
                HStack(spacing: 6) {
                    ForEach(suggestions, id: \.self) { tag in
                        AppChip(Text(verbatim: tag), systemImage: "plus") { draft = ""; controller.addTag(item, tag) }
                            .accessibilityLabel(Text("Add tag \(tag)", bundle: .module))
                    }
                }
            }
        }
    }

    private func add() {
        let tag = draft
        draft = ""
        guard !tag.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        controller.addTag(item, tag)
    }
}
