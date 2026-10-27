import SwiftUI

@MainActor
struct AudioPromptLibraryView: View {
    @ObservedObject var controller: AudioWorkspaceController
    @Environment(\.dismiss) private var dismiss
    @State private var search = ""
    @State private var name = ""
    @State private var showingName = false
    @State private var editing: AudioPreset?
    @State private var deleting: AudioPreset?

    @Environment(\.appTheme) private var theme

    var body: some View {
        let scale = AppMetrics.textScale(theme)
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("Prompt library", bundle: .module).themedFont(.title2, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Button {
                    editing = nil
                    name = String((controller.task == .music ? controller.recipe.caption : controller.recipe.text).prefix(60))
                    showingName = true
                } label: { Text("Save Prompt", bundle: .module) }
                    .appAction(.secondary)
                    .disabled(!controller.canSavePrompt)
            }
            Text("Choose a prompt, edit it, or write your own.", bundle: .module).foregroundStyle(.appSecondary)
            AppSearchField(text: $search, prompt: Text("Search prompts", bundle: .module), maxWidth: .infinity)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if controller.prompts(matching: search).isEmpty {
                        AppEmptyState(systemImage: "books.vertical", title: Text("No saved prompts", bundle: .module),
                                      message: Text("Choose a prompt, edit it, or write your own.", bundle: .module))
                    }
                    ForEach(controller.prompts(matching: search)) { prompt in
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text(verbatim: prompt.name).themedFont(.base, weight: .semibold)
                                Spacer()
                                Menu {
                                    Button {
                                        editing = prompt; name = prompt.name; showingName = true
                                    } label: { Text("Rename", bundle: .module) }
                                    Button(role: .destructive) { deleting = prompt } label: { Text("Delete", bundle: .module) }
                                } label: { Image(systemName: "ellipsis") }
                                    .menuStyle(.borderlessButton).fixedSize()
                                    .accessibilityLabel(Text(verbatim: prompt.name))
                            }
                            Text(controller.task == .music ? prompt.recipe.caption : prompt.recipe.text)
                                .lineLimit(4).foregroundStyle(.appSecondary)
                            if controller.task == .music && !prompt.recipe.lyrics.isEmpty {
                                Text(prompt.recipe.lyrics).lineLimit(2).themedFont(.small).foregroundStyle(.appSecondary)
                            }
                            Button { controller.loadPrompt(prompt); dismiss() } label: { Text("Load Prompt", bundle: .module) }
                                .appAction(.secondary, size: .small)
                                .disabled(controller.isBusy)
                        }
                        .appCardSurface(padding: 14)
                    }
                }
            }
            if let error = controller.error { Text(error).foregroundStyle(.appSecondary).lineLimit(3) }
            HStack {
                Text("Saved in this profile", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                Spacer()
                Button { dismiss() } label: { Text("Close", bundle: .module) }
                    .appAction(.secondary)
                    .keyboardShortcut(.cancelAction)
            }
        }
        .padding(24)
        .frame(minWidth: (520 * scale).rounded(), idealWidth: (600 * scale).rounded(), minHeight: (440 * scale).rounded(), idealHeight: (540 * scale).rounded())
        .themedFont(.base)
        .background(.appPage)
        .sheet(isPresented: $showingName) {
            AudioNameSheet(title: editing == nil ? "Save Prompt" : "Rename", value: $name) {
                let saved = editing.map { controller.renamePrompt($0, name: name) } ?? controller.savePrompt(name: name)
                if saved { showingName = false }
            }
        }
        .confirmationDialog(Text("Delete Preset", bundle: .module), isPresented: Binding(
            get: { deleting != nil }, set: { if !$0 { deleting = nil } }
        ), titleVisibility: .visible) {
            Button(role: .destructive) {
                if let deleting { controller.deletePrompt(deleting) }
                deleting = nil
            } label: { Text("Delete", bundle: .module) }
            Button(role: .cancel) { deleting = nil } label: { Text("Cancel", bundle: .module) }
        } message: { if let deleting { Text(verbatim: deleting.name) } }
    }
}
