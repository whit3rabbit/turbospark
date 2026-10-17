import SwiftUI
import TurboSpark

/// Organization card for managing custom model tags and personal notes.
struct InstalledModelOrganizationCardView: View {
    let installedModel: InstalledModel
    @ObservedObject private var orgStore = ModelOrganizationStore.shared

    @State private var notesText: String = ""
    @State private var isAddingTag = false
    @State private var newTagText: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label { Text("Organization & Notes", bundle: .module) } icon: { Image(systemName: "tag") }
                .themedFont(.small, weight: .semibold)

            // Tags section
            VStack(alignment: .leading, spacing: 6) {
                Text("Custom Tags", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)

                FlowLayout(spacing: 6, lineSpacing: 6) {
                    ForEach(orgStore.tags(installedModel), id: \.self) { tag in
                        HStack(spacing: 4) {
                            Text(verbatim: tag)
                                .themedFont(.tiny, weight: .medium)
                            AppIconButton("xmark", label: Text("Remove tag \(tag)", bundle: .module)) {
                                orgStore.removeTag(tag, for: installedModel)
                            }
                            .buttonStyle(.plain)
                            .themedFont(.micro, weight: .bold)
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color.accentColor.opacity(0.14), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(Color.accentColor)
                    }

                    if isAddingTag {
                        HStack(spacing: 4) {
                            TextField(text: $newTagText, prompt: Text("Tag name", bundle: .module)) {
                                Text("Tag name", bundle: .module)
                            }
                                .textFieldStyle(.plain)
                                .themedFont(.tiny)
                                .frame(width: 80)
                                .onSubmit {
                                    addTagAction()
                                }
                            Button {
                                addTagAction()
                            } label: { Text("Add", bundle: .module) }
                            .themedFont(.tiny)
                            .buttonStyle(.plain)
                            .disabled(newTagText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(.appElevated, in: RoundedRectangle(cornerRadius: 4))
                        .overlay { RoundedRectangle(cornerRadius: 4).stroke(Color.accentColor, lineWidth: 0.5) }
                    } else {
                        Button {
                            isAddingTag = true
                        } label: {
                            Label { Text("Add Tag", bundle: .module) } icon: { Image(systemName: "plus") }
                                .themedFont(.tiny, weight: .medium)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 3)
                        }
                        .buttonStyle(.plain)
                        .background(.appElevated, in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(.appSecondary)
                    }
                }
            }

            Divider()

            // Notes section
            VStack(alignment: .leading, spacing: 6) {
                Text("Personal Notes", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)

                TextField(
                    text: $notesText,
                    prompt: Text("Add personal notes for this model (e.g., best for Swift, fast test runner)...", bundle: .module),
                    axis: .vertical
                ) {
                    Text("Personal Notes", bundle: .module)
                }
                    .textFieldStyle(.plain)
                    .themedFont(.small)
                    .appEditorSurface()
                    .accessibilityLabel(Text("Personal Notes", bundle: .module))
                    .onChange(of: notesText) { _, newValue in
                        orgStore.setNotes(newValue, for: installedModel)
                    }
            }
        }
        .padding(14)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
        .onAppear {
            syncState()
        }
        .onChange(of: installedModel.alias) {
            syncState()
        }
        .onChange(of: installedModel.path) {
            syncState()
        }
    }

    private func syncState() {
        notesText = orgStore.notes(installedModel)
        isAddingTag = false
        newTagText = ""
    }

    private func addTagAction() {
        let trimmed = newTagText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        orgStore.addTag(trimmed, for: installedModel)
        newTagText = ""
        isAddingTag = false
    }
}
