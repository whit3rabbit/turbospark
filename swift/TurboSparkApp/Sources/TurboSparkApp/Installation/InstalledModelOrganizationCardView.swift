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
                    ForEach(orgStore.tags(alias: installedModel.alias, path: installedModel.path), id: \.self) { tag in
                        HStack(spacing: 4) {
                            Text(tag)
                                .themedFont(.tiny, weight: .medium)
                            Button {
                                orgStore.removeTag(tag, for: installedModel.alias, path: installedModel.path)
                            } label: {
                                Image(systemName: "xmark")
                                    .themedFont(.micro, weight: .bold)
                            }
                            .buttonStyle(.plain)
                            .help("Remove tag \(tag)")
                            .accessibilityLabel("Remove tag \(tag)")
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color.accentColor.opacity(0.14), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(Color.accentColor)
                    }

                    if isAddingTag {
                        HStack(spacing: 4) {
                            TextField("Tag name", text: $newTagText)
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
                        .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 4))
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

                TextField("Add personal notes for this model (e.g., best for Swift, fast test runner)...", text: $notesText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .themedFont(.small)
                    .padding(8)
                    .background(.appElevated, in: RoundedRectangle(cornerRadius: 6))
                    .overlay { RoundedRectangle(cornerRadius: 6).stroke(.appBorder, lineWidth: 0.5) }
                    .onChange(of: notesText) { _, newValue in
                        orgStore.setNotes(newValue, for: installedModel.alias, path: installedModel.path)
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
        notesText = orgStore.notes(alias: installedModel.alias, path: installedModel.path)
        isAddingTag = false
        newTagText = ""
    }

    private func addTagAction() {
        let trimmed = newTagText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        orgStore.addTag(trimmed, for: installedModel.alias, path: installedModel.path)
        newTagText = ""
        isAddingTag = false
    }
}
