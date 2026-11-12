import SwiftUI

/// Keeps the execution workspace visible at the point where a task is composed.
@MainActor
struct ProjectPickerView: View {
    @ObservedObject var model: AppModel
    let onNewProject: () -> Void
    @State private var isPresented = false
    @State private var searchText = ""

    private var navigationBlocked: Bool {
        model.isRunning || model.submitting || model.pendingToolCall != nil
    }

    var body: some View {
        HStack(spacing: 8) {
            Button {
                searchText = ""
                isPresented = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "folder")
                    Group {
                        if let name = model.selectedProject?.name {
                            Text(name)
                        } else {
                            Text("Projects", bundle: .module)
                        }
                    }
                    .lineLimit(1)
                    Image(systemName: "chevron.down")
                }
                .themedFont(.small, weight: .medium)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.appSurface, in: Capsule())
                .overlay(Capsule().strokeBorder(.appBorder, lineWidth: 1))
            }
            .buttonStyle(.plain)
            .disabled(navigationBlocked)
            .help(
                navigationBlocked
                    ? Text("Cannot save project while generation or tool execution is active", bundle: .module)
                    : Text("Projects", bundle: .module)
            )
            .popover(isPresented: $isPresented, arrowEdge: .top) {
                pickerContent
            }
            if let path = model.selectedProject?.rootDirectoryPath {
                SettingsPathText(path, size: .tiny)
            }
            Spacer(minLength: 0)
        }
    }

    private var pickerContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            AppSearchField(
                text: $searchText,
                prompt: Text("Filter projects and tasks...", bundle: .module),
                maxWidth: .infinity)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(model.projects.filter {
                        searchText.isEmpty || $0.name.localizedCaseInsensitiveContains(searchText)
                            || ($0.rootDirectoryPath?.localizedCaseInsensitiveContains(searchText) ?? false)
                    }) { project in
                        Button {
                            model.chooseProjectForTask(id: project.id)
                            isPresented = false
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "folder")
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(project.name).lineLimit(1)
                                    if let path = project.rootDirectoryPath {
                                        SettingsPathText(path, size: .tiny)
                                    }
                                }
                                Spacer(minLength: 0)
                                if model.selectedProjectID == project.id {
                                    Image(systemName: "checkmark")
                                }
                            }
                            .padding(8)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityAddTraits(model.selectedProjectID == project.id ? [.isButton, .isSelected] : .isButton)
                    }
                }
            }
            .frame(maxHeight: 240)
            Divider()
            Button {
                isPresented = false
                onNewProject()
            } label: {
                Label { Text("New Project...", bundle: .module) } icon: { Image(systemName: "plus") }
            }
            Button {
                model.chooseProjectForTask(id: nil)
                isPresented = false
            } label: {
                Label { Text("All Chats (No Project)", bundle: .module) } icon: { Image(systemName: "xmark") }
            }
        }
        .buttonStyle(.plain)
        .themedFont(.small)
        .padding(14)
        .frame(minWidth: 320, idealWidth: 340)
        .background(.appSurface)
        .disabled(navigationBlocked)
    }
}
