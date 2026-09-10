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
                    Text(model.selectedProject?.name ?? String(localized: "Projects", bundle: .module))
                        .lineLimit(1)
                    Image(systemName: "chevron.down")
                }
                .themedFont(.small, weight: .medium)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(TurboSparkTheme.surfaceColor, in: Capsule())
                .overlay(Capsule().strokeBorder(.appBorder, lineWidth: 1))
            }
            .buttonStyle(.plain)
            .disabled(navigationBlocked)
            .popover(isPresented: $isPresented, arrowEdge: .top) {
                pickerContent
            }
            if let path = model.selectedProject?.rootDirectoryPath {
                Text(path)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(path)
            }
            Spacer(minLength: 0)
        }
    }

    private var pickerContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Filter projects and tasks...", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .themedFont(.small)
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
                                        Text(path)
                                            .themedFont(.tiny)
                                            .foregroundStyle(.appSecondary)
                                            .lineLimit(1)
                                            .truncationMode(.middle)
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
                    }
                }
            }
            .frame(maxHeight: 240)
            Divider()
            Button {
                isPresented = false
                onNewProject()
            } label: {
                Label("New Project...", systemImage: "plus")
            }
            Button {
                model.chooseProjectForTask(id: nil)
                isPresented = false
            } label: {
                Label("All Chats (No Project)", systemImage: "xmark")
            }
        }
        .buttonStyle(.plain)
        .themedFont(.small)
        .padding(14)
        .frame(width: 340)
        .background(.appSurface)
        .disabled(navigationBlocked)
    }
}
