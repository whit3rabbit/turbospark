import AppKit
import SwiftUI

struct ChatSidebarView: View {
    @ObservedObject var model: AppModel
    @AppStorage(AppAppearance.storageKey)
    private var appearanceRawValue = AppAppearance.system.rawValue
    @AppStorage(AppTextSize.storageKey)
    private var textSizeRawValue = AppTextSize.standard.rawValue
    @AppStorage(AppLanguage.storageKey)
    private var languageRawValue = AppLanguage.system.rawValue

    @State private var hoveredChatID: UUID?
    @State private var hoveredProjectID: UUID?
    @State private var chatBeingRenamed: AppChat?
    @State private var renameText = ""
    @State private var chatPendingDeletion: AppChat?
    @State private var showingProjectSettingsSheet = false
    @State private var projectBeingEdited: AppProject?

    @ScaledMetric private var actionButtonSize: CGFloat = 26
    @ScaledMetric private var projectActionSize: CGFloat = 22
    @ScaledMetric private var sidebarFooterMinHeight: CGFloat = 42



    var body: some View {
        VStack(spacing: 0) {
            header
            sectionNav
            Divider()
            newChatButton
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            Divider()
            projectsSection
            Divider()
            chatList
            Divider()
            footer
        }
        .sheet(isPresented: $showingProjectSettingsSheet) {
            ProjectSettingsSheet(
                model: model,
                editingProject: projectBeingEdited,
                onDismiss: {
                    showingProjectSettingsSheet = false
                    projectBeingEdited = nil
                }
            )
        }
        .alert(
            "Rename chat",
            isPresented: renameAlertPresented,
            presenting: chatBeingRenamed
        ) { chat in
            TextField("Chat name", text: $renameText)
            Button("Cancel", role: .cancel) {}
            Button("Rename") {
                model.renameChat(id: chat.id, title: renameText)
            }
            .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        } message: { _ in
            Text("Choose a name that identifies this chat.")
        }
        .alert(
            "Delete chat?",
            isPresented: deletionAlertPresented,
            presenting: chatPendingDeletion
        ) { chat in
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                model.deleteChat(id: chat.id)
            }
        } message: { chat in
            Text("“\(chat.title)” and its conversation history will be removed.")
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "bolt.horizontal.circle.fill")
                .font(.title3)
                .foregroundStyle(TurboSparkTheme.accentColor)
                .accessibilityHidden(true)
            Text("TurboSpark")
                .font(.headline)
                .lineLimit(1)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.top, 38)
        .padding(.bottom, 12)
    }

    private var sectionNav: some View {
        VStack(spacing: 4) {
            navButton(
                title: "Chat",
                systemImage: "bubble.left.and.bubble.right.fill",
                section: .chat
            )

            navButton(
                title: "Model hub",
                systemImage: "square.grid.2x2.fill",
                section: .modelHub,
                badge: "\(model.installed.count)"
            )
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
    }

    private func navButton(
        title: String,
        systemImage: String,
        section: AppModel.AppNavigationSection,
        badge: String? = nil
    ) -> some View {
        let isSelected = model.activeSection == section
        return Button {
            model.activeSection = section
        } label: {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                    .font(.callout)
                    .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : .secondary)
                    .frame(width: 20)
                    .accessibilityHidden(true)
                Text(title)
                    .font(.callout.weight(isSelected ? .semibold : .medium))
                    .foregroundStyle(isSelected ? .primary : .secondary)
                Spacer()
                if let badge {
                    Text(badge)
                        .font(.caption2.weight(.bold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1.5)
                        .background(
                            isSelected ? TurboSparkTheme.accentColor.opacity(0.2) : Color(nsColor: .quaternaryLabelColor).opacity(0.3),
                            in: Capsule()
                        )
                        .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : .secondary)
                }
            }
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, minHeight: 32)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .background(
            isSelected ? Color(nsColor: .controlBackgroundColor) : Color.clear,
            in: RoundedRectangle(cornerRadius: 8)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? Color(nsColor: .separatorColor).opacity(0.5) : Color.clear, lineWidth: 0.5)
        )
        .accessibilityLabel(title)
        // Tell VoiceOver which sidebar is the current page so users get the
        // same selection feedback they get visually from the accent color.
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityHint("Switches the main pane to \(title)")
    }

    private var newChatButton: some View {
        Button {
            model.createChat()
        } label: {
            HStack(spacing: 10) {
                Image(systemName: "square.and.pencil")
                    .accessibilityHidden(true)
                Text("New chat")
                    .fontWeight(.medium)
                Spacer()
                Text("⌘N")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 12)
            .frame(maxWidth: .infinity, minHeight: 42)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .background(
            Color(nsColor: .controlBackgroundColor),
            in: .rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .stroke(.separator.opacity(0.45), lineWidth: 0.5)
        }
        .disabled(model.isRunning)
        .help("Create a new chat (⌘N)")
        .accessibilityHint("Starts a fresh conversation. Disabled while the model is generating.")
    }

    private var projectsSection: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text("Projects")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Button {
                    projectBeingEdited = nil
                    showingProjectSettingsSheet = true
                } label: {
                    Label("Add Project", systemImage: "plus")
                        .labelStyle(.iconOnly)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.borderless)
                .help("Add codebase project")
                .accessibilityLabel("Add project")
                .accessibilityHint("Opens project settings to create a new project")
            }
            .padding(.horizontal, 10)
            .padding(.top, 8)
            .padding(.bottom, 2)

            allChatsRow

            ForEach(model.projects) { project in
                projectRow(project)
            }
        }
        .padding(.horizontal, 8)
        .padding(.bottom, 6)
    }

    private var allChatsRow: some View {
        let isSelected = model.selectedProjectID == nil
        return Button {
            model.selectProject(id: nil)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: isSelected ? "folder.fill" : "folder")
                    .font(.caption)
                    .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                    .accessibilityHidden(true)
                Text("All Chats")
                    .font(.callout.weight(isSelected ? .semibold : .regular))
                    .foregroundStyle(.primary)
                Spacer()
                Text("\(model.chats.count)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("\(model.chats.count) chats")
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 6)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
            in: .rect(cornerRadius: 8)
        )
        .accessibilityLabel("All chats")
        .accessibilityValue("\(model.chats.count) total")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityHint("Shows every chat regardless of project")
    }

    private func projectRow(_ project: AppProject) -> some View {
        let isSelected = model.selectedProjectID == project.id
        let showsActions = hoveredProjectID == project.id || isSelected

        return HStack(spacing: 4) {
            Button {
                model.selectProject(id: project.id)
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: isSelected ? "folder.fill.badge.gearshape" : "folder.badge.gearshape")
                        .font(.caption)
                        .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(project.name)
                            .font(.callout.weight(isSelected ? .semibold : .regular))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        Text(project.agentType.label)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 0)
                }
                .padding(.leading, 9)
                .padding(.vertical, 6)
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Project \(project.name)")
            .accessibilityValue("\(project.agentType.label)\(isSelected ? ", selected" : "")")
            .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)

            Menu {
                Button("Project Settings", systemImage: "gearshape") {
                    projectBeingEdited = project
                    showingProjectSettingsSheet = true
                }
                if let path = project.rootDirectoryPath {
                    Button("Reveal in Finder", systemImage: "folder") {
                        let url = URL(fileURLWithPath: path)
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                }
                Divider()
                Button("Delete Project", systemImage: "trash", role: .destructive) {
                    model.deleteProject(id: project.id)
                }
            } label: {
                Label("Project actions", systemImage: "ellipsis")
                    .labelStyle(.iconOnly)
                    .frame(width: projectActionSize, height: projectActionSize)
                    .contentShape(Circle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            // Always reachable from VoiceOver, even though sighted users
            // only see the button when hovering or when the row is selected.
            .accessibilityLabel("Project actions for \(project.name)")
            .opacity(showsActions ? 1 : 0)
            .padding(.trailing, 4)
        }
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
            in: .rect(cornerRadius: 8)
        )
        .onHover { isHovering in
            hoveredProjectID = isHovering ? project.id : nil
        }
    }

    private var chatList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 3) {
                Text(model.selectedProject != nil ? "\(model.selectedProject!.name) Chats" : "Chats")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.top, 10)
                    .padding(.bottom, 3)
                    .accessibilityAddTraits(.isHeader)

                if historyChats.isEmpty {
                    VStack(spacing: 6) {
                        Image(systemName: "bubble.left.and.bubble.right")
                            .font(.system(size: 20))
                            .foregroundStyle(.tertiary)
                            .padding(.top, 18)
                            .padding(.bottom, 2)
                            .accessibilityHidden(true)
                        Text("No chats yet")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.secondary)
                        Text("Start a conversation to see history here")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .padding(.horizontal, 12)
                    // Read the empty state as a single block rather than
                    // three sequential announcements.
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("No chats yet. Start a conversation to see history here.")
                } else {
                    ForEach(historyChats) { chat in
                        chatRow(chat)
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func chatRow(_ chat: AppChat) -> some View {
        let isSelected = chat.id == model.selectedChatID
        let showsActions = hoveredChatID == chat.id || isSelected

        return HStack(spacing: 4) {
            Button {
                model.selectChat(id: chat.id)
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: isSelected ? "bubble.left.fill" : "bubble.left")
                        .font(.caption)
                        .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(chat.title)
                            .font(.callout.weight(isSelected ? .semibold : .regular))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        if !chat.preview.isEmpty {
                            Text(chat.preview)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    Spacer(minLength: 0)
                }
                .padding(.leading, 9)
                .padding(.vertical, 8)
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .disabled(model.isRunning && !isSelected)
            .accessibilityLabel(chat.title)
            .accessibilityValue(chat.preview.isEmpty
                                ? (isSelected ? "Selected" : "")
                                : "\(chat.preview)\(isSelected ? ", selected" : "")")
            .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)

            Menu {
                Button("Rename", systemImage: "pencil") {
                    renameText = chat.title
                    chatBeingRenamed = chat
                }
                Divider()
                Button("Delete", systemImage: "trash", role: .destructive) {
                    chatPendingDeletion = chat
                }
            } label: {
                Label("Chat actions", systemImage: "ellipsis")
                    .labelStyle(.iconOnly)
                    .frame(width: actionButtonSize, height: actionButtonSize)
                    .contentShape(Circle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            // Always reachable from VoiceOver, even though sighted users
            // only see the button when hovering or when the row is selected.
            .accessibilityLabel("Chat actions for \(chat.title)")
            .opacity(showsActions ? 1 : 0)
            .disabled(model.isRunning)
            .padding(.trailing, 5)
        }
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
            in: .rect(cornerRadius: 9))
        .contentShape(.rect(cornerRadius: 9))
        .onHover { isHovering in
            hoveredChatID = isHovering ? chat.id : nil
        }
        .contextMenu {
            Button("Rename") {
                renameText = chat.title
                chatBeingRenamed = chat
            }
            .disabled(model.isRunning)
            Button("Delete", role: .destructive) {
                chatPendingDeletion = chat
            }
            .disabled(model.isRunning)
        }
    }

    private var footer: some View {
        let count = historyChats.count
        return HStack(spacing: 7) {
            Image(systemName: "internaldrive")
                .accessibilityHidden(true)
            Text("\(count) local chats", bundle: .module)
            Spacer()
            appearanceMenu
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .frame(minHeight: sidebarFooterMinHeight)
    }


    private var appearanceMenu: some View {
        let appearance = AppAppearance.resolve(appearanceRawValue)
        return Menu {
            Picker("Appearance", selection: $appearanceRawValue) {
                ForEach(AppAppearance.allCases) { option in
                    Label(option.label, systemImage: option.systemImage)
                        .tag(option.rawValue)
                }
            }

            Divider()

            Picker("Text Size", selection: $textSizeRawValue) {
                ForEach(AppTextSize.allCases) { size in
                    Text(size.label)
                        .tag(size.rawValue)
                }
            }

            Divider()

            Picker("Language", selection: $languageRawValue) {
                ForEach(AppLanguage.allCases) { language in
                    Text(language.label)
                        .tag(language.rawValue)
                }
            }
        } label: {
            Label("Display & Language Settings", systemImage: appearance.systemImage)
                .labelStyle(.iconOnly)
                .frame(width: actionButtonSize, height: actionButtonSize)
                .contentShape(Circle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Appearance & Language Settings")
        .accessibilityLabel("Display and Language Settings")
        .accessibilityHint("Change theme, text size, and language")
    }


    private var historyChats: [AppChat] {
        sortedChats.filter { !$0.messages.isEmpty }
    }

    private var sortedChats: [AppChat] {
        model.filteredChats.sorted { lhs, rhs in
            lhs.updatedAt > rhs.updatedAt
        }
    }

    private var renameAlertPresented: Binding<Bool> {
        Binding(
            get: { chatBeingRenamed != nil },
            set: { if !$0 { chatBeingRenamed = nil } })
    }

    private var deletionAlertPresented: Binding<Bool> {
        Binding(
            get: { chatPendingDeletion != nil },
            set: { if !$0 { chatPendingDeletion = nil } })
    }
}
