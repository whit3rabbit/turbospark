import SwiftUI

/// The Cmd+K command center: a centered modal card over a dimmed
/// background that searches commands, chats and project files from one
/// field, with scope tabs (All, Commands, Chats, Files) and first-character
/// prefixes (`>`, `#`, `@`). It replaces the chat-only overlay: chat
/// matching, snippets and ordering are still `ChatSearch`'s, and opening a
/// chat hit goes through the same `openChat`.
///
/// Mounted by `RootView`, which toggles it on `.showChatSearch` (posted by
/// Cmd+K and by `/resume`; Cmd+Shift+P posts it with a `>` seed). The
/// providers and the action routing are pure (`PaletteProviders.swift`); this
/// view only feeds them in-memory state and renders the rows.
@MainActor
struct CommandPaletteView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @Binding var isPresented: Bool
    /// Text the field starts with (`>` from Cmd+Shift+P, empty from Cmd+K).
    let initialQuery: String
    /// Bumps when a seeded open arrives while the palette is already open.
    let seedToken: Int

    @State private var rawQuery = ""
    @State private var tab: PaletteScope = .all
    @State private var context = PaletteContext()
    @State private var sections: [PaletteSection] = []
    @State private var selectedIndex = 0
    @State private var rebuildTask: Task<Void, Never>?
    @FocusState private var fieldFocused: Bool

    private var query: PaletteQuery { .parse(rawQuery, tab: tab) }
    private var rows: [PaletteRow] { sections.flatMap(\.rows) }
    private var workspaceKey: String {
        model.selectedProjectID?.uuidString ?? PaletteHistory.globalKey
    }

    var body: some View {
        ZStack {
            Color.black.opacity(0.32)
                .ignoresSafeArea()
                .contentShape(.rect)
                .onTapGesture { isPresented = false }
                .accessibilityHidden(true)
            card
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Command Center", bundle: .module))
        .onAppear {
            rawQuery = initialQuery
            tab = .all
            // Everything below reads memory: the chat archive, the skill
            // list the composer already resolves per keystroke, and the file
            // index's CACHED listing. The walk itself runs in the
            // background and lands in `refreshFiles`.
            rebuildContext()
            fieldFocused = true
            refreshFiles()
        }
        .onDisappear { rebuildTask?.cancel() }
        .onChange(of: seedToken) { _, _ in
            rawQuery = initialQuery
            fieldFocused = true
        }
        .onReceive(model.$chats) { _ in
            // A turn can commit while the palette is open. Rebuild debounced
            // so a burst of commits costs one rebuild.
            guard isPresented else { return }
            scheduleRebuild()
        }
        .onChange(of: rawQuery) { _, _ in
            recompute(resetSelection: true)
        }
        .onChange(of: tab) { _, newTab in
            recompute(resetSelection: true)
            _ = AccessibilityNotification.Announcement.post(
                .init(AppStrings.localized(String.LocalizationValue(newTab.titleKey))))
        }
    }

    // MARK: - Layout

    private var card: some View {
        VStack(spacing: 0) {
            fieldRow
            scopePicker
            Rectangle()
                .fill(.appBorder)
                .frame(height: 0.5)
            resultsList
            Rectangle()
                .fill(.appBorder)
                .frame(height: 0.5)
            footer
        }
        .frame(minWidth: 500, idealWidth: 640, maxWidth: 800, minHeight: 380, idealHeight: 480, maxHeight: 650)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 0.5))
        .shadow(color: .black.opacity(0.25), radius: 24, y: 8)
        .background(hiddenShortcutButtons)
    }

    private var fieldRow: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            TextField(
                text: $rawQuery,
                prompt: Text("Search commands, chats, and files (> # @)...", bundle: .module)
            ) {
                Text("Command Center", bundle: .module)
            }
            .textFieldStyle(.plain)
            .font(theme.ui(.callout))
            .focused($fieldFocused)
            .accessibilityHint(Text("Searches commands, chats, and files", bundle: .module))
            if !rawQuery.isEmpty {
                Button {
                    rawQuery = ""
                    fieldFocused = true
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)
                .help(Text("Clear search", bundle: .module))
                .accessibilityLabel(Text("Clear search", bundle: .module))
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }

    /// The scope tabs. A prefix in the field overrides the selection, and
    /// the picker shows the EFFECTIVE scope so the two never disagree.
    private var scopePicker: some View {
        Picker(selection: scopeBinding) {
            ForEach(PaletteScope.allCases) { scope in
                Text(LocalizedStringKey(scope.titleKey), bundle: .module).tag(scope)
            }
        } label: {
            Text("Command Center", bundle: .module)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .padding(.horizontal, 14)
        .padding(.bottom, 10)
    }

    /// Choosing a tab drops a prefix (the user is overriding it), then sets
    /// the tab.
    private var scopeBinding: Binding<PaletteScope> {
        Binding(
            get: { query.scope },
            set: { newScope in
                if query.isPrefixed {
                    rawQuery = String(rawQuery.dropFirst()).trimmingCharacters(in: .whitespaces)
                }
                tab = newScope
            })
    }

    private var resultsList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 3) {
                    if sections.isEmpty {
                        emptyState
                    } else {
                        ForEach(sections) { section in
                            sectionHeader(section)
                            ForEach(section.rows) { row in
                                let index = flatIndex(of: row)
                                Button {
                                    selectedIndex = index
                                    perform(row)
                                } label: {
                                    rowView(row, isSelected: index == selectedIndex)
                                        .background(
                                            RoundedRectangle(
                                                cornerRadius: AppMetrics.cardRadius, style: .continuous
                                            )
                                            .fill(index == selectedIndex ? theme.selection : Color.clear))
                                }
                                .buttonStyle(.plain)
                                .id(row.id)
                            }
                        }
                    }
                    if query.scope != .chats, context.files == nil, context.hasProject,
                        query.scope == .files || !query.text.isEmpty
                    {
                        Text("Scanning project files...", bundle: .module)
                            .font(theme.ui(.tiny))
                            .foregroundStyle(.appSecondary)
                            .padding(.horizontal, 10)
                            .padding(.top, 6)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 8)
            }
            .onChange(of: selectedIndex) { _, newValue in
                guard rows.indices.contains(newValue) else { return }
                proxy.scrollTo(rows[newValue].id, anchor: .center)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var emptyState: some View {
        if query.scope == .files, !context.hasProject {
            AppEmptyState(
                systemImage: "folder",
                title: Text("Open a project to search its files", bundle: .module))
        } else if query.scope == .chats, context.documents.isEmpty {
            AppEmptyState(
                systemImage: "bubble.left.and.bubble.right",
                title: Text("No chats yet", bundle: .module),
                message: Text("Start a conversation to search it here", bundle: .module))
        } else {
            AppEmptyState(
                systemImage: "magnifyingglass",
                title: Text("No matching results", bundle: .module),
                message: Text("Try a different search term", bundle: .module))
        }
    }

    private func sectionHeader(_ section: PaletteSection) -> some View {
        HStack {
            Text(LocalizedStringKey(section.titleKey), bundle: .module)
                .font(theme.ui(.tiny, weight: .semibold))
                .foregroundStyle(.appSecondary)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 0)
            if section.kind == .history {
                Button {
                    PaletteHistoryStore.shared.clear(workspace: workspaceKey)
                    rebuildContext()
                } label: {
                    Text("Clear recent searches", bundle: .module)
                        .font(theme.ui(.tiny))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.appAccentText)
            }
        }
        .padding(.horizontal, 10)
        .padding(.top, 6)
    }

    @ViewBuilder
    private func rowView(_ row: PaletteRow, isSelected: Bool) -> some View {
        switch row.payload {
        case .chat(let hit):
            ChatSearchResultRowView(hit: hit, isSelected: isSelected)
                .help(Text("Open \(hit.title)", bundle: .module))
        case .command(let command):
            PaletteRowView(
                iconName: command.iconName,
                title: command.titleIsCatalogKey
                    ? Text(LocalizedStringKey(command.title), bundle: .module)
                    : Text(verbatim: command.title),
                subtitle: command.subtitle.map { subtitle in
                    command.subtitleIsCatalogKey
                        ? Text(LocalizedStringKey(subtitle), bundle: .module)
                        : Text(verbatim: subtitle)
                },
                trailing: command.binding,
                kindLabel: Text("Command", bundle: .module),
                isSelected: isSelected)
        case .file(let entry):
            let name = (entry.relativePath as NSString).lastPathComponent
            let directory = (entry.relativePath as NSString).deletingLastPathComponent
            PaletteRowView(
                iconName: entry.isDirectory ? "folder.fill" : "doc",
                title: Text(verbatim: name),
                subtitle: directory.isEmpty ? nil : Text(verbatim: directory),
                trailing: nil,
                kindLabel: entry.isDirectory
                    ? Text("Folder", bundle: .module) : Text("File", bundle: .module),
                isSelected: isSelected)
        case .history(let text):
            PaletteRowView(
                iconName: "clock.arrow.circlepath",
                title: Text(verbatim: text),
                subtitle: nil,
                trailing: nil,
                kindLabel: Text("Recent searches", bundle: .module),
                isSelected: isSelected)
        }
    }

    private var footer: some View {
        HStack {
            Text("Up/Down to navigate, Enter to select, Tab to switch scope, Esc to close", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }

    // MARK: - Keyboard surface

    /// The arrow, Tab, Return and Escape handling. Hidden buttons rather
    /// than a key monitor because a window-level shortcut keeps working
    /// while the search field is first responder: arrows move the result
    /// selection, not the text cursor, and Tab switches scope instead of
    /// moving focus off the field.
    private var hiddenShortcutButtons: some View {
        VStack {
            Button { moveSelection(-1) } label: { Text("Previous Result", bundle: .module) }
                .keyboardShortcut(.upArrow, modifiers: [])
            Button { moveSelection(1) } label: { Text("Next Result", bundle: .module) }
                .keyboardShortcut(.downArrow, modifiers: [])
            Button { switchScope(1) } label: { Text("Command Center", bundle: .module) }
                .keyboardShortcut(.tab, modifiers: [])
            Button { switchScope(-1) } label: { Text("Command Center", bundle: .module) }
                .keyboardShortcut(.tab, modifiers: .shift)
            Button { performSelected() } label: { Text("Open Selected", bundle: .module) }
                .keyboardShortcut(.defaultAction)
                .disabled(rows.isEmpty)
            Button { isPresented = false } label: { Text("Close Search", bundle: .module) }
                .keyboardShortcut(.cancelAction)
        }
        .opacity(0)
        .frame(width: 0, height: 0)
        .accessibilityHidden(true)
    }

    // MARK: - Actions

    private func flatIndex(of row: PaletteRow) -> Int {
        rows.firstIndex(where: { $0.id == row.id }) ?? 0
    }

    private func performSelected() {
        guard rows.indices.contains(selectedIndex) else { return }
        perform(rows[selectedIndex])
    }

    private func perform(_ row: PaletteRow) {
        let action = PaletteActionRouter.action(for: row)
        // History records what the user typed, once they act on a result;
        // choosing a history row itself only refills the field.
        if case .setQuery(let text) = action {
            rawQuery = text
            fieldFocused = true
            return
        }
        PaletteHistoryStore.shared.record(rawQuery, workspace: workspaceKey)
        PaletteActionRouter.perform(action, model: model)
        isPresented = false
    }

    private func moveSelection(_ direction: Int) {
        guard !rows.isEmpty else { return }
        selectedIndex = max(0, min(rows.count - 1, selectedIndex + direction))
    }

    private func switchScope(_ delta: Int) {
        let target = query.scope.advanced(by: delta)
        if query.isPrefixed {
            rawQuery = String(rawQuery.dropFirst()).trimmingCharacters(in: .whitespaces)
        }
        tab = target
    }

    // MARK: - Index maintenance

    private func rebuildContext() {
        context.documents = ChatSearch.buildDocuments(from: model.chats)
        context.commands = PaletteCommandCatalog.commands(
            skills: model.effectiveSkills,
            canCheckForUpdates: SparkleUpdateController.shared.canCheckForUpdates)
        context.history = PaletteHistoryStore.shared.entries(workspace: workspaceKey)
        context.hasProject = model.selectedProject?.rootDirectoryURL != nil
        if let root = model.selectedProject?.rootDirectoryURL {
            context.files = ProjectFileIndex.shared.cachedEntries(for: root)
        } else {
            context.files = nil
        }
        recompute(resetSelection: false)
    }

    /// Reads the file listing in the background (a fresh cache returns at
    /// once; a stale or cold one runs the bounded walk off the main thread).
    private func refreshFiles() {
        guard let root = model.selectedProject?.rootDirectoryURL else { return }
        Task {
            let entries = await ProjectFileIndex.shared.entries(for: root)
            guard isPresented else { return }
            context.files = entries
            recompute(resetSelection: false)
        }
    }

    private func recompute(resetSelection: Bool) {
        sections = PaletteEngine.sections(query: query, context: context)
        if resetSelection {
            // A new query starts at the best (first) row.
            selectedIndex = 0
        } else if !rows.indices.contains(selectedIndex) {
            selectedIndex = max(0, rows.count - 1)
        }
    }

    private func scheduleRebuild() {
        rebuildTask?.cancel()
        rebuildTask = Task {
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            rebuildContext()
        }
    }
}

/// One non-chat result row: icon, title, optional subtitle, optional
/// trailing chord. The selection highlight mirrors `ChatSearchResultRowView`.
@MainActor
private struct PaletteRowView: View {
    @Environment(\.appTheme) private var theme
    let iconName: String
    let title: Text
    let subtitle: Text?
    let trailing: String?
    /// Read first by VoiceOver so a row says what KIND of result it is.
    let kindLabel: Text
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: iconName)
                .font(theme.ui(.tiny))
                .foregroundStyle(isSelected ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appSecondary))
                .frame(width: 16)
                .accessibilityHidden(true)
            title
                .font(theme.ui(.small, weight: isSelected ? .semibold : .medium))
                .foregroundStyle(.appText)
                .lineLimit(1)
            if let subtitle {
                subtitle
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            if let trailing {
                Text(verbatim: trailing)
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.appSecondary)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(.rect)
        .accessibilityElement(children: .combine)
        .accessibilityHint(kindLabel)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

/// Executes a `PaletteAction` against the app. Kept apart from the pure
/// routing so the router stays testable without a model.
extension PaletteActionRouter {
    @MainActor
    static func perform(_ action: PaletteAction, model: AppModel) {
        switch action {
        case .openChat(let id):
            // `openChat` owns the busy-state guards and the project move:
            // mid-turn it refuses and the palette still closes.
            model.openChat(id: id)
        case .runLocalCommand(let draft):
            model.handleLocalCommand(draft)
        case .runLocalMetaCommand(let draft):
            model.handleLocalMetaCommand(draft)
        case .insertIntoComposer(let text):
            // The draft is model state, so the insert works even when the
            // composer is not mounted (another section, a collapsed pane):
            // go to Chat first, then write, then ask for focus.
            if model.activeSection != .chat { model.activeSection = .chat }
            model.writePromptTextDirectly(PaletteDraft.appending(text, to: model.promptText))
            NotificationCenter.default.post(name: .focusPrompt, object: nil)
        case .openSettings(let tab):
            model.openSettings(tab: tab)
        case .navigate(let section):
            model.activeSection = section
        case .checkForUpdates:
            SparkleUpdateController.shared.checkForUpdates()
        case .setQuery:
            break
        }
    }
}
