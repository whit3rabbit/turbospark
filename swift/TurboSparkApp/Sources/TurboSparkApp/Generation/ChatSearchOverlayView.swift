import SwiftUI

/// The Cmd+K "Search Chats" dialog: a centered modal card over a dimmed
/// background, listing previous chats whose title or content matches the
/// query. Matching and snippets are `ChatSearch`'s; this view only feeds it
/// the in-memory archive and renders the hits.
///
/// Mounted by `RootView`, which toggles it on `.showChatSearch` (posted by
/// the Chat menu's Cmd+K item, so the same key opens and closes it).
/// Opening a hit hands off to `selectChat`, which owns the busy-state
/// guards, so opening while a turn runs closes the dialog and changes
/// nothing else.
@MainActor
struct ChatSearchOverlayView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @Binding var isPresented: Bool

    @State private var query = ""
    @State private var documents: [ChatSearch.Document] = []
    @State private var hits: [ChatSearch.Hit] = []
    @State private var selectedIndex = 0
    @State private var rebuildTask: Task<Void, Never>?
    @FocusState private var fieldFocused: Bool

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
        .accessibilityLabel("Search chats")
        .onAppear {
            query = ""
            selectedIndex = 0
            rebuildDocuments()
            fieldFocused = true
        }
        .onDisappear { rebuildTask?.cancel() }
        .onReceive(model.$chats) { _ in
            // A turn can commit while the dialog is open. Rebuild debounced,
            // the same coalescing idea as the persist path, so a burst of
            // commits costs one rebuild.
            guard isPresented else { return }
            scheduleRebuild()
        }
        .onChange(of: query) { _, _ in recomputeHits() }
    }

    // MARK: - Layout

    private var card: some View {
        VStack(spacing: 0) {
            fieldRow
            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 0.5)
            resultsList
            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 0.5)
            footer
        }
        .frame(width: 640, height: 480)
        .background(TurboSparkTheme.surfaceColor, in: .rect(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5))
        .shadow(color: .black.opacity(0.25), radius: 24, y: 8)
        .background(hiddenShortcutButtons)
    }

    private var fieldRow: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(theme.ui(.small))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            TextField("Search chats...", text: $query)
                .textFieldStyle(.plain)
                .font(theme.ui(.callout))
                .focused($fieldFocused)
                .accessibilityLabel("Search chats")
            if !query.isEmpty {
                Button {
                    query = ""
                    fieldFocused = true
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .help("Clear search")
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }

    private var resultsList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 3) {
                    if documents.isEmpty {
                        emptyState(
                            title: "No chats yet",
                            detail: "Start a conversation to search it here")
                    } else if hits.isEmpty {
                        emptyState(
                            title: "No matching chats",
                            detail: "Try a different search term")
                    } else {
                        ForEach(Array(hits.enumerated()), id: \.element.id) { index, hit in
                            Button {
                                open(hit)
                            } label: {
                                ChatSearchResultRowView(
                                    hit: hit,
                                    isSelected: index == selectedIndex)
                            }
                            .buttonStyle(.plain)
                            .help("Open \(hit.title)")
                            .id(hit.id)
                        }
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 8)
            }
            .onChange(of: selectedIndex) { _, newValue in
                guard hits.indices.contains(newValue) else { return }
                proxy.scrollTo(hits[newValue].id, anchor: .center)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func emptyState(title: String, detail: String) -> some View {
        VStack(spacing: 6) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(theme.ui(.title2))
                .foregroundStyle(.tertiary)
                .padding(.top, 18)
                .padding(.bottom, 2)
                .accessibilityHidden(true)
            Text(title)
                .font(theme.ui(.tiny, weight: .medium))
                .foregroundStyle(.secondary)
            Text(detail)
                .font(theme.ui(.tiny))
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title). \(detail).")
    }

    private var footer: some View {
        HStack {
            Text("Up/Down to navigate, Enter to open, Esc to close", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.tertiary)
            Spacer(minLength: 0)
            Text(hits.count == documents.count
                ? "\(documents.count) chats"
                : "\(hits.count) of \(documents.count) chats")
                .font(theme.ui(.tiny))
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }

    // MARK: - Keyboard surface

    /// The arrow, Return and Escape handling. Hidden buttons rather than a
    /// key monitor because a window-level shortcut keeps working while the
    /// search field is first responder, which is exactly the palette
    /// behavior wanted here: arrows move the result selection, not the text
    /// cursor.
    private var hiddenShortcutButtons: some View {
        VStack {
            Button("Previous Result") { moveSelection(-1) }
                .keyboardShortcut(.upArrow, modifiers: [])
            Button("Next Result") { moveSelection(1) }
                .keyboardShortcut(.downArrow, modifiers: [])
            Button("Open Selected") { openSelected() }
                .keyboardShortcut(.defaultAction)
                .disabled(hits.isEmpty)
            Button("Close Search") { isPresented = false }
                .keyboardShortcut(.cancelAction)
        }
        .opacity(0)
        .frame(width: 0, height: 0)
        .accessibilityHidden(true)
    }

    // MARK: - Actions

    private func open(_ hit: ChatSearch.Hit) {
        // The sidebar list is project-scoped (`filteredChats`), so a hit
        // from another project must move the project selection FIRST or the
        // chat would open without appearing in the list. Both moves are
        // guarded on the model side: mid-turn they refuse, and the dialog
        // still closes.
        if model.selectedProjectID != hit.projectID {
            model.selectProject(id: hit.projectID)
        }
        model.selectChat(id: hit.id)
        isPresented = false
    }

    private func openSelected() {
        guard hits.indices.contains(selectedIndex) else { return }
        open(hits[selectedIndex])
    }

    private func moveSelection(_ direction: Int) {
        guard !hits.isEmpty else { return }
        selectedIndex = max(0, min(hits.count - 1, selectedIndex + direction))
    }

    // MARK: - Index maintenance

    private func rebuildDocuments() {
        documents = ChatSearch.buildDocuments(from: model.chats)
        recomputeHits()
    }

    private func recomputeHits() {
        hits = ChatSearch.hits(query: .parse(query), documents: documents)
        if !hits.indices.contains(selectedIndex) {
            selectedIndex = max(0, hits.count - 1)
        }
    }

    private func scheduleRebuild() {
        rebuildTask?.cancel()
        rebuildTask = Task {
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            rebuildDocuments()
        }
    }
}
