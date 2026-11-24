import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The model loader's chooser: a filter field over the installed models and
/// the enabled remote provider models, then the library actions the old menu
/// carried.
///
/// A popover rather than a `Menu` because a native menu cannot hold a text
/// field, and the per-row facts (size on disk, memory fit) need a layout a
/// menu item flattens. Filtering, ranking and the keyboard highlight live in
/// `ModelPickerList` so they are tested without a view.
///
/// Choosing does exactly what the menu did: a local row calls
/// `openChatWithModel`, a remote row calls `requestRemoteModel`, which raises
/// the composer Remote pill's "content leaves this Mac" confirmation. Nothing
/// here selects a remote model without that confirmation.
@MainActor
struct ModelPickerPopover: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var remote: RemoteProviderStore
    @Binding var isPresented: Bool

    @State private var query = ""
    @State private var highlight: Int?
    /// This Mac's fit verdict per alias, from the same cached recommend pass
    /// the installed-model pane reads. Empty until it loads; a row without a
    /// verdict shows no fit fact rather than a guess.
    @State private var fits: [String: ModelRecommendation.FitVerdict] = [:]
    @FocusState private var searchFocused: Bool

    private var localRows: [ModelPickerRow] {
        let selected = model.selected?.alias
        return model.loadableModels.map { ModelPickerRow.local($0, selectedAlias: selected) }
    }

    private var remoteRows: [ModelPickerRow] {
        let current = model.remoteSelections[model.selectedChatID]
        return remote.pickerEntries().map { entry in
            ModelPickerRow.remote(
                providerID: entry.provider.id, providerName: entry.provider.name,
                modelID: entry.model.id, displayName: entry.model.displayName,
                selected: current)
        }
    }

    var body: some View {
        let local = localRows
        let remoteList = remoteRows
        let sections = ModelPickerList.sections(local: local, remote: remoteList, query: query)
        let flat = ModelPickerList.flattened(sections)
        let hasAnyRow = !local.isEmpty || !remoteList.isEmpty

        VStack(alignment: .leading, spacing: 8) {
            if hasAnyRow {
                AppSearchField(
                    text: $query,
                    prompt: Text("Filter models...", bundle: .module),
                    maxWidth: .infinity,
                    focus: $searchFocused)
                    .onSubmit { choose(at: highlight, in: flat) }
                    .onKeyPress(.downArrow) { move(by: 1, count: flat.count) }
                    .onKeyPress(.upArrow) { move(by: -1, count: flat.count) }
                    .onKeyPress(.escape) {
                        isPresented = false
                        return .handled
                    }
            }

            if !hasAnyRow {
                emptyLibrary
            } else if sections.isEmpty {
                Text("Nothing matches \u{201C}\(query.trimmingCharacters(in: .whitespaces))\u{201D}.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .padding(.vertical, 8)
            } else {
                rowList(sections: sections, flat: flat)
            }

            Divider()
            actions
        }
        .buttonStyle(.plain)
        .themedFont(.small)
        .padding(12)
        .frame(minWidth: 340, idealWidth: 380)
        .background(.appSurface)
        .onAppear {
            query = ""
            highlight = ModelPickerList.initialHighlight(flat)
            searchFocused = true
        }
        .onChange(of: query) { _, fresh in
            let rows = ModelPickerList.flattened(
                ModelPickerList.sections(local: localRows, remote: remoteRows, query: fresh))
            // A typed query highlights the best match; clearing it returns to
            // the current model, the same start the popover opened with.
            highlight = fresh.trimmingCharacters(in: .whitespaces).isEmpty
                ? ModelPickerList.initialHighlight(rows)
                : (rows.isEmpty ? nil : 0)
        }
        .task(id: model.fitRecommendationConfigurationID) {
            guard let rows = try? await model.loadFitRecommendations(probeIfNeeded: false),
                  !Task.isCancelled else { return }
            fits = Dictionary(rows.map { ($0.alias, $0.verdict) }, uniquingKeysWith: { first, _ in first })
        }
    }

    // MARK: - Rows

    private func rowList(sections: [ModelPickerSection], flat: [ModelPickerRow]) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(sections) { section in
                        sectionHeader(section.group)
                        ForEach(section.rows) { row in
                            let index = flat.firstIndex(of: row)
                            rowButton(row, highlighted: index != nil && index == highlight)
                                .id(row.id)
                                .onHover { inside in if inside { highlight = index } }
                        }
                    }
                }
            }
            .frame(maxHeight: 300)
            .onChange(of: highlight) { _, fresh in
                guard let fresh, flat.indices.contains(fresh) else { return }
                proxy.scrollTo(flat[fresh].id)
            }
        }
    }

    private func sectionHeader(_ group: ModelPickerSection.Group) -> some View {
        Group {
            switch group {
            case .local: Text("Installed", bundle: .module)
            case .remote: Text("Remote (content leaves this Mac)", bundle: .module)
            }
        }
        .themedFont(.tiny, weight: .semibold)
        .foregroundStyle(.appSecondary)
        .padding(.horizontal, 8)
        .padding(.top, 6)
        .padding(.bottom, 2)
        .accessibilityAddTraits(.isHeader)
    }

    private func rowButton(_ row: ModelPickerRow, highlighted: Bool) -> some View {
        Button {
            choose(row)
        } label: {
            HStack(spacing: 8) {
                // A fixed slot so the titles align whether or not a row is
                // the current one.
                Image(systemName: "checkmark")
                    .themedFont(.tiny, weight: .bold)
                    .foregroundStyle(.appAccentText)
                    .opacity(row.isSelected ? 1 : 0)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 1) {
                    // Alias, family and quant are identifiers: nothing to translate.
                    Text(verbatim: row.title)
                        .themedFont(.small, weight: .semibold)
                        .foregroundStyle(.appText)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    facts(row)
                }

                Spacer(minLength: 6)

                if let alias = row.fitAlias, let verdict = fits[alias] {
                    // The app's one verdict pill: icon plus words in the
                    // theme's status colors, so the fit never rests on color.
                    ModelFitVerdictPill(verdict: verdict, compact: true)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                highlighted ? theme.accent.opacity(0.14) : Color.clear,
                in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .contentShape(Rectangle())
        }
        .disabled(isRemote(row) && model.isRunningInSelectedChat)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(row.isSelected ? [.isButton, .isSelected] : .isButton)
    }

    @ViewBuilder
    private func facts(_ row: ModelPickerRow) -> some View {
        let parts = [row.detail, row.quant, row.installBytes.map(MetricFormat.storage)].compactMap { $0 }
        if !parts.isEmpty {
            HStack(spacing: 6) {
                ForEach(parts, id: \.self) { part in
                    Text(verbatim: part)
                }
            }
            .themedFont(.tiny)
            .monospacedDigit()
            .foregroundStyle(.appSecondary)
            .lineLimit(1)
        }
    }

    /// The menu's old empty line, unchanged: a drive that is away is named,
    /// otherwise nothing is installed.
    @ViewBuilder
    private var emptyLibrary: some View {
        Group {
            if let statement = model.notConnectedStatement {
                Text(verbatim: statement)
            } else {
                Text("No models installed yet", bundle: .module)
            }
        }
        .foregroundStyle(.appSecondary)
        .padding(.vertical, 6)
    }

    // MARK: - Actions

    private var actions: some View {
        VStack(alignment: .leading, spacing: 6) {
            if model.remoteSelections[model.selectedChatID] != nil {
                actionButton(Text("Use local model", bundle: .module), systemImage: "desktopcomputer") {
                    model.useLocalModel(chatID: model.selectedChatID)
                }
            }
            actionButton(Text("Manage installed models\u{2026}", bundle: .module), systemImage: "internaldrive") {
                model.activeSection = .modelManager
            }
            actionButton(Text("Discover new models\u{2026}", bundle: .module), systemImage: "shippingbox") {
                model.activeSection = .modelHub
            }
            actionButton(Text("Open model folder\u{2026}", bundle: .module), systemImage: "folder") {
                ModelLocationPicker.choose(for: model)
            }
            if model.canReloadModel {
                actionButton(Text("Reload model", bundle: .module), systemImage: "arrow.clockwise") {
                    model.reloadModel()
                }
            }
        }
    }

    private func actionButton(_ title: Text, systemImage: String, action: @escaping () -> Void) -> some View {
        Button {
            isPresented = false
            action()
        } label: {
            Label { title } icon: { Image(systemName: systemImage) }
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
        }
    }

    // MARK: - Choosing

    private func move(by delta: Int, count: Int) -> KeyPress.Result {
        highlight = ModelPickerList.moveHighlight(from: highlight, by: delta, count: count)
        return .handled
    }

    private func choose(at index: Int?, in rows: [ModelPickerRow]) {
        guard let index, rows.indices.contains(index) else { return }
        choose(rows[index])
    }

    private func isRemote(_ row: ModelPickerRow) -> Bool {
        if case .remote = row.source { return true }
        return false
    }

    private func choose(_ row: ModelPickerRow) {
        switch row.source {
        case .local(let alias):
            guard let item = model.loadableModels.first(where: { $0.alias == alias }) else { return }
            isPresented = false
            model.openChatWithModel(item)
        case .remote(let providerID, let modelID):
            // Same guard the Remote pill applies by disabling itself.
            guard !model.isRunningInSelectedChat else { return }
            isPresented = false
            // After the popover closes, so the confirmation is not raised
            // while a transient popover still holds the window's focus.
            let chatID = model.selectedChatID
            Task { @MainActor in
                model.requestRemoteModel(providerID: providerID, modelID: modelID, chatID: chatID)
            }
        }
    }
}
