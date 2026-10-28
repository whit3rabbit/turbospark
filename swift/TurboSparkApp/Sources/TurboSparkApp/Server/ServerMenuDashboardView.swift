import AppKit
import SwiftUI
import TurboSpark

/// The menu bar panel: the server's status light and switch, the app's
/// memory, what is loaded and how to free it, and the server shortcuts.
///
/// Status, slots and memory come from `AppModel+MenuBar`, which derives
/// them from state the Server pane already owns, so the two never disagree.
// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
struct ServerMenuDashboardView: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme

    var body: some View {
        let phase = model.menuBarPhase
        VStack(alignment: .leading, spacing: 0) {
            header(phase)
            if case .failed(let message) = phase {
                failureCard(message)
            }
            MenuBarDivider()
            memorySection
            MenuBarDivider()
            loadedSection
            MenuBarDivider()
            MenuBarSubmenuRow(title: Text("Coding Agents", bundle: .module)) {
                ServerMenuBarView(model: model).agentsMenuItems
            }
            MenuBarSubmenuRow(title: Text("API Endpoints", bundle: .module)) {
                ServerMenuBarView(model: model).endpointsMenuItems
            }
            MenuBarSubmenuRow(title: Text("More", bundle: .module)) {
                ServerMenuBarView(model: model)
            }
            MenuBarDivider()
            appSection
        }
        .padding(6)
        .frame(width: 340 * AppMetrics.textScale(theme))
        .background(.appElevated)
    }

    // MARK: - Header

    /// On while a start is in flight too, so turning it off during the bind
    /// reaches `stopServer()`, which records the request (state#28).
    private var serverSwitch: Binding<Bool> {
        Binding(
            get: { model.server != nil || model.serverBusy },
            set: { on in on ? model.startServer() : model.stopServer() })
    }

    private func header(_ phase: MenuBarServerPhase) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(verbatim: "TurboSpark")
                    .themedFont(.callout, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer(minLength: 8)
                Toggle(isOn: serverSwitch) { Text("Server", bundle: .module) }
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .labelsHidden()
                    // An attach holds the server's lock; the status line
                    // says Loading while it does.
                    .disabled(model.server != nil && model.serverBusy)
            }
            MenuBarStatusLine(phase: phase, address: model.menuBarAddress(for: phase)) {
                guard let address = model.menuBarAddress(for: phase) else { return }
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(address, forType: .string)
                model.showToast(AppStrings.localized("Copied server address"), style: .info)
            }
            if let note = note(phase) {
                note
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            if model.server != nil, let info = model.serverInfo {
                statsLine(info)
            }
        }
        .padding(.horizontal, 8)
        .padding(.top, 8)
        .padding(.bottom, 4)
    }

    private func note(_ phase: MenuBarServerPhase) -> Text? {
        switch phase {
        case .idle: return Text("TurboSpark Server (no model)", bundle: .module)
        case .daemon: return Text("TurboSpark CLI Daemon is running", bundle: .module)
        default: return nil
        }
    }

    private func statsLine(_ info: ServerInfo) -> some View {
        let metrics = model.serverMetrics
        return HStack(spacing: 12) {
            Text("\(metrics.totalRequests) requests", bundle: .module)
            if let tokens = metrics.aggregateTokensPerSecond {
                Text(verbatim: String(format: "%.1f tok/s", tokens))
            }
            Text("Uptime: \(MenuBarUptime.text(info.uptimeSeconds))", bundle: .module)
        }
        .themedFont(.tiny)
        .monospacedDigit()
        .foregroundStyle(.appSecondary)
    }

    /// The last start's error and the next step, until the next start.
    private func failureCard(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(verbatim: message)
                .themedFont(.small)
                .foregroundStyle(.appText)
                .lineLimit(4)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            HStack(spacing: AppMetrics.actionSpacing) {
                // A pinned port is the common cause of a refused bind, and
                // automatic is the default the app ships with.
                if model.serverPinnedPort != 0 {
                    Button {
                        model.serverPinnedPort = 0
                        model.startServer()
                    } label: { Text("Use Automatic Port", bundle: .module) }
                    .appAction(.primary, size: APIControlSize.row)
                    Button { model.startServer() } label: { Text("Retry", bundle: .module) }
                        .appAction(.secondary, size: APIControlSize.row)
                } else {
                    Button { model.startServer() } label: { Text("Retry", bundle: .module) }
                        .appAction(.primary, size: APIControlSize.row)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .fill(.appDangerTint))
        .padding(.horizontal, 2)
        .padding(.vertical, 4)
    }

    // MARK: - Memory

    private var memorySection: some View {
        let reading = model.menuBarMemory
        let usage = Text(
            "\(MenuBarMemoryReading.format(reading.usedBytes)) of \(MenuBarMemoryReading.format(reading.capacityBytes))",
            bundle: .module)
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("Memory", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                Spacer(minLength: 8)
                usage
                    .themedCode(.tiny)
                    .foregroundStyle(.appText)
            }
            AppProgressBar(
                value: reading.fraction, label: Text("Memory", bundle: .module),
                valueText: usage, size: .compact, animated: false)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .help(Text("Physical footprint of this process (phys_footprint).", bundle: .module))
    }

    // MARK: - Loaded models

    private var loadedSection: some View {
        let slots = model.menuBarSlots
        return VStack(alignment: .leading, spacing: 2) {
            Text("Loaded", bundle: .module)
                .themedFont(.tiny, weight: .semibold)
                .foregroundStyle(.appSecondary)
                .accessibilityAddTraits(.isHeader)
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
            if slots.isEmpty {
                Text("No model loaded.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
            }
            ForEach(slots) { slot in
                MenuBarSlotRow(slot: slot, canUnload: model.menuBarCanUnload(slot.unload)) {
                    model.menuBarUnload(slot.unload)
                }
            }
            MenuBarSubmenuRow(title: Text("Load Model", bundle: .module)) {
                loadModelItems
            }
        }
    }

    /// Chat models load into the Chat pane, which serves them when the server
    /// is on. Image and audio load in their own workspaces, which own their
    /// family routing and consent rules, so the menu sends the user there.
    @ViewBuilder
    private var loadModelItems: some View {
        Section {
            // The loadable list only (requirement 3.3).
            let loadable = model.loadableModels
            if loadable.isEmpty {
                if let statement = model.notConnectedStatement {
                    Text(verbatim: statement)
                } else {
                    Text("No models installed", bundle: .module)
                }
            }
            ForEach(loadable) { installed in
                Button {
                    // A menu cannot host the "Load anyway" question: for a
                    // warned model the main window comes forward to ask.
                    model.requestLoad(installed, bringWindowForward: true)
                } label: {
                    if model.session != nil && model.selected?.path == installed.path {
                        Label { Text(verbatim: installed.alias) } icon: { Image(systemName: "checkmark") }
                    } else {
                        Text(verbatim: installed.alias)
                    }
                }
                .disabled(model.generating || model.opening)
            }
        } header: {
            Text("Chat", bundle: .module)
        }
        let embedding = model.serverEmbeddingModelInput.trimmingCharacters(in: .whitespacesAndNewlines)
        if model.server != nil, !embedding.isEmpty, !model.serverEmbeddingModelIDs.contains(embedding) {
            Section {
                Button { model.attachEmbeddingModelToServer(embedding) } label: { Text(verbatim: embedding) }
                    .disabled(model.serverBusy)
            } header: {
                Text("Embeddings", bundle: .module)
            }
        }
        Divider()
        Button { model.showMainWindow(navigatingTo: .images) } label: { Text("Open Images", bundle: .module) }
        if model.typeSafeServer == nil {
            Button { Task { await model.startTypeSafeServer() } } label: { Text("Start TypeSafe", bundle: .module) }
                .disabled(model.typeSafeBusy)
        }
        Divider()
        Button { model.showMainWindow(navigatingTo: .modelHub) } label: { Text("Discover Models...", bundle: .module) }
    }

    // MARK: - App

    @ViewBuilder
    private var appSection: some View {
        MenuBarRowButton(title: Text("Admin Panel", bundle: .module)) {
            model.showMainWindow(navigatingTo: .server)
        }
        MenuBarRowButton(title: Text("Chat with TurboSpark", bundle: .module)) {
            model.showMainWindow(navigatingTo: .chat)
        }
        MenuBarRowButton(title: Text("Preferences...", bundle: .module), trailing: Text(verbatim: "\u{2318},")) {
            model.openSettings(tab: .general)
        }
        .keyboardShortcut(",", modifiers: .command)
        MenuBarRowButton(title: Text("Quit TurboSpark", bundle: .module), trailing: Text(verbatim: "\u{2318}Q")) {
            NSApp.terminate(nil)
        }
        .keyboardShortcut("q", modifiers: .command)
    }
}
