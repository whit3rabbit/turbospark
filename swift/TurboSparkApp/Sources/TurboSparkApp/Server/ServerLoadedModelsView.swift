import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// What the server is serving, and how to add or remove one.
@MainActor
struct ServerLoadedModelsView: View {
    @ObservedObject var model: AppModel
    @State private var showingPicker = false

    private var rows: [ServerModelRow] { model.serverModelRows }

    var body: some View {
        APIPanel(title: Text("Serving", bundle: .module)) {
            // attachChatSession is a no-op without a running, idle server.
            if model.session != nil, model.server != nil, !model.serverBusy,
               !rows.contains(where: \.isChatSession) {
                // The cheap path, and worth its own button: the chat
                // model is already mapped, so serving it costs nothing
                // where a second install costs its whole footprint.
                Button { model.attachChatSession() } label: { Text("Serve loaded model", bundle: .module) }
                    .appAction(.secondary, size: APIControlSize.panel)
            }
            Button {
                showingPicker = true
            } label: {
                Label { Text("Attach Model...", bundle: .module) } icon: { Image(systemName: "plus") }
            }
            .appAction(.primary, size: APIControlSize.panel)
            .disabled(model.serverBusy || model.server == nil)
        } content: {
            APIRowList {
                if rows.isEmpty {
                    APINothingServedRow()
                } else {
                    ForEach(rows) { row in
                        modelRow(row)
                        if row.id != rows.last?.id {
                            Divider()
                        }
                    }
                }
            }
        }
        .sheet(isPresented: $showingPicker) {
            ServerModelPickerSheet(model: model)
        }
    }

    private func modelRow(_ row: ServerModelRow) -> some View {
        APIServedModelRow(
            name: row.id,
            badge: row.isChatSession
                ? Text("chat", bundle: .module)
                : (row.isAudioModel ? Text("audio", bundle: .module) : nil),
            badgeTint: row.isChatSession ? TurboSparkTheme.accentColor : .purple,
            detachHelp: row.isChatSession
                ? Text("Stop serving this over HTTP. It stays loaded for the Chat pane.", bundle: .module)
                : Text("Stop serving this and release it from memory.", bundle: .module),
            detach: { model.detachModelFromServer(id: row.id) }
        ) {
            detailLine(row)
            // READ-ONLY. Steering resolves at OPEN and this server serves
            // already-open sessions, so it is a property of the model's
            // load rather than of the server. Showing every known state
            // lets an operator distinguish active, supported-but-off,
            // and unsupported models without implying that this pane can
            // change an already-open session.
            if let status = row.steeringStatus {
                Label { steeringText(status) } icon: { Image(systemName: status.systemImageName) }
                    .foregroundStyle(
                        status.isActive || status.isUnsupported
                            ? AnyShapeStyle(.appWarning) : AnyShapeStyle(.appSecondary))
            }
        } trailing: {
            Text("\(row.requestsServed) requests", bundle: .module)
                .themedFont(.tiny)
                .monospacedDigit()
                .foregroundStyle(.appSecondary)
            APICopyButton(label: Text("Copy the model id clients should send", bundle: .module)) { row.id }
        }
    }

    /// The steering label is built in State/ as English; the two fixed
    /// readings are mapped to catalog text here, and an engine-supplied
    /// summary or reason is shown as given.
    private func steeringText(_ status: ServerModelSteeringStatus) -> Text {
        switch status {
        case let .active(summary): return Text(verbatim: summary)
        case .off: return Text("Steering Off (supported)", bundle: .module)
        case let .unsupported(reason): return Text("Steering unavailable: \(reason)", bundle: .module)
        }
    }

    /// **NOTHING HERE IS STATED WHEN IT WAS NOT MEASURED** (`swift/CLAUDE.md`
    /// Gotcha 23). A model attached from outside this app has no session
    /// here to read a window or a slot count off, and a zero rendered
    /// through the same row would read as a measurement of zero -- worse,
    /// the slot count would read as a plausible 16 arrived at by ignorance.
    private func detailLine(_ row: ServerModelRow) -> Text {
        if row.isAudioModel {
            return Text("Audio model", bundle: .module)
        }
        guard row.maxContext > 0 else { return Text("Attached", bundle: .module) }
        let context = Text("\(row.maxContext.formatted()) context", bundle: .module)
        guard row.expertCacheSlots > 0 else { return context }
        return context + Text(verbatim: " - ") + Text("\(row.expertCacheSlots) expert slots", bundle: .module)
    }
}

/// Picks an installed model to serve.
///
/// **OFFERS ONLY WHAT IS NOT ALREADY ATTACHED**, because attaching the same
/// install twice is refused by the engine (its id would collide) and a row
/// that can only fail is a row that should not be there.
private struct ServerModelPickerSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Attach another model", bundle: .module)
                .themedFont(.callout, weight: .semibold)
                .padding(16)

            Divider()

            let candidates = model.serverAttachableModels
            if candidates.isEmpty {
                Text("Every installed model is already attached.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .padding(24)
                    .frame(maxWidth: .infinity)
            } else {
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(candidates) { candidate in
                            row(candidate)
                            Divider()
                        }
                    }
                }
                .frame(maxHeight: 320)
            }

            Divider()

            HStack {
                // Opening a second model maps its weights again, and on a
                // machine that can hold one comfortably it may not hold two.
                // Said before the click rather than surfaced as a load
                // failure after it.
                Text("Each model is opened separately and holds its own memory.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                Spacer()
                Button { dismiss() } label: { Text("Done", bundle: .module) }
                    .appAction(.primary, size: APIControlSize.panel)
                    .keyboardShortcut(.defaultAction)
            }
            .padding(12)
        }
        .frame(width: 460)
    }

    private func row(_ candidate: InstalledModel) -> some View {
        Button {
            model.attachModelToServer(candidate)
            dismiss()
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(verbatim: candidate.alias)
                        .themedFont(.small, weight: .medium)
                    Text(verbatim: (candidate.path as NSString).lastPathComponent)
                        .themedCode(.callout)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                Image(systemName: "plus.circle")
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
            }
            .contentShape(Rectangle())
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
        }
        .buttonStyle(.plain)
        .help(Text("Attach \(candidate.alias) to server", bundle: .module))
        .accessibilityLabel(Text("Attach \(candidate.alias) to server", bundle: .module))
    }
}
