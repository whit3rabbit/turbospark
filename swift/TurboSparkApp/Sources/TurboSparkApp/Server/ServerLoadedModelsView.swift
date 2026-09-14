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
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Serving", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                Spacer()
                if model.session != nil, !rows.contains(where: \.isChatSession) {
                    // The cheap path, and worth its own button: the chat
                    // model is already mapped, so serving it costs nothing
                    // where a second install costs its whole footprint.
                    Button { model.attachChatSession() } label: { Text("Serve loaded model", bundle: .module) }
                        .buttonStyle(.link)
                        .themedFont(.tiny)
                }
                Button {
                    showingPicker = true
                } label: {
                    Label { Text("Load Model", bundle: .module) } icon: { Image(systemName: "plus") }
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
                .disabled(model.serverBusy || model.server == nil)
            }

            if rows.isEmpty {
                Text("Nothing attached yet.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 10)
            } else {
                VStack(spacing: 0) {
                    ForEach(rows) { row in
                        modelRow(row)
                        if row.id != rows.last?.id {
                            Divider()
                        }
                    }
                }
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(.appSurface))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(.appBorder, lineWidth: 0.5))
            }


        }
        .sheet(isPresented: $showingPicker) {
            ServerModelPickerSheet(model: model)
        }
    }

    private func modelRow(_ row: ServerModelRow) -> some View {
        HStack(spacing: 12) {
            Circle().fill(Color.green).frame(width: 7, height: 7)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(row.id)
                        .themedCode(.base)
                        .textSelection(.enabled)
                    if row.isChatSession {
                        Text("chat", bundle: .module)
                            .themedFont(.micro, weight: .medium)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(
                                TurboSparkTheme.accentColor.opacity(0.15),
                                in: Capsule())
                    }
                }
                Text(detailLine(row))
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                // READ-ONLY. Steering resolves at OPEN and this server serves
                // already-open sessions, so it is a property of the model's
                // load rather than of the server. Showing every known state
                // lets an operator distinguish active, supported-but-off,
                // and unsupported models without implying that this pane can
                // change an already-open session.
                if let status = row.steeringStatus {
                    Label(status.label, systemImage: status.systemImageName)
                        .themedFont(.tiny)
                        .foregroundStyle(
                            status.isActive || status.isUnsupported
                                ? Color.orange : Color.secondary)
                }
            }

            Spacer(minLength: 8)

            Text(verbatim: "\(row.requestsServed) req")
                .themedFont(.tiny)
                .monospacedDigit()
                .foregroundStyle(.appSecondary)

            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(row.id, forType: .string)
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .buttonStyle(.borderless)
            .help("Copy the model id clients should send")

            Button {
                model.detachModelFromServer(id: row.id)
            } label: {
                Label { Text("Eject", bundle: .module) } icon: { Image(systemName: "eject") }
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.borderless)
            .help(
                row.isChatSession
                    ? "Stop serving this over HTTP. It stays loaded for the Chat pane."
                    : "Stop serving this and release it from memory.")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
    }

    /// **NOTHING HERE IS STATED WHEN IT WAS NOT MEASURED** (`swift/CLAUDE.md`
    /// Gotcha 23). A model attached from outside this app has no session
    /// here to read a window or a slot count off, and a zero rendered
    /// through the same row would read as a measurement of zero -- worse,
    /// the slot count would read as a plausible 16 arrived at by ignorance.
    private func detailLine(_ row: ServerModelRow) -> String {
        guard row.maxContext > 0 else { return "attached" }
        var parts = ["\(row.maxContext.formatted()) ctx"]
        if row.expertCacheSlots > 0 {
            parts.append("\(row.expertCacheSlots) slots")
        }
        return parts.joined(separator: " - ")
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
            Text("Serve another model", bundle: .module)
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
                    Text(candidate.alias)
                        .themedFont(.small, weight: .medium)
                    Text((candidate.path as NSString).lastPathComponent)
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
        .help("Attach \(candidate.alias) to server")
        .accessibilityLabel("Attach \(candidate.alias) to server")
    }
}
