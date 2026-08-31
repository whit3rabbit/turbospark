import AppKit
import SwiftUI
import TurboSpark

/// What the server is serving, and how to add or remove one.
struct ServerLoadedModelsView: View {
    @ObservedObject var model: AppModel
    @State private var showingPicker = false

    private var rows: [ServerModelRow] { model.serverModelRows }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Serving")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if model.session != nil, !rows.contains(where: \.isChatSession) {
                    // The cheap path, and worth its own button: the chat
                    // model is already mapped, so serving it costs nothing
                    // where a second install costs its whole footprint.
                    Button("Serve loaded model") { model.attachChatSession() }
                        .buttonStyle(.link)
                        .font(.system(size: 11))
                }
                Button {
                    showingPicker = true
                } label: {
                    Label("Load Model", systemImage: "plus")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(model.serverBusy)
            }

            if rows.isEmpty {
                Text("Nothing attached yet.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
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
                        .fill(Color(nsColor: .controlBackgroundColor)))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5))
            }

            memoryFootprintBar
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
                        .font(.system(size: 12, design: .monospaced))
                        .textSelection(.enabled)
                    if row.isChatSession {
                        Text("chat")
                            .font(.system(size: 9, weight: .medium))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(
                                TurboSparkTheme.accentColor.opacity(0.15),
                                in: Capsule())
                    }
                }
                Text(detailLine(row))
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 8)

            Text("\(row.requestsServed) req")
                .font(.system(size: 10))
                .monospacedDigit()
                .foregroundStyle(.secondary)

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
                Label("Eject", systemImage: "eject")
                    .labelStyle(.iconOnly)
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

    /// What the machine has left, so a user can see whether a second model
    /// fits before they try.
    @ViewBuilder
    private var memoryFootprintBar: some View {
        if let telemetry = model.telemetry, telemetry.physicalMemoryBytes > 0 {
            let used = Double(TurboSparkSession.peakFootprintBytes ?? 0)
            let total = Double(telemetry.physicalMemoryBytes)
            let fraction = min(max(used / total, 0), 1)
            VStack(alignment: .leading, spacing: 3) {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.secondary.opacity(0.15))
                        Capsule()
                            .fill(fraction > 0.8 ? Color.orange : TurboSparkTheme.accentColor)
                            .frame(width: geometry.size.width * fraction)
                    }
                }
                .frame(height: 4)

                // **`peakFootprintBytes` IS A PEAK, AND THE LABEL SAYS SO.**
                // It is the high-water mark this process reached rather than
                // what it holds now, so calling it "used" would overstate
                // the headroom question it is here to answer.
                Text(
                    "\(byteText(UInt64(used))) peak of \(byteText(telemetry.physicalMemoryBytes)) "
                        + "unified memory")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func byteText(_ bytes: UInt64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        return formatter.string(fromByteCount: Int64(bytes))
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
            Text("Serve another model")
                .font(.system(size: 14, weight: .semibold))
                .padding(16)

            Divider()

            let candidates = model.serverAttachableModels
            if candidates.isEmpty {
                Text("Every installed model is already attached.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
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
                Text("Each model is opened separately and holds its own memory.")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Done") { dismiss() }
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
                        .font(.system(size: 12, weight: .medium))
                    Text((candidate.path as NSString).lastPathComponent)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "plus.circle")
                    .foregroundStyle(TurboSparkTheme.accentColor)
            }
            .contentShape(Rectangle())
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
        }
        .buttonStyle(.plain)
    }
}
