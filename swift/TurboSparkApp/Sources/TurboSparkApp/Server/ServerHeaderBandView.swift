import AppKit
import SwiftUI
import TurboSpark

/// The one row that is always visible: is it on, where is it, what is it
/// doing.
struct ServerHeaderBandView: View {
    @ObservedObject var model: AppModel
    @State private var copied = false

    private var info: ServerInfo? { model.serverInfo }
    private var isRunning: Bool { model.server != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                stateDot
                VStack(alignment: .leading, spacing: 2) {
                    Text(stateTitle)
                        .font(.system(size: 15, weight: .semibold))
                    Text(stateSubtitle)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 12)

                if model.serverBusy {
                    ProgressView().controlSize(.small)
                }

                Button(isRunning ? "Stop" : "Start Server") {
                    isRunning ? model.stopServer() : model.startServer()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(model.serverBusy)
            }

            if let info, isRunning {
                addressRow(info)
                summaryRow(info)
            }

            if isRunning, (info?.models.isEmpty ?? true) {
                // **NAMED RATHER THAN LEFT TO BE DISCOVERED BY A 503.** A
                // bound server with nothing attached is a real and useful
                // state, and it is also the one a person is most likely to
                // misread as broken.
                Label(
                    "Bound and answering, with no model attached. "
                        + "Requests get a 503 until you add one below.",
                    systemImage: "info.circle")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5))
    }

    private var stateDot: some View {
        Circle()
            .fill(stateColor)
            .frame(width: 10, height: 10)
            .overlay(Circle().stroke(stateColor.opacity(0.25), lineWidth: 6))
    }

    /// **AMBER FOR RUNNING-BUT-EMPTY, not green.** Green beside a server that
    /// answers every generation with a 503 is the light saying a thing works
    /// when it does not yet.
    private var stateColor: Color {
        guard isRunning else { return .secondary }
        if info?.models.isEmpty ?? true { return .orange }
        return model.serverMetrics.totalErrors > 0 ? .yellow : .green
    }

    private var stateTitle: String {
        guard isRunning else { return "Server stopped" }
        return (info?.models.isEmpty ?? true) ? "Running, no model" : "Running"
    }

    private var stateSubtitle: String {
        guard isRunning, let info else {
            return "Serve your loaded models over HTTP to any app on this machine."
        }
        let models = info.models.count
        let noun = models == 1 ? "model" : "models"
        return "\(models) \(noun) - up \(uptimeText(info.uptimeSeconds))"
    }

    private func addressRow(_ info: ServerInfo) -> some View {
        let rows = ServerStatusRows(info: info)
        return HStack(spacing: 8) {
            Text(rows.address)
                .font(.system(size: 13, design: .monospaced))
                .textSelection(.enabled)

            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(rows.address, forType: .string)
                copied = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
            } label: {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
            }
            .buttonStyle(.borderless)
            .help("Copy the address")

            Divider().frame(height: 12)

            Label(rows.authLabel, systemImage: rows.authIsWarning ? "lock.open" : "lock")
                .font(.system(size: 11))
                .foregroundStyle(rows.authIsWarning ? Color.orange : Color.secondary)
                // The warning is not decoration: an unauthenticated loopback
                // socket is reachable by every process on this machine, and a
                // key of nothing but spaces produces exactly this state while
                // looking like it took.
                .help(
                    rows.authIsWarning
                        ? "Any process on this machine can reach this server. Set a key under Advanced."
                        : "Requests must send this key.")
        }
    }

    private func summaryRow(_ info: ServerInfo) -> some View {
        let metrics = model.serverMetrics
        return HStack(spacing: 16) {
            stat("Requests", "\(metrics.totalRequests)")
            stat("In flight", "\(metrics.inFlight)")
            stat(
                "Errors", "\(metrics.totalErrors)",
                tint: metrics.totalErrors > 0 ? .orange : nil)
            if let rate = metrics.aggregateTokensPerSecond {
                stat("Decode", String(format: "%.1f tok/s", rate))
            }
            if metrics.droppedEvents > 0 {
                // Surfaced rather than swallowed: a console missing rows
                // reads exactly like a server that was idle.
                stat("Log gaps", "\(metrics.droppedEvents)", tint: .orange)
            }
        }
        .font(.system(size: 11))
    }

    private func stat(_ label: String, _ value: String, tint: Color? = nil) -> some View {
        HStack(spacing: 4) {
            Text(label).foregroundStyle(.secondary)
            Text(value)
                .monospacedDigit()
                .foregroundStyle(tint ?? .primary)
        }
    }

    private func uptimeText(_ seconds: UInt64) -> String {
        if seconds < 60 { return "\(seconds)s" }
        if seconds < 3600 { return "\(seconds / 60)m" }
        return String(format: "%dh %dm", seconds / 3600, (seconds % 3600) / 60)
    }
}
