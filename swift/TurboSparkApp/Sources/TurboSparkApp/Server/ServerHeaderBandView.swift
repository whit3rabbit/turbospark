import AppKit
import SwiftUI
import TurboSpark

/// The one row that is always visible: is it on, where is it, what is it
/// doing.
struct ServerHeaderBandView: View {
    @ObservedObject var model: AppModel

    private var info: ServerInfo? { model.serverInfo }
    private var isRunning: Bool { model.server != nil }

    var body: some View {
        APIStatusBand(
            color: stateColor,
            title: stateTitle,
            subtitle: stateSubtitle,
            isRunning: isRunning,
            busy: model.serverBusy,
            startLabel: Text("Start Server", bundle: .module),
            stopLabel: Text("Stop Server", bundle: .module),
            startDisabled: model.serverBusy || !model.serverPortIsValid,
            stopDisabled: model.serverBusy,
            start: { model.startServer() },
            stop: { model.stopServer() }
        ) {
            if let info, isRunning {
                addressRow(info)
                summaryRow(info)
            }

            if isRunning, servesNothing {
                // **NAMED RATHER THAN LEFT TO BE DISCOVERED BY A 503.** A
                // bound server with nothing attached is a real and useful
                // state, and it is also the one a person is most likely to
                // misread as broken.
                Label {
                    Text("Bound and answering, with no model attached. Requests get a 503 until you add one below.", bundle: .module)
                } icon: {
                    Image(systemName: "info.circle")
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }
        }
    }

    private var servesNothing: Bool {
        (info?.models.isEmpty ?? true) && (info?.imageModels.isEmpty ?? true) && (info?.audioModels.isEmpty ?? true)
    }

    /// **AMBER FOR RUNNING-BUT-EMPTY, not green.** Green beside a server that
    /// answers every generation with a 503 is the light saying a thing works
    /// when it does not yet.
    private var stateColor: Color {
        guard isRunning else { return .secondary }
        if servesNothing { return .orange }
        return model.serverMetrics.totalErrors > 0 ? .yellow : .green
    }

    private var stateTitle: Text {
        guard isRunning else { return Text("Server stopped", bundle: .module) }
        return servesNothing ? Text("Running, no model", bundle: .module) : Text("Running", bundle: .module)
    }

    private var stateSubtitle: Text {
        guard isRunning, let info else {
            return Text("Serve your loaded models over HTTP to any app on this machine.", bundle: .module)
        }
        let models = info.models.count + info.imageModels.count + info.audioModels.count
        let uptime = uptimeText(info.uptimeSeconds)
        return models == 1
            ? Text("1 model, up \(uptime)", bundle: .module)
            : Text("\(models) models, up \(uptime)", bundle: .module)
    }

    private func addressRow(_ info: ServerInfo) -> some View {
        let rows = ServerStatusRows(info: info, guardrails: model.serverStartedGuardrails)
        return HStack(spacing: 8) {
            Text(rows.address)
                .themedCode(.large)
                .textSelection(.enabled)

            APICopyButton(label: Text("Copy the address", bundle: .module)) { rows.address }

            Divider().frame(height: 12)

            // The guardrails the SERVER started with, which is a different
            // enforcement point from the Chat pane's own engine: an HTTP
            // client bypasses that entirely. Reported here because it cannot
            // be changed without a restart.
            Label { guardrailsText(rows) } icon: { Image(systemName: "shield.lefthalf.filled") }
                .themedFont(.small)
                .foregroundStyle(Color.secondary)
                .help(Text("Tool-call guardrails for requests this server answers. Resolved when the server started; restart it to change them.", bundle: .module))

            Label {
                rows.authIsWarning ? Text("No API key", bundle: .module) : Text("API key required", bundle: .module)
            } icon: {
                Image(systemName: rows.authIsWarning ? "lock.open" : "lock")
            }
                .themedFont(.small)
                .foregroundStyle(rows.authIsWarning ? Color.orange : Color.secondary)
                // The warning is not decoration: an unauthenticated loopback
                // socket is reachable by every process on this machine, and a
                // key of nothing but spaces produces exactly this state while
                // looking like it took.
                .help(
                    rows.authIsWarning
                        ? Text("Any process on this machine can reach this server. Set a key under Advanced.", bundle: .module)
                        : Text("Requests must send this key.", bundle: .module))
        }
    }

    /// `ServerStatusRows.guardrailsLabel` is an English word chosen in State/
    /// (and asserted by tests there), so the three known readings are mapped
    /// to catalog text here and anything new renders as given.
    private func guardrailsText(_ rows: ServerStatusRows) -> Text {
        switch rows.guardrailsLabel {
        case "on": return Text("Guardrails on", bundle: .module)
        case "off": return Text("Guardrails off", bundle: .module)
        case "unknown (restart the server to report it)":
            return Text("Guardrails unknown (restart the server to report it)", bundle: .module)
        default: return Text(verbatim: "guardrails " + rows.guardrailsLabel)
        }
    }

    private func summaryRow(_ info: ServerInfo) -> some View {
        let metrics = model.serverMetrics
        return HStack(spacing: 16) {
            stat(Text("Requests", bundle: .module), "\(metrics.totalRequests)")
            stat(Text("In flight", bundle: .module), "\(metrics.inFlight)")
            stat(
                Text("Errors", bundle: .module), "\(metrics.totalErrors)",
                tint: metrics.totalErrors > 0 ? .orange : nil)
            if let rate = metrics.aggregateTokensPerSecond {
                stat(Text("Decode", bundle: .module), String(format: "%.1f tok/s", rate))
            }
            if metrics.droppedEvents > 0 {
                // Surfaced rather than swallowed: a console missing rows
                // reads exactly like a server that was idle.
                stat(Text("Log gaps", bundle: .module), "\(metrics.droppedEvents)", tint: .orange)
            }
        }
        .themedFont(.small)
        .accessibilityElement(children: .combine)
    }

    private func stat(_ label: Text, _ value: String, tint: Color? = nil) -> some View {
        HStack(spacing: 4) {
            label.foregroundStyle(.appSecondary)
            Text(verbatim: value)
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
