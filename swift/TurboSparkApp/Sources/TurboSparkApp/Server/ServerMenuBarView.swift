import AppKit
import SwiftUI
import TurboSpark

/// Native macOS menu bar dropdown view for TurboSpark server management.
/// Inspired by the oMLX status item layout, providing quick server control,
/// listening address discovery, API endpoint clipboard copying, model
/// switching, and live serving statistics.
public struct ServerMenuBarView: View {
    @ObservedObject public var model: AppModel
    /// True while a daemon verb runs off the main actor, so a second click
    /// cannot start an overlapping stop/restart.
    @State private var daemonBusy = false

    public init(model: AppModel) {
        self.model = model
    }

    private var isRunning: Bool { model.server != nil }
    private var info: ServerInfo? { model.serverInfo }

    public var body: some View {
        statusSection
        Divider()
        serverControlSection
        Divider()
        statsSubmenu
        endpointsSubmenu
        agentsSubmenu
        modelsSubmenu
        Divider()
        navigationSection
        Divider()
        appManagementSection
    }

    // MARK: - Sections

    private var daemonStatus: TurboSparkDaemonStatus? {
        try? TurboSparkDaemon.status()
    }

    @ViewBuilder
    private var statusSection: some View {
        Label { statusTitle } icon: { Image(systemName: statusSymbol) }

        if isRunning, let info = model.serverInfo {
            let address = info.baseURL?.absoluteString ?? "http://\(info.host):\(info.port)"
            Button {
                copyToClipboard(address)
                model.showToast(AppStrings.localized("Copied server address"), style: .info)
            } label: { Text("Listening: \(address)", bundle: .module) }
        } else if let daemon = daemonStatus, daemon.running {
            let address = daemon.endpoint ?? "http://127.0.0.1:\(daemon.port ?? 8080)/v1"
            Button {
                copyToClipboard(address)
                model.showToast(AppStrings.localized("Copied daemon address"), style: .info)
            } label: { Text("CLI Daemon: \(address)", bundle: .module) }
            if let logPath = daemon.logPath {
                Button {
                    ModelStorageManager.revealInFinder(path: logPath)
                } label: { Text("Reveal Daemon Log", bundle: .module) }
            }
        } else {
            if model.serverPinnedPort == 0 {
                Text("Port: automatic", bundle: .module)
            } else {
                Text("Port: \(String(model.serverPinnedPort))", bundle: .module)
            }
        }
    }

    @ViewBuilder
    private var serverControlSection: some View {
        if isRunning {
            Button {
                model.stopServer()
            } label: { Text("Stop Server", bundle: .module) }
            .disabled(model.serverBusy)
        } else if let daemon = daemonStatus, daemon.running {
            Button {
                stopDaemon()
            } label: {
                Text("Stop Background Daemon (PID \(daemon.pid.map(String.init) ?? "?"))", bundle: .module)
            }
            .disabled(daemonBusy)
            Button {
                restartDaemon()
            } label: { Text("Restart Background Daemon", bundle: .module) }
            .disabled(daemonBusy)
            Button {
                model.startServer()
            } label: { Text("Start In-App Server", bundle: .module) }
            .disabled(model.serverBusy)
        } else {
            Button {
                model.startServer()
            } label: { Text("Start Server", bundle: .module) }
            .disabled(model.serverBusy)
            Button {
                startDaemon()
            } label: { Text("Start Background Daemon", bundle: .module) }
            .disabled(daemonBusy)
        }
    }

    // MARK: - Daemon control

    /// The three daemon verbs, each reporting what ACTUALLY happened.
    ///
    /// **`try?` UNDER AN UNCONDITIONAL SUCCESS TOAST REPORTED FAILURES AS
    /// SUCCESSES.** A daemon start fails for ordinary reasons -- no
    /// `turbospark-server` binary beside the app, a refused port, the child
    /// exiting on a bad flag -- and "Started background daemon" over any of
    /// them sends the user looking at a server that is not there.
    ///
    /// Start and restart carry `ServerDaemonLaunch.args`, because the
    /// daemon is a separate process that inherits nothing: without them it
    /// binds its own default port with no API key, whatever the Advanced
    /// pane has configured.
    // Whole sentences per verb rather than a verb spliced into one
    // sentence: word order and inflection differ by language.
    private func startDaemon() {
        runDaemonVerb(
            success: AppStrings.localized("Started background daemon"),
            failure: { AppStrings.localized("Could not start the background daemon: \($0)") }
        ) { args in
            try TurboSparkDaemon.start(args: args)
        }
    }

    private func stopDaemon() {
        runDaemonVerb(
            success: AppStrings.localized("Stopped background daemon"),
            failure: { AppStrings.localized("Could not stop the background daemon: \($0)") }
        ) { _ in
            try TurboSparkDaemon.stop()
        }
    }

    private func restartDaemon() {
        runDaemonVerb(
            success: AppStrings.localized("Restarted background daemon"),
            failure: { AppStrings.localized("Could not restart the background daemon: \($0)") }
        ) { args in
            try TurboSparkDaemon.restart(args: args)
        }
    }

    /// Runs one daemon verb detached. Stop waits up to ~3 s for SIGTERM
    /// shutdown plus the liveness checks, and may wait longer on the daemon
    /// lock, so doing it inline froze the whole UI (streaming chat included).
    /// The FFI calls only touch files and processes, so they are safe off the
    /// main actor; the toast is published back on it.
    private func runDaemonVerb(
        success: String,
        failure: @escaping (String) -> String,
        _ body: @escaping @Sendable ([String]) throws -> Void
    ) {
        guard !daemonBusy else { return }
        daemonBusy = true
        let args = daemonArgs
        let model = model
        Task {
            let outcome = await Task.detached(priority: .userInitiated) {
                Result { try body(args) }
            }.value
            daemonBusy = false
            switch outcome {
            case .success:
                model.showToast(success, style: .info)
            case .failure(let error):
                model.showToast(failure(error.localizedDescription), style: .error)
            }
        }
    }

    /// The user's own server settings, carried to the daemon process.
    private var daemonArgs: [String] {
        ServerDaemonLaunch.args(
            port: model.serverPinnedPort,
            apiKey: AppModel.serverAPIKey(from: model.serverAPIKeyInput),
            guardrails: AppModel.serverGuardrails(from: model.guardrailsMode))
    }

    @ViewBuilder
    private var statsSubmenu: some View {
        Menu {
            let metrics = model.serverMetrics
            Text("Total Tokens Processed: \(formatNumber(metrics.totalTokensProcessed))", bundle: .module)
            Text("Requests Served: \(metrics.totalRequests)", bundle: .module)
            Text("In Flight: \(metrics.inFlight)", bundle: .module)
            if let pp = metrics.aggregatePromptTokensPerSecond {
                Text("Avg PP Speed: \(String(format: "%.1f", pp)) tok/s", bundle: .module)
            }
            if let tg = metrics.aggregateTokensPerSecond {
                Text("Avg TG Speed: \(String(format: "%.1f", tg)) tok/s", bundle: .module)
            }
            if metrics.totalReusedPrefixTokens > 0 {
                Text("Reused Prefix Tokens: \(formatNumber(metrics.totalReusedPrefixTokens))", bundle: .module)
            }
            if metrics.totalSessionSlotEvictions > 0 {
                Text("Session Slot Evictions: \(metrics.totalSessionSlotEvictions)", bundle: .module)
            }
            Text("Errors: \(metrics.totalErrors)", bundle: .module)
            if isRunning, let info = model.serverInfo {
                Text("Uptime: \(uptimeText(info.uptimeSeconds))", bundle: .module)
            }
        } label: {
            Text("Serving Stats", bundle: .module)
        }
    }

    @ViewBuilder
    private var agentsSubmenu: some View {
        Menu {
            let port = model.serverPinnedPort == 0 ? (info?.port ?? 8080) : model.serverPinnedPort
            let host = info?.host ?? "127.0.0.1"
            // The key the running server will actually accept, or the same
            // "unused" placeholder the Connect card sends when there is
            // none. The helper's own default ("local") is the key NO server
            // of ours is configured with, so an export copied while a key
            // was set authenticated every agent against nothing.
            let key = AppModel.serverAPIKey(from: model.serverAPIKeyInput) ?? "unused"

            Text("Connect Terminal Agents:", bundle: .module)

            // Product names and their commands: nothing to translate.
            Button {
                let cmd = TurboSparkAgent.launchCommand(for: "claude", host: host, port: port, apiKey: key)
                copyToClipboard(cmd)
                model.showToast(AppStrings.localized("Copied Claude Code export command"), style: .info)
            } label: { Text(verbatim: "Claude Code (claude)") }

            Button {
                let cmd = TurboSparkAgent.launchCommand(for: "codex", host: host, port: port, apiKey: key)
                copyToClipboard(cmd)
                model.showToast(AppStrings.localized("Copied Codex export command"), style: .info)
            } label: { Text(verbatim: "OpenAI CLI / Codex (codex)") }

            Button {
                let cmd = TurboSparkAgent.launchCommand(for: "opencode", host: host, port: port, apiKey: key)
                copyToClipboard(cmd)
                model.showToast(AppStrings.localized("Copied OpenCode export command"), style: .info)
            } label: { Text(verbatim: "OpenCode (opencode)") }

            Button {
                let cmd = TurboSparkAgent.launchCommand(for: "hermes", host: host, port: port, apiKey: key)
                copyToClipboard(cmd)
                model.showToast(AppStrings.localized("Copied environment export command"), style: .info)
            } label: { Text(verbatim: "Hermes / OpenClaw") }

            Divider()

            let starter = "turbospark start claude"
            Button {
                copyToClipboard(starter)
                model.showToast(AppStrings.localized("Copied \(starter)"), style: .info)
            } label: { Text("CLI Starter: \(starter)", bundle: .module) }
        } label: {
            Text("Coding Agents", bundle: .module)
        }
    }

    @ViewBuilder
    private var endpointsSubmenu: some View {
        Menu {
            let base = info?.baseURL?.absoluteString
                ?? "http://127.0.0.1:\(model.serverPinnedPort == 0 ? 8080 : model.serverPinnedPort)"

            Button {
                copyToClipboard(base)
                model.showToast(AppStrings.localized("Copied Base URL"), style: .info)
            } label: { Text("Copy Base URL (\(base))", bundle: .module) }

            Divider()

            Text("OpenAI Compatible:", bundle: .module)
            endpointButton("POST", "/v1/chat/completions", base: base)
            endpointButton("POST", "/v1/embeddings", base: base)
            endpointButton("GET", "/v1/models", base: base)

            Divider()

            Text("Anthropic Compatible:", bundle: .module)
            endpointButton("POST", "/v1/messages", base: base)

            Divider()

            Text("Ollama Compatible:", bundle: .module)
            endpointButton("POST", "/api/chat", base: base)
            endpointButton("POST", "/api/embeddings", base: base)
            endpointButton("GET", "/api/tags", base: base)

            Divider()

            Text("Service:", bundle: .module)
            endpointButton("GET", "/health", base: base)
        } label: {
            Text("API Endpoints", bundle: .module)
        }
    }

    /// One route in the Endpoints submenu: the label is the route itself,
    /// and the toast names it.
    private func endpointButton(_ method: String, _ path: String, base: String) -> some View {
        Button {
            copyToClipboard(base + path)
            model.showToast(AppStrings.localized("Copied \(path) endpoint"), style: .info)
        } label: {
            Text(verbatim: "\(method) \(path)")
        }
    }

    @ViewBuilder
    private var modelsSubmenu: some View {
        Menu {
            // The loadable list only (requirement 3.3).
            let loadable = model.loadableModels
            if loadable.isEmpty {
                if let statement = model.notConnectedStatement {
                    Text(verbatim: statement)
                } else {
                    Text("No models installed", bundle: .module)
                    Button {
                        model.showMainWindow(navigatingTo: .modelHub)
                    } label: { Text("Discover Models...", bundle: .module) }
                }
            } else {
                ForEach(loadable) { installedModel in
                    let isAttached = isModelAttached(installedModel)
                    let isCurrentChat = model.selected?.path == installedModel.path
                    Button {
                        // A menu cannot host the "Load anyway" question: for a
                        // warned model the main window comes forward to ask.
                        model.requestLoad(installedModel, bringWindowForward: true)
                    } label: {
                        if isAttached {
                            Label { Text("\(installedModel.alias) (Active)", bundle: .module) } icon: { Image(systemName: "checkmark") }
                        } else if isCurrentChat {
                            Label { Text("\(installedModel.alias) (Loaded)", bundle: .module) } icon: { Image(systemName: "checkmark") }
                        } else {
                            Text(verbatim: installedModel.alias)
                        }
                    }
                }
            }
        } label: {
            Text("Model", bundle: .module)
        }
    }

    @ViewBuilder
    private var navigationSection: some View {
        Button {
            model.showMainWindow(navigatingTo: .server)
        } label: { Text("Admin Panel", bundle: .module) }

        Button {
            model.showMainWindow(navigatingTo: .chat)
        } label: { Text("Chat with TurboSpark", bundle: .module) }
    }

    @ViewBuilder
    private var appManagementSection: some View {
        Button {
            model.openSettings(tab: .general)
        } label: { Text("Preferences...", bundle: .module) }
        .keyboardShortcut(",", modifiers: .command)

        Button {
            NSApp.activate(ignoringOtherApps: true)
            NSApp.orderFrontStandardAboutPanel(nil)
        } label: { Text("About TurboSpark", bundle: .module) }

        Divider()

        Button {
            NSApp.terminate(nil)
        } label: { Text("Quit TurboSpark", bundle: .module) }
        .keyboardShortcut("q", modifiers: .command)
    }

    // MARK: - Helpers

    private var statusTitle: Text {
        if model.serverBusy {
            return isRunning
                ? Text("Stopping TurboSpark Server...", bundle: .module)
                : Text("Starting TurboSpark Server...", bundle: .module)
        }
        if isRunning {
            if info?.models.isEmpty ?? true {
                return Text("TurboSpark Server (no model)", bundle: .module)
            }
            return Text("TurboSpark Server is running", bundle: .module)
        }
        if let daemon = daemonStatus, daemon.running {
            return Text("TurboSpark CLI Daemon is running", bundle: .module)
        }
        return Text("TurboSpark Server is stopped", bundle: .module)
    }

    private var statusSymbol: String {
        if isRunning {
            return "circle.fill"
        }
        if let daemon = daemonStatus, daemon.running {
            return "circle.fill"
        }
        return "circle"
    }

    private func isModelAttached(_ installedModel: InstalledModel) -> Bool {
        let servedID = AppModel.servedModelID(for: installedModel)
        return model.serverInfo?.models.contains(servedID) ?? false
    }

    private func copyToClipboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func formatNumber(_ value: UInt64) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }

    private func uptimeText(_ seconds: UInt64) -> String {
        if seconds < 60 { return "\(seconds)s" }
        if seconds < 3600 { return "\(seconds / 60)m" }
        return String(format: "%dh %dm", seconds / 3600, (seconds % 3600) / 60)
    }
}
