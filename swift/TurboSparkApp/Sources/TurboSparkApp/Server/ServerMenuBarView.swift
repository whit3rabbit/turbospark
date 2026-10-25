import AppKit
import SwiftUI
import TurboSpark

/// The menu bar panel's submenus and its More menu.
///
/// `ServerMenuDashboardView` draws the panel itself (status, the server
/// switch, loaded models, navigation). This view owns what lives one level
/// down: the Coding Agents and API Endpoints menus, and as its `body` the
/// More menu with the background daemon controls, the full serving stats,
/// and About.
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
        daemonSection
        Divider()
        statsSubmenu
        Divider()
        Button {
            NSApp.activate(ignoringOtherApps: true)
            NSApp.orderFrontStandardAboutPanel(nil)
        } label: { Text("About TurboSpark", bundle: .module) }
    }

    // MARK: - Sections

    /// The background daemon: a separate process that outlives the app.
    @ViewBuilder
    private var daemonSection: some View {
        // Read once per render: the status call touches files and processes.
        let daemon = try? TurboSparkDaemon.status()
        if let daemon, daemon.running {
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
        } else {
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
                Text("Uptime: \(MenuBarUptime.text(info.uptimeSeconds))", bundle: .module)
            }
        } label: {
            Text("Serving Stats", bundle: .module)
        }
    }

    /// The Coding Agents menu's items, for the panel's own Menu row.
    @ViewBuilder
    var agentsMenuItems: some View {
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
    }

    /// The API Endpoints menu's items, for the panel's own Menu row.
    @ViewBuilder
    var endpointsMenuItems: some View {
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

    // MARK: - Helpers

    private func copyToClipboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func formatNumber(_ value: UInt64) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }
}
