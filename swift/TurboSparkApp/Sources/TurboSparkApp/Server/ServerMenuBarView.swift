import AppKit
import SwiftUI
import TurboSpark

/// Native macOS menu bar dropdown view for TurboSpark server management.
/// Inspired by the oMLX status item layout, providing quick server control,
/// listening address discovery, API endpoint clipboard copying, model
/// switching, and live serving statistics.
public struct ServerMenuBarView: View {
    @ObservedObject public var model: AppModel

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
        Label(statusTitle, systemImage: statusSymbol)

        if isRunning, let info = model.serverInfo {
            let address = info.baseURL?.absoluteString ?? "http://\(info.host):\(info.port)"
            Button("Listening: \(address)") {
                copyToClipboard(address)
                model.showToast("Copied server address", style: .info)
            }
        } else if let daemon = daemonStatus, daemon.running {
            let address = daemon.endpoint ?? "http://127.0.0.1:\(daemon.port ?? 8080)/v1"
            Button("CLI Daemon: \(address)") {
                copyToClipboard(address)
                model.showToast("Copied daemon address", style: .info)
            }
            if let logPath = daemon.logPath {
                Button("Reveal Daemon Log") {
                    ModelStorageManager.revealInFinder(path: logPath)
                }
            }
        } else {
            let portStr = model.serverPinnedPort == 0 ? "automatic" : "\(model.serverPinnedPort)"
            Text("Port: \(portStr)", bundle: .module)
        }
    }

    @ViewBuilder
    private var serverControlSection: some View {
        if isRunning {
            Button("Stop Server") {
                model.stopServer()
            }
            .disabled(model.serverBusy)
        } else if let daemon = daemonStatus, daemon.running {
            Button("Stop Background Daemon (PID \(daemon.pid ?? 0))") {
                try? TurboSparkDaemon.stop()
                model.showToast("Stopped background daemon", style: .info)
            }
            Button("Restart Background Daemon") {
                try? TurboSparkDaemon.restart()
                model.showToast("Restarted background daemon", style: .info)
            }
            Button("Start In-App Server") {
                model.startServer()
            }
            .disabled(model.serverBusy)
        } else {
            Button("Start Server") {
                model.startServer()
            }
            .disabled(model.serverBusy)
            Button("Start Background Daemon") {
                try? TurboSparkDaemon.start()
                model.showToast("Started background daemon", style: .info)
            }
        }
    }

    @ViewBuilder
    private var statsSubmenu: some View {
        Menu("Serving Stats") {
            let metrics = model.serverMetrics
            Text("Total Tokens Processed: \(formatNumber(metrics.totalTokensProcessed))", bundle: .module)
            Text("Requests Served: \(metrics.totalRequests)", bundle: .module)
            Text("In Flight: \(metrics.inFlight)", bundle: .module)
            if let pp = metrics.aggregatePromptTokensPerSecond {
                Text(String(format: "Avg PP Speed: %.1f tok/s", pp))
            }
            if let tg = metrics.aggregateTokensPerSecond {
                Text(String(format: "Avg TG Speed: %.1f tok/s", tg))
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
        }
    }

    @ViewBuilder
    private var agentsSubmenu: some View {
        Menu("Coding Agents") {
            let port = model.serverPinnedPort == 0 ? (info?.port ?? 8080) : model.serverPinnedPort
            let host = info?.host ?? "127.0.0.1"

            Text("Connect Terminal Agents:", bundle: .module)

            Button("Claude Code (claude)") {
                let cmd = TurboSparkAgent.launchCommand(for: "claude", host: host, port: port)
                copyToClipboard(cmd)
                model.showToast("Copied Claude Code export command", style: .info)
            }

            Button("OpenAI CLI / Codex (codex)") {
                let cmd = TurboSparkAgent.launchCommand(for: "codex", host: host, port: port)
                copyToClipboard(cmd)
                model.showToast("Copied Codex export command", style: .info)
            }

            Button("OpenCode (opencode)") {
                let cmd = TurboSparkAgent.launchCommand(for: "opencode", host: host, port: port)
                copyToClipboard(cmd)
                model.showToast("Copied OpenCode export command", style: .info)
            }

            Button("Hermes / OpenClaw") {
                let cmd = TurboSparkAgent.launchCommand(for: "hermes", host: host, port: port)
                copyToClipboard(cmd)
                model.showToast("Copied environment export command", style: .info)
            }

            Divider()

            Button("CLI Starter: turbospark start claude") {
                copyToClipboard("turbospark start claude")
                model.showToast("Copied turbospark start claude", style: .info)
            }
        }
    }

    @ViewBuilder
    private var endpointsSubmenu: some View {
        Menu("API Endpoints") {
            let base = info?.baseURL?.absoluteString
                ?? "http://127.0.0.1:\(model.serverPinnedPort == 0 ? 8080 : model.serverPinnedPort)"

            Button("Copy Base URL (\(base))") {
                copyToClipboard(base)
                model.showToast("Copied Base URL", style: .info)
            }

            Divider()

            Text("OpenAI Compatible:", bundle: .module)
            Button("POST /v1/chat/completions") {
                copyToClipboard("\(base)/v1/chat/completions")
                model.showToast("Copied /v1/chat/completions endpoint", style: .info)
            }
            Button("POST /v1/embeddings") {
                copyToClipboard("\(base)/v1/embeddings")
                model.showToast("Copied /v1/embeddings endpoint", style: .info)
            }
            Button("GET /v1/models") {
                copyToClipboard("\(base)/v1/models")
                model.showToast("Copied /v1/models endpoint", style: .info)
            }

            Divider()

            Text("Anthropic Compatible:", bundle: .module)
            Button("POST /v1/messages") {
                copyToClipboard("\(base)/v1/messages")
                model.showToast("Copied /v1/messages endpoint", style: .info)
            }

            Divider()

            Text("Ollama Compatible:", bundle: .module)
            Button("POST /api/chat") {
                copyToClipboard("\(base)/api/chat")
                model.showToast("Copied /api/chat endpoint", style: .info)
            }
            Button("POST /api/embeddings") {
                copyToClipboard("\(base)/api/embeddings")
                model.showToast("Copied /api/embeddings endpoint", style: .info)
            }
            Button("GET /api/tags") {
                copyToClipboard("\(base)/api/tags")
                model.showToast("Copied /api/tags endpoint", style: .info)
            }

            Divider()

            Text("Service:", bundle: .module)
            Button("GET /health") {
                copyToClipboard("\(base)/health")
                model.showToast("Copied /health endpoint", style: .info)
            }
        }
    }

    @ViewBuilder
    private var modelsSubmenu: some View {
        Menu("Model") {
            if model.installed.isEmpty {
                Text("No models installed", bundle: .module)
                Button("Discover Models...") {
                    model.showMainWindow(navigatingTo: .modelHub)
                }
            } else {
                ForEach(model.installed) { installedModel in
                    let isAttached = isModelAttached(installedModel)
                    let isCurrentChat = model.selected?.path == installedModel.path
                    Button {
                        model.selectModel(installedModel)
                    } label: {
                        if isAttached {
                            Label(installedModel.alias + " (Active)", systemImage: "checkmark")
                        } else if isCurrentChat {
                            Label(installedModel.alias + " (Loaded)", systemImage: "checkmark")
                        } else {
                            Text(installedModel.alias)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var navigationSection: some View {
        Button("Admin Panel") {
            model.showMainWindow(navigatingTo: .server)
        }

        Button("Chat with TurboSpark") {
            model.showMainWindow(navigatingTo: .chat)
        }
    }

    @ViewBuilder
    private var appManagementSection: some View {
        Button("Preferences...") {
            model.openSettings(tab: .general)
        }
        .keyboardShortcut(",", modifiers: .command)

        Button("About TurboSpark") {
            NSApp.activate(ignoringOtherApps: true)
            NSApp.orderFrontStandardAboutPanel(nil)
        }

        Divider()

        Button("Quit TurboSpark") {
            NSApp.terminate(nil)
        }
        .keyboardShortcut("q", modifiers: .command)
    }

    // MARK: - Helpers

    private var statusTitle: String {
        if model.serverBusy {
            return isRunning ? "Stopping TurboSpark Server..." : "Starting TurboSpark Server..."
        }
        if isRunning {
            if info?.models.isEmpty ?? true {
                return "TurboSpark Server (no model)"
            }
            return "TurboSpark Server is running"
        }
        if let daemon = daemonStatus, daemon.running {
            return "TurboSpark CLI Daemon is running"
        }
        return "TurboSpark Server is stopped"
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
