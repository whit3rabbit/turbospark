import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Modal sheet for adding or editing an MCP server configuration.
@MainActor
public struct McpServerEditorSheet: View {
    let existingConfig: McpServerConfig?
    let workingDirectory: URL?
    /// Names already taken at the scope this server is being saved into,
    /// with the server BEING EDITED excluded by the caller so it can keep its
    /// own name.
    ///
    /// A server's NAME is its identity: `AppToolRegistry.executeMcpCall` picks
    /// with `first(where:)` and `AppToolPermissionEngine.evaluate` resolves its
    /// auto-approve arm the same way, so two servers sharing one name means the
    /// second is unreachable and the approval card cannot tell them apart.
    let existingNames: [String]
    let onSave: (McpServerConfig) -> Void
    let onDismiss: () -> Void

    @State private var name: String = ""
    @State private var transportType: String = "stdio"
    @State private var command: String = "npx"
    @State private var argsText: String = ""
    @State private var envText: String = ""
    @State private var cwdText: String = ""
    @State private var envPassthroughText: String = ""
    @State private var urlText: String = ""
    @State private var headersText: String = ""
    @State private var isEnabled: Bool = true
    @State private var autoApprove: Bool = false
    @State private var serverDescription: String = ""
    @State private var discoveredTools: [McpDiscoveredTool] = []

    @State private var isTesting: Bool = false
    @State private var testResult: String?
    @State private var testIsError: Bool = false

    public init(
        existingConfig: McpServerConfig? = nil,
        workingDirectory: URL? = nil,
        existingNames: [String] = [],
        onSave: @escaping (McpServerConfig) -> Void,
        onDismiss: @escaping () -> Void
    ) {
        self.existingConfig = existingConfig
        self.workingDirectory = workingDirectory
        self.existingNames = existingNames
        self.onSave = onSave
        self.onDismiss = onDismiss
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    basicSection
                    transportSection
                    if transportType == "stdio" {
                        stdioDetailsSection
                    } else {
                        sseDetailsSection
                    }
                    permissionsSection
                    testConnectionSection
                }
                .padding(20)
            }
            Divider()
            footer
        }
        .frame(width: 540, height: 640)
        .onAppear(perform: populateValues)
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(existingConfig == nil ? "Add MCP Server" : "Edit MCP Server")
                    .themedFont(.base, weight: .semibold)
                Text("Configure Model Context Protocol server parameters and execution transport.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Button("Close") { onDismiss() }
                .buttonStyle(.plain)
                .foregroundStyle(.appSecondary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var basicSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Server Identity", bundle: .module)
                .themedFont(.small, weight: .semibold)

            VStack(alignment: .leading, spacing: 4) {
                Text("Server Name", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                TextField("e.g. codebase-memory-mcp", text: $name)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Description (Optional)", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                TextField("e.g. Knowledge graph and symbol search", text: $serverDescription)
                    .textFieldStyle(.roundedBorder)
            }
        }
    }

    private var transportSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Transport Type", bundle: .module)
                .themedFont(.small, weight: .semibold)

            Picker("Transport", selection: $transportType) {
                Text("Local Subprocess (Stdio)", bundle: .module).tag("stdio")
                Text("Remote Endpoint (SSE)", bundle: .module).tag("sse")
            }
            .pickerStyle(.segmented)
        }
    }

    private var stdioDetailsSection: some View {
        McpStdioTransportFields(
            command: $command,
            argsText: $argsText,
            envText: $envText,
            cwdText: $cwdText,
            envPassthroughText: $envPassthroughText)
    }

    private var sseDetailsSection: some View {
        McpRemoteTransportFields(
            urlText: $urlText,
            headersText: $headersText,
            urlIsValid: parsedEndpointURL != nil)
    }

    private var permissionsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Execution Settings", bundle: .module)
                .themedFont(.small, weight: .semibold)

            Toggle("Enable Server", isOn: $isEnabled)
            Toggle("Auto-Approve Tools (skip interactive prompt)", isOn: $autoApprove)
        }
    }

    private var testConnectionSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Connection & Discovered Tools", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                Spacer()
                Button {
                    Task { await runConnectionTest() }
                } label: {
                    if isTesting {
                        ProgressView()
                            .scaleEffect(0.6)
                            .frame(width: 16, height: 16)
                    } else {
                        Label("Test Connection", systemImage: "bolt.fill")
                    }
                }
                .buttonStyle(.bordered)
                .disabled(isTesting || validationMessage != nil)
            }

            if let testResult {
                Text(testResult)
                    .themedFont(.small)
                    .foregroundStyle(testIsError ? .red : .green)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(testIsError ? Color.red.opacity(0.1) : Color.green.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            if !discoveredTools.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Discovered Tools (\(discoveredTools.count)):", bundle: .module)
                        .themedFont(.small, weight: .medium)
                    ForEach(discoveredTools) { tool in
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: "wrench.fill")
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                                .padding(.top, 2)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(tool.name)
                                    .themedCode(.small, weight: .semibold)
                                Text(tool.description)
                                    .themedFont(.tiny)
                                    .foregroundStyle(.appSecondary)
                                    .lineLimit(2)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
                .padding(8)
                .background(Color.secondary.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    private var footer: some View {
        HStack {
            Button("Cancel") { onDismiss() }
                .keyboardShortcut(.cancelAction)
            if let validationMessage {
                Text(validationMessage)
                    .themedFont(.small)
                    .foregroundStyle(.red)
                    .lineLimit(2)
            }
            Spacer()
            Button("Save Server") {
                saveServer()
            }
            .buttonStyle(.borderedProminent)
            .disabled(validationMessage != nil)
            .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    // MARK: - Validation

    /// The rules live in `McpServerFormValidation`, a pure value, so they can
    /// be tested without a window (Gotcha 26).
    private var validation: McpServerFormValidation {
        McpServerFormValidation(
            name: name,
            transport: transportType == "stdio" ? .stdio : .sse,
            command: command,
            endpointURLText: urlText,
            existingNames: existingNames)
    }

    private var parsedEndpointURL: URL? { validation.parsedEndpointURL }

    private var validationMessage: String? { validation.message }

    private func populateValues() {
        guard let config = existingConfig else { return }
        name = config.name
        isEnabled = config.isEnabled
        autoApprove = config.autoApprove
        serverDescription = config.serverDescription ?? ""
        discoveredTools = config.discoveredTools

        switch config.transport {
        case .stdio(let cmd, let args, let env, let cwd, let passthrough):
            transportType = "stdio"
            command = cmd
            argsText = args.joined(separator: "\n")
            envText = env.map { "\($0.key)=\($0.value)" }.joined(separator: "\n")
            cwdText = cwd ?? ""
            envPassthroughText = passthrough.joined(separator: "\n")
        case .sse(let url, let headers):
            transportType = "sse"
            urlText = url.absoluteString
            headersText = headers.map { "\($0.key): \($0.value)" }.joined(separator: "\n")
        }
    }

    private func buildConfig() -> McpServerConfig? {
        let cleanName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let transport: McpTransportSpec
        if transportType == "stdio" {
            let parsedArgs = argsText.components(separatedBy: .newlines)
                .flatMap { $0.components(separatedBy: " ") }
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }

            var envDict: [String: String] = [:]
            for line in envText.components(separatedBy: .newlines) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard let eqIndex = trimmed.firstIndex(of: "=") else { continue }
                let k = String(trimmed[..<eqIndex]).trimmingCharacters(in: .whitespaces)
                let v = String(trimmed[trimmed.index(after: eqIndex)...]).trimmingCharacters(in: .whitespaces)
                if !k.isEmpty {
                    envDict[k] = v
                }
            }
            let trimmedCwd = cwdText.trimmingCharacters(in: .whitespacesAndNewlines)
            let passthrough = envPassthroughText.components(separatedBy: .newlines)
                .flatMap { $0.components(separatedBy: " ") }
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }

            transport = .stdio(
                command: command.trimmingCharacters(in: .whitespaces),
                args: parsedArgs,
                env: envDict,
                cwd: trimmedCwd.isEmpty ? nil : trimmedCwd,
                envPassthrough: passthrough)
        } else {
            // Refuse rather than substitute: see `parsedEndpointURL`.
            guard let cleanUrl = parsedEndpointURL else { return nil }
            var headersDict: [String: String] = [:]
            for line in headersText.components(separatedBy: .newlines) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard let colonIndex = trimmed.firstIndex(of: ":") else { continue }
                let k = String(trimmed[..<colonIndex]).trimmingCharacters(in: .whitespaces)
                let v = String(trimmed[trimmed.index(after: colonIndex)...]).trimmingCharacters(in: .whitespaces)
                if !k.isEmpty {
                    headersDict[k] = v
                }
            }
            transport = .sse(url: cleanUrl, headers: headersDict)
        }

        return McpServerConfig(
            id: existingConfig?.id ?? UUID(),
            name: cleanName,
            transport: transport,
            isEnabled: isEnabled,
            autoApprove: autoApprove,
            sourcePath: existingConfig?.sourcePath,
            serverDescription: serverDescription.isEmpty ? nil : serverDescription,
            discoveredTools: discoveredTools,
            createdAt: existingConfig?.createdAt ?? Date(),
            updatedAt: Date()
        )
    }

    private func saveServer() {
        guard let config = buildConfig() else { return }
        onSave(config)
        onDismiss()
    }

    private func runConnectionTest() async {
        isTesting = true
        testResult = nil
        testIsError = false

        guard let config = buildConfig() else {
            testResult = validationMessage ?? "This server is not fully configured yet."
            testIsError = true
            isTesting = false
            return
        }
        do {
            let tools = try await McpClientEngine.shared.discoverTools(for: config, workingDirectory: workingDirectory)
            discoveredTools = tools
            testResult = "Connected successfully! Discovered \(tools.count) tool(s)."
            testIsError = false
        } catch {
            testResult = "Connection failed: \(error.localizedDescription)"
            testIsError = true
        }
        isTesting = false
    }
}
