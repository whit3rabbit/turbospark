import SwiftUI

/// Modal sheet for adding or editing an MCP server configuration.
public struct McpServerEditorSheet: View {
    let existingConfig: McpServerConfig?
    let workingDirectory: URL?
    let onSave: (McpServerConfig) -> Void
    let onDismiss: () -> Void

    @State private var name: String = ""
    @State private var transportType: String = "stdio"
    @State private var command: String = "npx"
    @State private var argsText: String = ""
    @State private var envText: String = ""
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
        onSave: @escaping (McpServerConfig) -> Void,
        onDismiss: @escaping () -> Void
    ) {
        self.existingConfig = existingConfig
        self.workingDirectory = workingDirectory
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
                    .font(.headline)
                Text("Configure Model Context Protocol server parameters and execution transport.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Close") { onDismiss() }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var basicSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Server Identity")
                .font(.subheadline.weight(.semibold))

            VStack(alignment: .leading, spacing: 4) {
                Text("Server Name")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("e.g. codebase-memory-mcp", text: $name)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Description (Optional)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("e.g. Knowledge graph and symbol search", text: $serverDescription)
                    .textFieldStyle(.roundedBorder)
            }
        }
    }

    private var transportSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Transport Type")
                .font(.subheadline.weight(.semibold))

            Picker("Transport", selection: $transportType) {
                Text("Local Subprocess (Stdio)").tag("stdio")
                Text("Remote Endpoint (SSE)").tag("sse")
            }
            .pickerStyle(.segmented)
        }
    }

    private var stdioDetailsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Subprocess Configuration")
                .font(.subheadline.weight(.semibold))

            VStack(alignment: .leading, spacing: 4) {
                Text("Command / Executable")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("npx, python3, uvx, docker, or absolute path", text: $command)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Arguments (one per line or space-separated)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: $argsText)
                    .font(.system(.caption, design: .monospaced))
                    .frame(height: 60)
                    .padding(4)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Environment Variables (KEY=VALUE per line)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: $envText)
                    .font(.system(.caption, design: .monospaced))
                    .frame(height: 60)
                    .padding(4)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5))
            }
        }
    }

    private var sseDetailsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("SSE Endpoint Configuration")
                .font(.subheadline.weight(.semibold))

            VStack(alignment: .leading, spacing: 4) {
                Text("Server URL")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("https://example.com/sse", text: $urlText)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Headers (Header: Value per line)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: $headersText)
                    .font(.system(.caption, design: .monospaced))
                    .frame(height: 60)
                    .padding(4)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5))
            }
        }
    }

    private var permissionsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Execution Settings")
                .font(.subheadline.weight(.semibold))

            Toggle("Enable Server", isOn: $isEnabled)
            Toggle("Auto-Approve Tools (skip interactive prompt)", isOn: $autoApprove)
        }
    }

    private var testConnectionSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Connection & Discovered Tools")
                    .font(.subheadline.weight(.semibold))
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
                .disabled(isTesting || name.trimmingCharacters(in: .whitespaces).isEmpty)
            }

            if let testResult {
                Text(testResult)
                    .font(.caption)
                    .foregroundStyle(testIsError ? .red : .green)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(testIsError ? Color.red.opacity(0.1) : Color.green.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            if !discoveredTools.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Discovered Tools (\(discoveredTools.count)):")
                        .font(.caption.weight(.medium))
                    ForEach(discoveredTools) { tool in
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: "wrench.fill")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .padding(.top, 2)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(tool.name)
                                    .font(.caption.monospaced().weight(.semibold))
                                Text(tool.description)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
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
            Spacer()
            Button("Save Server") {
                saveServer()
            }
            .buttonStyle(.borderedProminent)
            .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private func populateValues() {
        guard let config = existingConfig else { return }
        name = config.name
        isEnabled = config.isEnabled
        autoApprove = config.autoApprove
        serverDescription = config.serverDescription ?? ""
        discoveredTools = config.discoveredTools

        switch config.transport {
        case .stdio(let cmd, let args, let env):
            transportType = "stdio"
            command = cmd
            argsText = args.joined(separator: "\n")
            envText = env.map { "\($0.key)=\($0.value)" }.joined(separator: "\n")
        case .sse(let url, let headers):
            transportType = "sse"
            urlText = url.absoluteString
            headersText = headers.map { "\($0.key): \($0.value)" }.joined(separator: "\n")
        }
    }

    private func buildConfig() -> McpServerConfig {
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
            transport = .stdio(command: command.trimmingCharacters(in: .whitespaces), args: parsedArgs, env: envDict)
        } else {
            let cleanUrl = URL(string: urlText.trimmingCharacters(in: .whitespaces)) ?? URL(string: "http://localhost:8000/sse")!
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
        let config = buildConfig()
        onSave(config)
        onDismiss()
    }

    private func runConnectionTest() async {
        isTesting = true
        testResult = nil
        testIsError = false

        let config = buildConfig()
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
