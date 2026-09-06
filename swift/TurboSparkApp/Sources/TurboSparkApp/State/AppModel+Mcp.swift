import Foundation
import SwiftUI

// MARK: - MCP Server Management

/// One repo-declared MCP server awaiting the user's approve/reject decision.
public struct PendingMcpServerApproval: Identifiable, Sendable, Equatable {
    public var id: UUID { approvalID }
    let approvalID = UUID()
    /// The project whose config file declared the server.
    public let projectID: UUID
    /// The server as parsed from the config file, with `autoApprove`
    /// forced off and `sourcePath` recording the declaring file.
    public let config: McpServerConfig
    /// The config file's path relative to the project root, for the sheet.
    public let sourceRelativePath: String

    public static func == (lhs: PendingMcpServerApproval, rhs: PendingMcpServerApproval) -> Bool {
        lhs.approvalID == rhs.approvalID
    }
}

extension AppModel {
    /// Adds a new global MCP server.
    public func addGlobalMcpServer(_ config: McpServerConfig) {
        globalMcpServers.append(config)
        persistGlobalMcpServers()
        refreshMcpToolCatalog(for: selectedProject)
    }

    /// Updates an existing global MCP server.
    public func updateGlobalMcpServer(_ config: McpServerConfig) {
        if let index = globalMcpServers.firstIndex(where: { $0.id == config.id }) {
            let renamedFrom = globalMcpServers[index].name
            globalMcpServers[index] = config
            persistGlobalMcpServers()
            // A rename changes the advertised `mcp__<server>__<tool>` key;
            // drop the old entry so no stale tools survive under it.
            if renamedFrom.caseInsensitiveCompare(config.name) != .orderedSame {
                McpToolCatalogCache.shared.removeServer(named: renamedFrom)
            }
            refreshMcpToolCatalog(for: selectedProject)
        }
    }

    /// Deletes a global MCP server by ID.
    public func deleteGlobalMcpServer(id: UUID) {
        if let removed = globalMcpServers.first(where: { $0.id == id }) {
            globalMcpServers.removeAll { $0.id == id }
            persistGlobalMcpServers()
            McpToolCatalogCache.shared.removeServer(named: removed.name)
        }
    }

    /// Toggles the enabled state of a global MCP server.
    public func toggleGlobalMcpServer(id: UUID, isEnabled: Bool) {
        if let index = globalMcpServers.firstIndex(where: { $0.id == id }) {
            globalMcpServers[index].isEnabled = isEnabled
            globalMcpServers[index].updatedAt = Date()
            persistGlobalMcpServers()
            if !isEnabled {
                // Disabled means unadvertised immediately, not on the next
                // refresh: a cached tools list must not outlive the switch.
                McpToolCatalogCache.shared.removeServer(named: globalMcpServers[index].name)
            }
            refreshMcpToolCatalog(for: selectedProject)
        }
    }

    /// Adds or imports an MCP server into a specific project.
    public func addProjectMcpServer(projectID: UUID, config: McpServerConfig) {
        if let index = projects.firstIndex(where: { $0.id == projectID }) {
            projects[index].mcpServers.append(config)
            projects[index].updatedAt = Date()
            persistProjects()
        }
        refreshMcpToolCatalog(for: projects.first(where: { $0.id == projectID }))
    }

    /// Updates an MCP server within a specific project.
    public func updateProjectMcpServer(projectID: UUID, config: McpServerConfig) {
        if let pIndex = projects.firstIndex(where: { $0.id == projectID }),
           let sIndex = projects[pIndex].mcpServers.firstIndex(where: { $0.id == config.id }) {
            let renamedFrom = projects[pIndex].mcpServers[sIndex].name
            projects[pIndex].mcpServers[sIndex] = config
            projects[pIndex].updatedAt = Date()
            persistProjects()
            if renamedFrom.caseInsensitiveCompare(config.name) != .orderedSame {
                McpToolCatalogCache.shared.removeServer(named: renamedFrom)
            }
        }
        refreshMcpToolCatalog(for: projects.first(where: { $0.id == projectID }))
    }

    /// Deletes an MCP server from a specific project.
    public func deleteProjectMcpServer(projectID: UUID, serverID: UUID) {
        if let pIndex = projects.firstIndex(where: { $0.id == projectID }),
           let removed = projects[pIndex].mcpServers.first(where: { $0.id == serverID }) {
            projects[pIndex].mcpServers.removeAll { $0.id == serverID }
            projects[pIndex].updatedAt = Date()
            persistProjects()
            McpToolCatalogCache.shared.removeServer(named: removed.name)
        }
    }

    /// Toggles the enabled state of an MCP server within a specific project.
    public func toggleProjectMcpServer(projectID: UUID, serverID: UUID, isEnabled: Bool) {
        if let pIndex = projects.firstIndex(where: { $0.id == projectID }),
           let sIndex = projects[pIndex].mcpServers.firstIndex(where: { $0.id == serverID }) {
            projects[pIndex].mcpServers[sIndex].isEnabled = isEnabled
            projects[pIndex].mcpServers[sIndex].updatedAt = Date()
            projects[pIndex].updatedAt = Date()
            persistProjects()
            if !isEnabled {
                McpToolCatalogCache.shared.removeServer(named: projects[pIndex].mcpServers[sIndex].name)
            }
        }
        refreshMcpToolCatalog(for: projects.first(where: { $0.id == projectID }))
    }

    /// Tests connection to an MCP server and queries its available tools.
    ///
    /// A successful discovery also WARMS the catalog cache: the tested
    /// config is the truth about this server, so its tools can be
    /// advertised without waiting for the next background refresh.
    public func testMcpServer(_ config: McpServerConfig, workingDirectory: URL? = nil) async -> Result<[McpDiscoveredTool], Error> {
        do {
            let tools = try await McpClientEngine.shared.discoverTools(for: config, workingDirectory: workingDirectory)
            McpToolCatalogCache.shared.setTools(tools, for: config)
            return .success(tools)
        } catch {
            return .failure(error)
        }
    }

    // MARK: - Tool catalog refresh

    /// Kicks off background `tools/list` discovery for every enabled server
    /// visible to `project`, warming the cache the prompt and catalog read.
    /// Cheap to call often: fresh entries and in-flight servers are skipped.
    public func refreshMcpToolCatalog(for project: AppProject?) {
        let servers = AppToolCatalogMcp.visibleServers(global: globalMcpServers, project: project)
        McpToolCatalogCache.shared.refreshEnabled(servers: servers, workingDirectory: project?.rootDirectoryURL)
    }

    // MARK: - Project approval lifecycle

    /// Scans `project`'s root (default: the selected project) for MCP
    /// config files and decides what to do with each declared server:
    /// already-imported or already-decided names are skipped, names covered
    /// by `approveAllProjectMcpServers` are imported silently (still
    /// `autoApprove = false`), and everything else lands in
    /// `pendingMcpApprovals` for the approval sheet.
    ///
    /// Detection scans the project ROOT only. The reference implementation
    /// also walks parent directories; importing servers declared OUTSIDE
    /// the workspace the user pointed at is the riskier behavior, so this
    /// port does not.
    public func detectProjectMcpServers(for candidate: AppProject? = nil) {
        guard let project = candidate ?? selectedProject,
              let root = project.rootDirectoryURL else { return }

        let detected = ProjectMcpDetector.detectInProject(rootURL: root)
        guard !detected.isEmpty else { return }

        var autoImports: [McpServerConfig] = []
        var approvedNames = project.approvedMcpJsonServers
        var pending: [PendingMcpServerApproval] = []

        for file in detected {
            for var server in file.servers {
                let alreadyImported = project.mcpServers.contains {
                    $0.name.lowercased() == server.name.lowercased()
                }
                let alreadyApproved = project.approvedMcpJsonServers.contains {
                    $0.caseInsensitiveCompare(server.name) == .orderedSame
                }
                let alreadyRejected = project.rejectedMcpJsonServers.contains {
                    $0.caseInsensitiveCompare(server.name) == .orderedSame
                }
                guard !alreadyImported, !alreadyApproved, !alreadyRejected else { continue }

                // The declaring file is the origin of record; the parse
                // keeps whatever path it was handed, which for a detection
                // hit is exactly this file.
                server.sourcePath = file.fileURL.path

                if project.approveAllProjectMcpServers {
                    autoImports.append(server)
                    if !approvedNames.contains(where: { $0.caseInsensitiveCompare(server.name) == .orderedSame }) {
                        approvedNames.append(server.name)
                    }
                } else {
                    let duplicate = pendingMcpApprovals.contains {
                        $0.projectID == project.id
                            && $0.config.name.lowercased() == server.name.lowercased()
                    }
                    if !duplicate {
                        pending.append(PendingMcpServerApproval(
                            projectID: project.id,
                            config: server,
                            sourceRelativePath: file.relativePath))
                    }
                }
            }
        }

        if !autoImports.isEmpty || approvedNames != project.approvedMcpJsonServers {
            if let index = projects.firstIndex(where: { $0.id == project.id }) {
                for server in autoImports
                where !projects[index].mcpServers.contains(where: { $0.name.lowercased() == server.name.lowercased() }) {
                    projects[index].mcpServers.append(server)
                }
                projects[index].approvedMcpJsonServers = approvedNames
                projects[index].updatedAt = Date()
                persistProjects()
            }
        }

        pendingMcpApprovals.append(contentsOf: pending)
    }

    /// Approves a pending repo-declared server: imports it into the project
    /// (enabled, never auto-approved) and records the decision so the same
    /// name never prompts again. `approveAllFuture` additionally imports
    /// every CURRENT-and-FUTURE server this project declares without
    /// prompting.
    public func approvePendingMcpServer(id: UUID, approveAllFuture: Bool = false) {
        guard let index = pendingMcpApprovals.firstIndex(where: { $0.id == id }) else { return }
        let approval = pendingMcpApprovals.remove(at: index)
        var config = approval.config
        config.isEnabled = true
        config.autoApprove = false

        if let pIndex = projects.firstIndex(where: { $0.id == approval.projectID }) {
            if !projects[pIndex].mcpServers.contains(where: { $0.name.lowercased() == config.name.lowercased() }) {
                projects[pIndex].mcpServers.append(config)
            }
            if !projects[pIndex].approvedMcpJsonServers.contains(where: { $0.caseInsensitiveCompare(config.name) == .orderedSame }) {
                projects[pIndex].approvedMcpJsonServers.append(config.name)
            }
            if approveAllFuture {
                projects[pIndex].approveAllProjectMcpServers = true
                // Everything the config already declares but the user has
                // not seen yet is covered by the same grant; import it now
                // rather than re-prompting one server at a time.
                for var sibling in ProjectMcpDetector.detectInProject(rootURL: projects[pIndex].rootDirectoryURL ?? URL(fileURLWithPath: "/")).flatMap({ $0.servers })
                where !projects[pIndex].mcpServers.contains(where: { $0.name.lowercased() == sibling.name.lowercased() })
                    && !projects[pIndex].rejectedMcpJsonServers.contains(where: { $0.caseInsensitiveCompare(sibling.name) == .orderedSame }) {
                    sibling.sourcePath = approval.config.sourcePath
                    sibling.isEnabled = true
                    sibling.autoApprove = false
                    projects[pIndex].mcpServers.append(sibling)
                }
            }
            projects[pIndex].updatedAt = Date()
            persistProjects()
            refreshMcpToolCatalog(for: projects[pIndex])
            // The all-future grant may have imported servers that still had
            // pending entries; their decision is made, so drop the cards.
            pendingMcpApprovals.removeAll { entry in
                entry.projectID == approval.projectID
                    && projects[pIndex].mcpServers.contains(where: {
                        $0.name.lowercased() == entry.config.name.lowercased()
                    })
            }
        }
    }

    /// Rejects a pending repo-declared server: nothing is imported and the
    /// name is recorded so it never prompts again.
    public func rejectPendingMcpServer(id: UUID) {
        guard let index = pendingMcpApprovals.firstIndex(where: { $0.id == id }) else { return }
        let approval = pendingMcpApprovals.remove(at: index)
        if let pIndex = projects.firstIndex(where: { $0.id == approval.projectID }) {
            if !projects[pIndex].rejectedMcpJsonServers.contains(where: { $0.caseInsensitiveCompare(approval.config.name) == .orderedSame }) {
                projects[pIndex].rejectedMcpJsonServers.append(approval.config.name)
                projects[pIndex].updatedAt = Date()
                persistProjects()
            }
        }
    }

    // MARK: - Persisted MCP permission rules

    /// Records a persistent allow/deny rule for an MCP server or one of its
    /// tools on `projectID`. Default resolution prefers the project the
    /// PENDING call was evaluated against -- the card can outlive a project
    /// switch, and the rule must land where the engine will read it -- then
    /// the selected project. No project anywhere means nowhere to persist
    /// the rule, so this is a no-op.
    public func addMcpPermissionRule(serverName: String, toolName: String?, allow: Bool, projectID: UUID? = nil) {
        guard let target = projectID ?? pendingToolCallProject?.id ?? selectedProjectID,
              let index = projects.firstIndex(where: { $0.id == target }) else { return }
        projects[index].permissions = projects[index].permissions.addingMcpRule(
            serverName: serverName, toolName: toolName, allow: allow)
        projects[index].updatedAt = Date()
        persistProjects()
    }
}
