import Foundation
import SwiftUI

// MARK: - MCP Server Management

extension AppModel {
    /// Adds a new global MCP server.
    public func addGlobalMcpServer(_ config: McpServerConfig) {
        globalMcpServers.append(config)
        persistGlobalMcpServers()
    }

    /// Updates an existing global MCP server.
    public func updateGlobalMcpServer(_ config: McpServerConfig) {
        if let index = globalMcpServers.firstIndex(where: { $0.id == config.id }) {
            globalMcpServers[index] = config
            persistGlobalMcpServers()
        }
    }

    /// Deletes a global MCP server by ID.
    public func deleteGlobalMcpServer(id: UUID) {
        globalMcpServers.removeAll { $0.id == id }
        persistGlobalMcpServers()
    }

    /// Toggles the enabled state of a global MCP server.
    public func toggleGlobalMcpServer(id: UUID, isEnabled: Bool) {
        if let index = globalMcpServers.firstIndex(where: { $0.id == id }) {
            globalMcpServers[index].isEnabled = isEnabled
            globalMcpServers[index].updatedAt = Date()
            persistGlobalMcpServers()
        }
    }

    /// Adds or imports an MCP server into a specific project.
    public func addProjectMcpServer(projectID: UUID, config: McpServerConfig) {
        if let index = projects.firstIndex(where: { $0.id == projectID }) {
            projects[index].mcpServers.append(config)
            projects[index].updatedAt = Date()
            persistProjects()
        }
    }

    /// Updates an MCP server within a specific project.
    public func updateProjectMcpServer(projectID: UUID, config: McpServerConfig) {
        if let pIndex = projects.firstIndex(where: { $0.id == projectID }),
           let sIndex = projects[pIndex].mcpServers.firstIndex(where: { $0.id == config.id }) {
            projects[pIndex].mcpServers[sIndex] = config
            projects[pIndex].updatedAt = Date()
            persistProjects()
        }
    }

    /// Deletes an MCP server from a specific project.
    public func deleteProjectMcpServer(projectID: UUID, serverID: UUID) {
        if let pIndex = projects.firstIndex(where: { $0.id == projectID }) {
            projects[pIndex].mcpServers.removeAll { $0.id == serverID }
            projects[pIndex].updatedAt = Date()
            persistProjects()
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
        }
    }

    /// Tests connection to an MCP server and queries its available tools.
    public func testMcpServer(_ config: McpServerConfig, workingDirectory: URL? = nil) async -> Result<[McpDiscoveredTool], Error> {
        do {
            let tools = try await McpClientEngine.shared.discoverTools(for: config, workingDirectory: workingDirectory)
            return .success(tools)
        } catch {
            return .failure(error)
        }
    }
}
