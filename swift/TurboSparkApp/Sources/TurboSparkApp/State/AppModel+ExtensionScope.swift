import Foundation

extension SkillScope {
    var projectRootURL: URL? {
        guard case .projectLocal(let path) = self, !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
}

extension SkillManager {
    public func ownsSkill(_ skill: AppSkill) -> Bool {
        let root: URL
        switch skill.scope {
        case .userGlobal: root = defaultUserSkillsDirectory
        case .projectLocal(let path):
            root = URL(fileURLWithPath: path).appendingPathComponent(".turbospark/skills")
        case .bundled, .plugin: return false
        }
        let prefix = root.standardizedFileURL.resolvingSymlinksInPath().path + "/"
        let target = (skill.skillDirectoryURL ?? skill.sourceURL)
            .standardizedFileURL.resolvingSymlinksInPath().path
        return target.hasPrefix(prefix)
    }

    func requireOwnedSkill(_ skill: AppSkill) throws {
        guard ownsSkill(skill) else {
            throw NSError(domain: "TurboSparkSkill", code: 5, userInfo: [NSLocalizedDescriptionKey:
                "Copy this skill into TurboSpark before editing. Its source belongs to another tool or plugin."])
        }
    }
}

extension AppModel {
    public func setPluginPreference(id: String, enabled: Bool?, scope: PluginInstallScope) {
        switch scope {
        case .user:
            pluginEnableState[id] = enabled
            persistSettings()
        case .project(let captured):
            guard var current = projects.first(where: { $0.id == captured.id }) else { return }
            current.enabledPlugins[id] = enabled
            updateProject(current)
        }
        pluginStateChanged()
    }

    public func pluginInstallationScopes(_ id: String) -> [PluginInstallScope] {
        (PluginLedgerStore(root: nil).load().plugins[id] ?? []).compactMap { record in
            if record.scope == "user" { return .user }
            guard let path = record.projectPath else { return nil }
            let project = projects.first { $0.rootDirectoryURL?.standardizedFileURL.path == path }
                ?? AppProject(name: URL(fileURLWithPath: path).lastPathComponent, rootDirectoryPath: path)
            return .project(project)
        }
    }

    @discardableResult
    public func uninstallPluginID(
        _ id: String, scope: PluginInstallScope,
        manager: PluginMarketplaceManager = .shared
    ) -> Bool {
        do {
            try manager.uninstall(pluginID: id, scope: scope.ledgerValue, projectRootURL: scope.projectRootURL)
            setPluginPreference(id: id, enabled: nil, scope: scope)
            showToast("Removed from \(scope.label).", style: .info)
            return true
        } catch {
            showToast("Uninstall failed: \(error.localizedDescription)", style: .error)
            return false
        }
    }

    public func setSkillOverride(name: String, enabled: Bool?, projectID: UUID) {
        guard var project = projects.first(where: { $0.id == projectID }) else { return }
        project.enabledSkills[name.lowercased()] = enabled
        updateProject(project)
        reloadSkills()
    }

    public func setMcpOverride(serverID: UUID, enabled: Bool?, projectID: UUID) {
        guard var project = projects.first(where: { $0.id == projectID }) else { return }
        project.enabledMcpServers[serverID.uuidString] = enabled
        updateProject(project)
        McpToolCatalogCache.shared.removeAll()
        refreshMcpToolCatalog(for: selectedProject)
    }
}
