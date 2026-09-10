import Foundation

public enum MarketplaceKind: String, Codable, CaseIterable, Sendable { case plugins, skills, mcp }

public struct ProjectMarketplaces: Codable, Equatable, Sendable {
    public var sources: [String: [String: MarketplaceSource]] = [:]
    public var hidden: [String: Set<String>] = [:]

    public func resolved(_ user: [String: MarketplaceSource], kind: MarketplaceKind) -> [String: MarketplaceSource] {
        user.filter { !(hidden[kind.rawValue] ?? []).contains($0.key) }
            .merging(sources[kind.rawValue] ?? [:]) { _, project in project }
    }
}

@MainActor
extension AppModel {
    public func marketplaceSources(kind: MarketplaceKind, projectID: UUID?) -> [String: MarketplaceSource] {
        let user: [String: MarketplaceSource]
        switch kind {
        case .plugins: user = PluginMarketplaceManager.shared.loadKnownMarketplaces()
        case .skills: user = SkillMarketplaceManager.shared.loadKnownMarketplaces()
        case .mcp: user = McpMarketplaceManager.shared.registeredSources()
        }
        return projects.first { $0.id == projectID }?.marketplaces.resolved(user, kind: kind) ?? user
    }

    public func saveMarketplace(name: String, source: MarketplaceSource, kind: MarketplaceKind, projectID: UUID?) throws {
        if let reason = PluginManifestParser.validateMarketplaceName(name) {
            throw PluginLoadError(pluginName: nil, reason: reason)
        }
        if let projectID {
            guard var project = projects.first(where: { $0.id == projectID }) else {
                throw PluginLoadError(pluginName: nil, reason: "The target project no longer exists.")
            }
            project.marketplaces.sources[kind.rawValue, default: [:]][name] = source
            project.marketplaces.hidden[kind.rawValue, default: []].remove(name)
            updateProject(project)
        } else {
            switch kind {
            case .plugins: try PluginMarketplaceManager.shared.saveKnownMarketplace(name: name, source: source)
            case .skills: try SkillMarketplaceManager.shared.saveKnownMarketplace(name: name, source: source)
            case .mcp:
                guard McpMarketplaceManager.shared.register(name: name, source: source) else {
                    throw PluginLoadError(pluginName: nil, reason: "Could not save marketplace.")
                }
            }
        }
    }

    public func removeMarketplace(name: String, kind: MarketplaceKind, projectID: UUID?) throws {
        if let projectID, var project = projects.first(where: { $0.id == projectID }) {
            project.marketplaces.sources[kind.rawValue, default: [:]].removeValue(forKey: name)
            project.marketplaces.hidden[kind.rawValue, default: []].insert(name)
            updateProject(project)
        } else if projectID == nil {
            switch kind {
            case .plugins: try PluginMarketplaceManager.shared.removeKnownMarketplace(name: name)
            case .skills: try SkillMarketplaceManager.shared.removeKnownMarketplace(name: name)
            case .mcp:
                guard McpMarketplaceManager.shared.unregister(name: name) else {
                    throw PluginLoadError(pluginName: nil, reason: "Could not remove marketplace source.")
                }
            }
        } else {
            throw PluginLoadError(pluginName: nil, reason: "The target project no longer exists.")
        }
    }

    public func restoreMarketplace(name: String, kind: MarketplaceKind, projectID: UUID) {
        guard var project = projects.first(where: { $0.id == projectID }) else { return }
        project.marketplaces.hidden[kind.rawValue, default: []].remove(name)
        updateProject(project)
    }
}
