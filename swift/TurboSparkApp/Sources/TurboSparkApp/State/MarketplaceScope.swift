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

/// Whether a marketplace just FETCHED for browsing may be remembered under the
/// name its remote manifest declares. That name is attacker-controlled, so a
/// fetch must never repoint an existing entry (a trusted "company-skills"
/// source) at a different source; the manifest is still shown either way.
public enum FetchedMarketplacePersistence: Equatable, Sendable {
    case save
    case alreadySaved
    case nameTakenByDifferentSource

    public static func decide(
        name: String, source: MarketplaceSource, existing: [String: MarketplaceSource]
    ) -> FetchedMarketplacePersistence {
        guard let current = existing[name] else { return .save }
        return current == source ? .alreadySaved : .nameTakenByDifferentSource
    }
}

@MainActor
extension AppModel {
    /// Remembers a browsed marketplace unless that would overwrite another
    /// source's entry. Never throws: a failed save must not hide the manifest.
    @discardableResult
    public func saveFetchedMarketplace(
        name: String, source: MarketplaceSource, kind: MarketplaceKind, projectID: UUID?
    ) -> FetchedMarketplacePersistence {
        let decision = FetchedMarketplacePersistence.decide(
            name: name, source: source,
            existing: marketplaceSources(kind: kind, projectID: projectID))
        if decision == .save {
            try? saveMarketplace(name: name, source: source, kind: kind, projectID: projectID)
        }
        return decision
    }

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
                throw PluginLoadError(pluginName: nil, reason: String(localized: "The target project no longer exists.", bundle: .module))
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
            throw PluginLoadError(pluginName: nil, reason: String(localized: "The target project no longer exists.", bundle: .module))
        }
    }

    public func restoreMarketplace(name: String, kind: MarketplaceKind, projectID: UUID) {
        guard var project = projects.first(where: { $0.id == projectID }) else { return }
        project.marketplaces.hidden[kind.rawValue, default: []].remove(name)
        updateProject(project)
    }
}
