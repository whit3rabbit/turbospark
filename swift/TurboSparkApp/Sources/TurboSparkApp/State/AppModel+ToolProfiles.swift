import Foundation

/// Tool profiles: which tools an interactive turn is offered.
///
/// Three layers, narrowest wins: the selected profile (persisted), an unsaved
/// session override made from the composer popover, and in Chat mode the
/// master `chatToolsEnabled` switch. `captureTurnTools` reads `activeToolProfile`
/// and `AppToolCatalog.captureTurnAvailableTools` applies it.
extension AppModel {
    public var allToolProfiles: [AppToolProfile] {
        AppToolProfile.builtIns + customToolProfiles
    }

    /// The persisted selection. A stale ID resolves to Default, never to
    /// another profile, so removing a row cannot silently change what a turn
    /// is offered.
    public var selectedToolProfile: AppToolProfile {
        allToolProfiles.first { $0.id == activeToolProfileID } ?? .default
    }

    /// What turns actually use: the unsaved override when there is one.
    public var activeToolProfile: AppToolProfile {
        toolProfileOverride ?? selectedToolProfile
    }

    public func selectToolProfile(_ id: UUID) {
        guard allToolProfiles.contains(where: { $0.id == id }) else { return }
        activeToolProfileID = id
        // Choosing a profile is choosing it as saved, so the override ends.
        toolProfileOverride = nil
        persistSettings()
    }

    // MARK: Composer overrides

    /// Chat has no workspace root, so the tools that need one are neither
    /// listed nor counted there.
    public var toolsContextIsRootless: Bool { interactionMode == .chat }

    /// The tools the composer lists for this mode: those the turn's agent
    /// profile is ever offered, without the workspace-rooted ones in Chat.
    public var toolGroupsForContext: [ToolProfileGroup] {
        let agentType: AppAgentType = toolsContextIsRootless
            ? .general
            : (turnProject(chatID: selectedChatID)?.agentType ?? .coder)
        return AppToolCatalog.profileGroups(agentType: agentType, rootless: toolsContextIsRootless)
    }

    /// How many of this mode's tools the active profile offers, and how many
    /// there are.
    public var activeToolCounts: (enabled: Int, total: Int) {
        let keys = toolGroupsForContext.flatMap { $0.entries.map(\.id) }
        guard let allowed = activeToolProfile.allowedTools else { return (keys.count, keys.count) }
        let set = Set(allowed)
        return (keys.filter(set.contains).count, keys.count)
    }

    /// The dot: on when a turn would be offered at least one tool. In Chat the
    /// master switch also has to be on; Projects mode has no master switch but
    /// needs a project to run anything.
    public var toolsAreOn: Bool {
        if interactionMode == .chat && !chatToolsEnabled { return false }
        if interactionMode == .projects && turnProject(chatID: selectedChatID) == nil { return false }
        // MCP and custom tools are Projects-only, so they only count there.
        let mcp = !toolsContextIsRootless && activeToolProfile.includesMcpAndCustom
        return activeToolCounts.enabled > 0 || mcp
    }

    public func isToolEnabled(_ key: String) -> Bool {
        guard let allowed = activeToolProfile.allowedTools else { return true }
        return allowed.contains(key)
    }

    /// Flips one tool for this session only.
    public func setToolEnabled(_ key: String, _ enabled: Bool) {
        mutateToolOverride { profile, universe in
            var keys = Set(profile.allowedTools ?? universe)
            if enabled { keys.insert(key) } else { keys.remove(key) }
            profile.allowedTools = Array(keys)
        }
    }

    public func setMcpAndCustomEnabled(_ enabled: Bool) {
        mutateToolOverride { profile, _ in profile.includesMcpAndCustom = enabled }
    }

    public func resetToolOverride() {
        toolProfileOverride = nil
    }

    /// Applies an edit on top of the selected profile. An edit that lands back
    /// on the selected profile clears the override, so "modified" never shows
    /// for a profile that has not changed.
    private func mutateToolOverride(
        _ edit: (inout AppToolProfile, _ universe: [String]) -> Void
    ) {
        // The full universe, not this mode's: collapsing "everything the
        // composer lists" to "no allowlist" in Chat would switch the file and
        // shell tools on the moment the user changes to Projects.
        let universe = AppToolCatalog.profileGroups().flatMap { $0.entries.map(\.id) }
        var candidate = toolProfileOverride ?? selectedToolProfile
        edit(&candidate, universe)
        candidate = AppToolProfile(
            id: selectedToolProfile.id, name: selectedToolProfile.name,
            allowedTools: candidate.allowedTools, includesMcpAndCustom: candidate.includesMcpAndCustom)
        // Every tool listed is the same thing as no allowlist.
        if let allowed = candidate.allowedTools, Set(universe).isSubset(of: allowed) {
            candidate.allowedTools = nil
        }
        toolProfileOverride = candidate == selectedToolProfile ? nil : candidate
    }

    // MARK: Profile library

    /// Adds a profile and selects it. Returns its ID, or nil for an empty name.
    @discardableResult
    public func addToolProfile(
        name: String, allowedTools: [String]?, includesMcpAndCustom: Bool
    ) -> UUID? {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let profile = AppToolProfile(
            name: trimmed, allowedTools: allowedTools, includesMcpAndCustom: includesMcpAndCustom)
        customToolProfiles.append(profile)
        activeToolProfileID = profile.id
        toolProfileOverride = nil
        persistSettings()
        return profile.id
    }

    /// Copies any profile, built-in included, into an editable one.
    @discardableResult
    public func duplicateToolProfile(_ id: UUID) -> UUID? {
        guard let source = allToolProfiles.first(where: { $0.id == id }) else { return nil }
        let copyName = AppStrings.localized("\(source.displayName) copy")
        return addToolProfile(
            name: copyName, allowedTools: source.allowedTools,
            includesMcpAndCustom: source.includesMcpAndCustom)
    }

    /// Saves the composer's unsaved changes as a new profile.
    @discardableResult
    public func saveToolOverride(as name: String) -> UUID? {
        guard let override = toolProfileOverride else { return nil }
        return addToolProfile(
            name: name, allowedTools: override.allowedTools,
            includesMcpAndCustom: override.includesMcpAndCustom)
    }

    /// Edits a custom profile. Built-ins are refused: they are code-defined.
    public func updateToolProfile(
        id: UUID, name: String, allowedTools: [String]?, includesMcpAndCustom: Bool
    ) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let index = customToolProfiles.firstIndex(where: { $0.id == id })
        else { return }
        customToolProfiles[index] = AppToolProfile(
            id: id, name: trimmed, allowedTools: allowedTools,
            includesMcpAndCustom: includesMcpAndCustom)
        // A pending override was a diff against the old row.
        if activeToolProfileID == id { toolProfileOverride = nil }
        persistSettings()
    }

    /// Deletes a custom profile. Deleting the selected one returns to Default.
    public func deleteToolProfile(_ id: UUID) {
        guard customToolProfiles.contains(where: { $0.id == id }) else { return }
        customToolProfiles.removeAll { $0.id == id }
        if activeToolProfileID == id {
            activeToolProfileID = AppToolProfile.defaultID
            toolProfileOverride = nil
        }
        persistSettings()
    }
}
