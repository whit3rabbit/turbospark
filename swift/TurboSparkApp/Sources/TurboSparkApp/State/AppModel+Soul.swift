import Foundation

extension AppModel {
    /// The selected library row, suitable for rendering in Settings. A stale
    /// selection resolves to no row, never another entry.
    public var selectedSoulPrompt: AppSoulPrompt? {
        guard let selectedSoulPromptID else { return nil }
        return soulPrompts.first(where: { $0.id == selectedSoulPromptID })
    }

    /// The SOUL section injected into a turn's system prompt. Disabled or
    /// unselected contributes nothing: detected external SOUL.md files are
    /// never consumed without an explicit import into the library.
    public var resolvedSoulPrompt: String {
        guard soulPromptEnabled, let selectedSoulPrompt else { return "" }
        return selectedSoulPrompt.content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Known external SOUL.md files currently present on this machine.
    public var detectedSoulImportSources: [SoulPromptImportSource] {
        SoulPromptStore.detectedImportSources
    }

    /// Loads a saved soul or explicitly selects none.
    public func selectSoulPrompt(_ id: UUID?) {
        selectedSoulPromptID = id.flatMap { candidate in
            soulPrompts.contains(where: { $0.id == candidate }) ? candidate : nil
        }
        persistSettings()
    }

    /// Adds a new saved soul and selects it. A colliding name gets a numeric
    /// suffix so every "Save As New" creates a distinct row. Import does not
    /// enable SOUL: turning the section on stays an explicit choice.
    @discardableResult
    public func addSoulPrompt(name: String, content: String) -> AppSoulPrompt {
        let prompt = AppSoulPrompt(name: uniqueSoulName(name), content: content)
        soulPrompts.append(prompt)
        selectedSoulPromptID = prompt.id
        persistSettings()
        return prompt
    }

    /// Saves the content of the selected row.
    public func updateSoulPrompt(id: UUID, content: String) {
        guard let index = soulPrompts.firstIndex(where: { $0.id == id }) else { return }
        soulPrompts[index].content = content
        persistSettings()
    }

    /// Deletes a saved soul. Deleting the selected row explicitly leaves
    /// nothing selected rather than selecting a different entry.
    public func deleteSoulPrompt(_ id: UUID) {
        guard soulPrompts.contains(where: { $0.id == id }) else { return }
        soulPrompts.removeAll { $0.id == id }
        if selectedSoulPromptID == id {
            selectedSoulPromptID = nil
        }
        persistSettings()
    }

    /// Copies a detected external SOUL.md into the library. The external
    /// file is never modified or replaced.
    public func importSoul(from source: SoulPromptImportSource) throws {
        try importSoul(from: source.fileURL, name: source.displayName)
    }

    /// Copies a user-selected SOUL.md into the library under its file name.
    /// Re-importing refreshes the same-named row in place instead of
    /// stacking duplicates.
    public func importSoul(from fileURL: URL, name: String? = nil) throws {
        let content = try SoulPromptStore.read(fileURL: fileURL)
        let baseName = (name ?? fileURL.deletingPathExtension().lastPathComponent)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let entryName = baseName.isEmpty ? SoulPromptStore.fileName : baseName
        if let index = soulPrompts.firstIndex(where: { $0.name == entryName }) {
            soulPrompts[index].content = content
            selectedSoulPromptID = soulPrompts[index].id
        } else {
            let prompt = AppSoulPrompt(name: entryName, content: content)
            soulPrompts.append(prompt)
            selectedSoulPromptID = prompt.id
        }
        persistSettings()
    }

    /// Creates the missing Hermes file from the editor content. Existing
    /// Hermes content is never replaced by this action.
    public func createHermesSoul(content: String) throws {
        try SoulPromptStore.createHermes(content: content)
    }

    private func uniqueSoulName(_ raw: String) -> String {
        let base = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallback = base.isEmpty ? "SOUL" : base
        guard soulPrompts.contains(where: { $0.name == fallback }) else {
            return fallback
        }
        var suffix = 2
        while soulPrompts.contains(where: { $0.name == "\(fallback) \(suffix)" }) {
            suffix += 1
        }
        return "\(fallback) \(suffix)"
    }
}
