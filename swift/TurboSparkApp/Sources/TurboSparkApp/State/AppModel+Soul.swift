import Foundation

extension AppModel {
    /// Resolves the live Hermes file before the per-profile fallback.
    public var soulPromptResolution: SoulPromptResolution {
        SoulPromptStore.resolve(fallback: soulPrompt)
    }

    /// The nonblank global SOUL content used by prompt assembly.
    public var resolvedSoulPrompt: String {
        soulPromptResolution.content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Known external SOUL.md files currently present on this machine.
    public var detectedSoulImportSources: [SoulPromptImportSource] {
        SoulPromptStore.detectedImportSources
    }

    /// Saves the currently active source. Hermes content is edited in place;
    /// the native fallback is persisted with the rest of this profile.
    public func saveSoulPrompt(_ content: String) throws {
        switch soulPromptResolution.source {
        case .hermes:
            try SoulPromptStore.writeHermes(content: content)
        case .turboSpark:
            soulPrompt = content
            persistSettings()
        }
    }

    /// Copies Hermes content into the native fallback without changing the
    /// Hermes file or changing the automatic source preference.
    public func importHermesSoul() throws {
        try importSoul(from: SoulPromptImportSource(
            kind: .hermes, fileURL: SoulPromptStore.hermesFileURL))
    }

    /// Copies a detected external SOUL.md into the native profile fallback.
    /// The external file is never modified or replaced.
    public func importSoul(from source: SoulPromptImportSource) throws {
        try importSoul(from: source.fileURL)
    }

    /// Copies a user-selected SOUL.md into the native profile fallback.
    public func importSoul(from fileURL: URL) throws {
        soulPrompt = try SoulPromptStore.read(fileURL: fileURL)
        persistSettings()
    }

    /// Creates the missing Hermes file from the native fallback. Existing
    /// Hermes content is never replaced by this action.
    public func createHermesSoul(content: String? = nil) throws {
        let value = content ?? soulPrompt
        try SoulPromptStore.createHermes(content: value)
        if let content, content != soulPrompt {
            soulPrompt = content
            persistSettings()
        }
    }
}
