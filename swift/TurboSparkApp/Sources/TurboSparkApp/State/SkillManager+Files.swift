import Foundation

/// Writing skills to disk: creating one, importing one, removing one.
///
/// Separate from discovery and precedence, which only ever READ. The
/// distinction is not cosmetic here -- these are the calls that take a
/// directory the user just picked, so they are where containment has to be
/// checked (state#107) and where a name that resolves outside the skills
/// directory has to be refused.
extension SkillManager {
    // MARK: - Skill Creation & File Management

    /// Creates and writes a new skill to disk in the given scope.
    @discardableResult
    public func createSkill(
        name: String,
        description: String,
        content: String,
        allowedTools: [String] = [],
        paths: [String] = [],
        scope: SkillScope,
        projectRootURL: URL? = nil
    ) throws -> AppSkill {
        // `..` survived the slash replacement, so `createSkill(name: "..")`
        // resolved to the skills directory's PARENT and wrote a SKILL.md
        // there. User-typed rather than model-controlled, so this is a
        // footgun rather than an escalation -- and cheaper to refuse than to
        // reason about.
        let sanitizedName = name
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: "\\", with: "-")
            .lowercased()
        guard !sanitizedName.isEmpty, sanitizedName != ".", sanitizedName != ".." else {
            throw NSError(
                domain: "TurboSparkSkill", code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "'\(name)' is not a usable skill name: it resolves outside the skills directory."
                ])
        }

        let targetDir: URL
        switch scope {
        case .userGlobal, .bundled:
            targetDir = defaultUserSkillsDirectory.appendingPathComponent(sanitizedName, isDirectory: true)
        case .projectLocal:
            guard let projectRootURL else {
                throw NSError(domain: "TurboSparkSkill", code: 1, userInfo: [NSLocalizedDescriptionKey: "Project root URL required for project-scoped skill."])
            }
            targetDir = projectRootURL
                .appendingPathComponent(".turbospark", isDirectory: true)
                .appendingPathComponent("skills", isDirectory: true)
                .appendingPathComponent(sanitizedName, isDirectory: true)
        case .plugin:
            throw NSError(domain: "TurboSparkSkill", code: 4, userInfo: [NSLocalizedDescriptionKey: "Skills contributed by a plugin are owned by that plugin and cannot be created here. Edit the plugin's own files or create the skill in user or project scope."])
        }

        try fileManager.createDirectory(at: targetDir, withIntermediateDirectories: true)
        let skillMdURL = targetDir.appendingPathComponent("SKILL.md")

        let manifest = SkillManifest(
            name: name,
            description: description,
            allowedTools: allowedTools,
            paths: paths
        )

        let appSkill = AppSkill(
            manifest: manifest,
            content: content,
            sourceURL: skillMdURL,
            skillDirectoryURL: targetDir,
            scope: scope,
            agentOrigin: .turboSpark,
            isEnabled: true
        )

        let serialized = SkillParser.serializeSkill(appSkill)
        try serialized.write(to: skillMdURL, atomically: true, encoding: .utf8)

        return appSkill
    }

    /// Saves modifications to an existing skill on disk.
    public func saveSkill(_ skill: AppSkill) throws {
        try requireOwnedSkill(skill)
        let serialized = SkillParser.serializeSkill(skill)
        let targetURL = skill.sourceURL
        let targetDir = targetURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: targetDir, withIntermediateDirectories: true)
        try serialized.write(to: targetURL, atomically: true, encoding: .utf8)
    }

    /// Deletes a skill and its containing folder if it is directory-based.
    public func deleteSkill(_ skill: AppSkill) throws {
        try requireOwnedSkill(skill)
        if skill.isDirectoryBased, let dirURL = skill.skillDirectoryURL {
            try fileManager.removeItem(at: dirURL)
        } else {
            try fileManager.removeItem(at: skill.sourceURL)
        }
        try SkillMarketplaceManager.shared.removeInstallationRecord(for: skill)
    }

    /// Imports a skill directory or file into TurboSpark user or project scope.
    @discardableResult
    /// - Parameter overwrite: whether to replace an existing skill of the
    ///   same name. Defaults to false so a collision is REPORTED; the import
    ///   used to delete the existing directory outright, which is
    ///   unrecoverable user content.
    public func importSkill(
        from sourceURL: URL,
        targetScope: SkillScope,
        projectRootURL: URL? = nil,
        overwrite: Bool = false
    ) throws -> AppSkill {
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: sourceURL.path, isDirectory: &isDir) else {
            throw SkillParseError.fileNotFound(sourceURL.path)
        }

        let destinationBaseDir: URL
        switch targetScope {
        case .userGlobal, .bundled:
            destinationBaseDir = defaultUserSkillsDirectory
        case .projectLocal:
            guard let projectRootURL else {
                throw NSError(domain: "TurboSparkSkill", code: 2, userInfo: [NSLocalizedDescriptionKey: "Project root required for project-scoped import."])
            }
            destinationBaseDir = projectRootURL
                .appendingPathComponent(".turbospark", isDirectory: true)
                .appendingPathComponent("skills", isDirectory: true)
        case .plugin:
            throw NSError(domain: "TurboSparkSkill", code: 5, userInfo: [NSLocalizedDescriptionKey: "A skill cannot be imported into plugin scope; plugins own their own files."])
        }

        try fileManager.createDirectory(at: destinationBaseDir, withIntermediateDirectories: true)

        if isDir.boolValue {
            let folderName = sourceURL.lastPathComponent
            let destFolderURL = destinationBaseDir.appendingPathComponent(folderName, isDirectory: true)
            // **AN IMPORT USED TO DELETE WHATEVER WAS ALREADY THERE.** A skill
            // directory is the user's own edited content; replacing it
            // silently on a name collision is unrecoverable. The caller
            // decides, by passing `overwrite`.
            // **PARSED BEFORE IT IS COPIED** (state#107). The copy ran
            // first and the parse second, so an unparseable skill -- or one
            // whose SKILL.md is a symlink out of the source directory -- was
            // already installed by the time the throw happened, and the
            // caller had no reason to think anything had been written. The
            // source's own directory is the containment root, which is the
            // check `parseFile` takes and this call site was not passing at
            // all (state#39's rule, unapplied at the one entry point that
            // takes an ARBITRARY directory the user just picked).
            let sourceSkillMd = sourceURL.appendingPathComponent("SKILL.md")
            let sourceFallbackMd = sourceURL.appendingPathComponent("skill.md")
            let sourceReadURL =
                fileManager.fileExists(atPath: sourceSkillMd.path)
                ? sourceSkillMd : sourceFallbackMd
            _ = try SkillParser.parseFile(
                at: sourceReadURL, scope: targetScope, agentOrigin: .custom,
                containedIn: sourceURL)

            if fileManager.fileExists(atPath: destFolderURL.path) {
                guard overwrite else {
                    throw SkillImportError.destinationExists(name: folderName)
                }
                try fileManager.removeItem(at: destFolderURL)
            }
            try fileManager.copyItem(at: sourceURL, to: destFolderURL)

            let skillMdURL = destFolderURL.appendingPathComponent("SKILL.md")
            let fallbackMdURL = destFolderURL.appendingPathComponent("skill.md")
            let readURL = fileManager.fileExists(atPath: skillMdURL.path) ? skillMdURL : fallbackMdURL

            return try SkillParser.parseFile(
                at: readURL, scope: targetScope, agentOrigin: .custom,
                containedIn: destFolderURL)
        } else {
            let fileName = sourceURL.lastPathComponent
            let destFileURL = destinationBaseDir.appendingPathComponent(fileName)
            // Source first, for the reason above (state#107).
            _ = try SkillParser.parseFile(
                at: sourceURL, scope: targetScope, agentOrigin: .custom,
                containedIn: sourceURL.deletingLastPathComponent())
            if fileManager.fileExists(atPath: destFileURL.path) {
                guard overwrite else {
                    throw SkillImportError.destinationExists(name: fileName)
                }
                try fileManager.removeItem(at: destFileURL)
            }
            try fileManager.copyItem(at: sourceURL, to: destFileURL)
            return try SkillParser.parseFile(
                at: destFileURL, scope: targetScope, agentOrigin: .custom,
                containedIn: destinationBaseDir)
        }
    }
}
