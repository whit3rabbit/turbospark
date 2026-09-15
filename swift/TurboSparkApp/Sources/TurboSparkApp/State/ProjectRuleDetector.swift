import Foundation

/// Preference policy when resolving project instruction files (AGENTS.md vs CLAUDE.md).
public enum AppRulePreference: String, Codable, CaseIterable, Identifiable, Sendable {
    case agentsFirst = "agents_first"
    case claudeFirst = "claude_first"
    case mergeBoth = "merge_both"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .agentsFirst:
            return "Prefer AGENTS.md"
        case .claudeFirst:
            return "Prefer CLAUDE.md"
        case .mergeBoth:
            return "Merge Both Files"
        }
    }

    public var shortLabel: String {
        switch self {
        case .agentsFirst:
            return "AGENTS.md"
        case .claudeFirst:
            return "CLAUDE.md"
        case .mergeBoth:
            return "Merge"
        }
    }

    public var descriptionText: String {
        switch self {
        case .agentsFirst:
            return "Uses AGENTS.md if both exist. Falls back to CLAUDE.md or other rule files."
        case .claudeFirst:
            return "Uses CLAUDE.md if both exist. Falls back to AGENTS.md or other rule files."
        case .mergeBoth:
            return "Combines both AGENTS.md and CLAUDE.md into a merged instruction set."
        }
    }
}

/// Detailed result of a project rules scan.
public struct ProjectRulesDetectionResult: Equatable, Sendable {
    /// Extracted and trimmed instructions text.
    public let content: String
    /// Names of rule files detected in the directory.
    public let detectedFiles: [String]
    /// Whether both AGENTS.md and CLAUDE.md existed with conflicting/different contents.
    public let hasConflict: Bool
    /// Whether a symbolic link was detected and resolved.
    public let isSymlink: Bool
    /// Canonical path of the resolved primary rules file if applicable.
    public let resolvedCanonicalPath: String?
    /// User-visible status summary message describing the detection outcome.
    public let statusDescription: String

    public init(
        content: String,
        detectedFiles: [String],
        hasConflict: Bool,
        isSymlink: Bool,
        resolvedCanonicalPath: String? = nil,
        statusDescription: String
    ) {
        self.content = content
        self.detectedFiles = detectedFiles
        self.hasConflict = hasConflict
        self.isSymlink = isSymlink
        self.resolvedCanonicalPath = resolvedCanonicalPath
        self.statusDescription = statusDescription
    }
}

/// Repository instructions resolved afresh for a project turn.
///
/// The project model deliberately does not persist this text. Repository
/// instruction files are configuration, so a later turn must see an edit to
/// them without requiring the user to reopen and save the project sheet.
public struct ProjectLiveInstructions: Equatable, Sendable {
    /// Combined, bounded instruction content ready to be added to a prompt.
    public let content: String
    /// Names of instruction files that contributed content.
    public let detectedFiles: [String]
    /// Whether any contributing file was a symbolic link.
    public let isSymlink: Bool

    public init(content: String, detectedFiles: [String], isSymlink: Bool) {
        self.content = content
        self.detectedFiles = detectedFiles
        self.isSymlink = isSymlink
    }
}

/// Utility for scanning, detecting, and resolving project context files.
public enum ProjectRuleDetector {
    /// Resolves the repository instruction files for one prompt assembly.
    ///
    /// `AGENTS.md` and `CLAUDE.md` retain the project's selected conflict
    /// preference. `CONTEXT.md` and `SOUL.md` are complementary context, so they are
    /// appended whenever present rather than competing with either rule file.
    /// Every contributing file remains contained in the project root, even
    /// when it is a symlink.
    public static func liveInstructions(
        in directoryPath: String,
        preference: AppRulePreference = .agentsFirst,
        maxCharacters: Int = 8000
    ) -> ProjectLiveInstructions? {
        let trimmedPath = directoryPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty, maxCharacters > 0 else { return nil }

        let rootURL = URL(fileURLWithPath: trimmedPath, isDirectory: true)
        let rules = detectRules(
            in: trimmedPath,
            preference: preference,
            maxCharacters: maxCharacters)
        var content = rules?.content.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        var files = rules?.detectedFiles ?? []
        var containsSymlink = rules?.isSymlink ?? false

        let remainingCharacters = maxCharacters - content.count
        let contextURL = rootURL.appendingPathComponent("CONTEXT.md")
        let fileManager = FileManager.default
        if remainingCharacters > 0,
           fileExistsOrSymlink(at: contextURL, fileManager: fileManager),
           let context = (readText(at: contextURL, containedIn: rootURL))?.trimmingCharacters(in: .whitespacesAndNewlines),
           !context.isEmpty {
            let separator = content.isEmpty ? "" : "\n\n"
            let contextHeader = "# CONTEXT.md\n"
            let availableForContext = max(
                0,
                remainingCharacters - separator.count - contextHeader.count)
            if availableForContext > 0 {
                content += separator + contextHeader + String(context.prefix(availableForContext))
                files.append("CONTEXT.md")
                containsSymlink = containsSymlink || isSymlink(at: contextURL, fileManager: fileManager)
            }
        }

        let remainingAfterContext = maxCharacters - content.count
        let soulURL = rootURL.appendingPathComponent("SOUL.md")
        if remainingAfterContext > 0,
           fileExistsOrSymlink(at: soulURL, fileManager: fileManager),
           let soul = (readText(at: soulURL, containedIn: rootURL))?.trimmingCharacters(in: .whitespacesAndNewlines),
           !soul.isEmpty {
            let separator = content.isEmpty ? "" : "\n\n"
            let soulHeader = "# SOUL.md\n"
            let availableForSoul = max(
                0,
                remainingAfterContext - separator.count - soulHeader.count)
            if availableForSoul > 0 {
                content += separator + soulHeader + String(soul.prefix(availableForSoul))
                files.append("SOUL.md")
                containsSymlink = containsSymlink || isSymlink(at: soulURL, fileManager: fileManager)
            }
        }

        guard !content.isEmpty else { return nil }
        return ProjectLiveInstructions(
            content: content,
            detectedFiles: files,
            isSymlink: containsSymlink)
    }

    /// Scans a directory and returns detailed detection result with symlink resolution and preference handling.
    public static func detectRules(
        in directoryPath: String,
        preference: AppRulePreference = .agentsFirst,
        maxCharacters: Int = 8000
    ) -> ProjectRulesDetectionResult? {
        let trimmedPath = directoryPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty else { return nil }

        let fileManager = FileManager.default
        let rootURL = URL(fileURLWithPath: trimmedPath, isDirectory: true)

        let agentsURL = rootURL.appendingPathComponent("AGENTS.md")
        let claudeURL = rootURL.appendingPathComponent("CLAUDE.md")

        let agentsExists = fileExistsOrSymlink(at: agentsURL, fileManager: fileManager)
        let claudeExists = fileExistsOrSymlink(at: claudeURL, fileManager: fileManager)

        let agentsIsSymlink = isSymlink(at: agentsURL, fileManager: fileManager)
        let claudeIsSymlink = isSymlink(at: claudeURL, fileManager: fileManager)

        // Case 1: Both AGENTS.md and CLAUDE.md are present in some form
        if agentsExists && claudeExists {
            let agentsCanonical = agentsURL.resolvingSymlinksInPath().path
            let claudeCanonical = claudeURL.resolvingSymlinksInPath().path

            // Check if both files point to the exact same canonical target (e.g. symlink)
            if agentsCanonical == claudeCanonical {
                let content = readText(at: agentsURL, containedIn: rootURL) ?? readText(at: claudeURL, containedIn: rootURL) ?? ""
                let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { return nil }

                let symlinkName = agentsIsSymlink ? "AGENTS.md" : (claudeIsSymlink ? "CLAUDE.md" : "symlink")
                let targetName = agentsIsSymlink ? "CLAUDE.md" : "AGENTS.md"
                let desc: String
                if agentsIsSymlink || claudeIsSymlink {
                    desc = "Detected \(targetName) (\(symlinkName) is a symlink)"
                } else {
                    desc = "Detected AGENTS.md and CLAUDE.md (identical target)"
                }

                return ProjectRulesDetectionResult(
                    content: String(trimmed.prefix(maxCharacters)),
                    detectedFiles: ["AGENTS.md", "CLAUDE.md"],
                    hasConflict: false,
                    isSymlink: true,
                    resolvedCanonicalPath: agentsCanonical,
                    statusDescription: desc
                )
            }

            // Both exist as distinct files
            let agentsContent = (readText(at: agentsURL, containedIn: rootURL) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let claudeContent = (readText(at: claudeURL, containedIn: rootURL) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)

            // If contents happen to be identical despite different paths, treat as non-conflicting
            if agentsContent == claudeContent && !agentsContent.isEmpty {
                return ProjectRulesDetectionResult(
                    content: String(agentsContent.prefix(maxCharacters)),
                    detectedFiles: ["AGENTS.md", "CLAUDE.md"],
                    hasConflict: false,
                    isSymlink: agentsIsSymlink || claudeIsSymlink,
                    resolvedCanonicalPath: agentsCanonical,
                    statusDescription: "Detected AGENTS.md and CLAUDE.md (identical content)"
                )
            }

            // Real conflict between distinct files
            switch preference {
            case .agentsFirst:
                if !agentsContent.isEmpty {
                    return ProjectRulesDetectionResult(
                        content: String(agentsContent.prefix(maxCharacters)),
                        detectedFiles: ["AGENTS.md", "CLAUDE.md"],
                        hasConflict: true,
                        isSymlink: agentsIsSymlink,
                        resolvedCanonicalPath: agentsCanonical,
                        statusDescription: "Loaded AGENTS.md (preferred over CLAUDE.md)"
                    )
                } else if !claudeContent.isEmpty {
                    return ProjectRulesDetectionResult(
                        content: String(claudeContent.prefix(maxCharacters)),
                        detectedFiles: ["AGENTS.md", "CLAUDE.md"],
                        hasConflict: false,
                        isSymlink: claudeIsSymlink,
                        resolvedCanonicalPath: claudeCanonical,
                        statusDescription: "Loaded CLAUDE.md (AGENTS.md was empty)"
                    )
                }

            case .claudeFirst:
                if !claudeContent.isEmpty {
                    return ProjectRulesDetectionResult(
                        content: String(claudeContent.prefix(maxCharacters)),
                        detectedFiles: ["CLAUDE.md", "AGENTS.md"],
                        hasConflict: true,
                        isSymlink: claudeIsSymlink,
                        resolvedCanonicalPath: claudeCanonical,
                        statusDescription: "Loaded CLAUDE.md (preferred over AGENTS.md)"
                    )
                } else if !agentsContent.isEmpty {
                    return ProjectRulesDetectionResult(
                        content: String(agentsContent.prefix(maxCharacters)),
                        detectedFiles: ["CLAUDE.md", "AGENTS.md"],
                        hasConflict: false,
                        isSymlink: agentsIsSymlink,
                        resolvedCanonicalPath: agentsCanonical,
                        statusDescription: "Loaded AGENTS.md (CLAUDE.md was empty)"
                    )
                }

            case .mergeBoth:
                var merged = ""
                if !agentsContent.isEmpty && !claudeContent.isEmpty {
                    merged = "# AGENTS.md Instructions\n\(agentsContent)\n\n---\n# CLAUDE.md Instructions\n\(claudeContent)"
                } else if !agentsContent.isEmpty {
                    merged = agentsContent
                } else {
                    merged = claudeContent
                }

                guard !merged.isEmpty else { return nil }
                return ProjectRulesDetectionResult(
                    content: String(merged.prefix(maxCharacters)),
                    detectedFiles: ["AGENTS.md", "CLAUDE.md"],
                    hasConflict: true,
                    isSymlink: agentsIsSymlink || claudeIsSymlink,
                    resolvedCanonicalPath: nil,
                    statusDescription: "Merged AGENTS.md and CLAUDE.md rules"
                )
            }
        }

        // Case 2: Only AGENTS.md exists
        if agentsExists, let content = readText(at: agentsURL, containedIn: rootURL) {
            let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                let desc = agentsIsSymlink ? "Loaded AGENTS.md (symlink resolved)" : "Loaded AGENTS.md"
                return ProjectRulesDetectionResult(
                    content: String(trimmed.prefix(maxCharacters)),
                    detectedFiles: ["AGENTS.md"],
                    hasConflict: false,
                    isSymlink: agentsIsSymlink,
                    resolvedCanonicalPath: agentsURL.resolvingSymlinksInPath().path,
                    statusDescription: desc
                )
            }
        }

        // Case 3: Only CLAUDE.md exists
        if claudeExists, let content = readText(at: claudeURL, containedIn: rootURL) {
            let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                let desc = claudeIsSymlink ? "Loaded CLAUDE.md (symlink resolved)" : "Loaded CLAUDE.md"
                return ProjectRulesDetectionResult(
                    content: String(trimmed.prefix(maxCharacters)),
                    detectedFiles: ["CLAUDE.md"],
                    hasConflict: false,
                    isSymlink: claudeIsSymlink,
                    resolvedCanonicalPath: claudeURL.resolvingSymlinksInPath().path,
                    statusDescription: desc
                )
            }
        }

        // Case 4: Fallback candidates (.rules, RULES.md, .cursorrules)
        let fallbackCandidates = [".rules", "RULES.md", ".cursorrules"]
        for candidate in fallbackCandidates {
            let fileURL = rootURL.appendingPathComponent(candidate)
            if fileExistsOrSymlink(at: fileURL, fileManager: fileManager),
               let content = readText(at: fileURL, containedIn: rootURL) {
                let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    let symlink = isSymlink(at: fileURL, fileManager: fileManager)
                    return ProjectRulesDetectionResult(
                        content: String(trimmed.prefix(maxCharacters)),
                        detectedFiles: [candidate],
                        hasConflict: false,
                        isSymlink: symlink,
                        resolvedCanonicalPath: fileURL.resolvingSymlinksInPath().path,
                        statusDescription: "Loaded \(candidate)"
                    )
                }
            }
        }

        return nil
    }

    private static func fileExistsOrSymlink(at url: URL, fileManager: FileManager) -> Bool {
        if fileManager.fileExists(atPath: url.path) {
            return true
        }
        if (try? fileManager.destinationOfSymbolicLink(atPath: url.path)) != nil {
            return true
        }
        return false
    }

    private static func isSymlink(at url: URL, fileManager: FileManager) -> Bool {
        if let values = try? url.resourceValues(forKeys: [.isSymbolicLinkKey]), values.isSymbolicLink == true {
            return true
        }
        if (try? fileManager.destinationOfSymbolicLink(atPath: url.path)) != nil {
            return true
        }
        return false
    }

    /// Decodes a byte range that may end mid-character.
    ///
    /// **A BOUNDED READ CUTS UTF-8 WHEREVER THE BYTE COUNT LANDS.** Both read
    /// branches took exactly `maxBytes` and decoded with no fallback, so a
    /// `CLAUDE.md` over 1 MB whose 65,536th byte fell inside a multi-byte
    /// character decoded to nil and the project loaded NO rules at all -- the
    /// worst outcome for a file whose whole job is to state them. Trimming
    /// back to the last valid boundary costs at most three bytes.
    private static func decodeTruncatedUTF8(_ data: Data) -> String? {
        if let str = String(data: data, encoding: .utf8) {
            return str
        }
        // A UTF-8 sequence is at most 4 bytes, so at most 3 can be dangling.
        for drop in 1...3 where data.count > drop {
            if let str = String(data: data.dropLast(drop), encoding: .utf8) {
                return str
            }
        }
        return nil
    }

    /// Reads a rules file, refusing one that resolves outside the project.
    ///
    /// **A SYMLINK IS FOLLOWED WHEREVER IT POINTS, AND THIS TEXT GOES INTO THE
    /// SYSTEM PROMPT** (state#23). `resolvingSymlinksInPath` was applied and the result
    /// read unconditionally, so a cloned repository shipping
    /// `AGENTS.md -> ~/.aws/credentials` put that file's first 64 KB into
    /// every turn's prompt -- exfiltration through a file the user never
    /// opened. `resolveSecurePath` has had this containment check since
    /// 2026-08-28 (swift/CLAUDE.md Gotcha 11); this reader predates it and
    /// never got one.
    ///
    /// A symlink INSIDE the project still resolves: this repository's own
    /// `CLAUDE.md` is a symlink to `AGENTS.md`, and refusing that would break
    /// the common case. The ROOT is resolved too, or a project under a
    /// symlinked path (`/tmp` is one on macOS) fails its own containment test.
    private static func readText(at url: URL, containedIn root: URL, maxBytes: Int = 65536) -> String? {
        // `PathContainment` rather than the four lines this used to spell
        // inline (state#39): `SkillParser` and `AgentParser` read the same
        // class of file out of the same untrusted clone and had no check,
        // which is easier to notice when the rule has a name.
        guard let canonical = PathContainment.resolvedIfContained(url, in: root) else {
            return nil
        }
        let fileManager = FileManager.default
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: canonical.path, isDirectory: &isDir), !isDir.boolValue else {
            return nil
        }

        // Size check: bound reads to avoid materializing huge files or device nodes
        if let attrs = try? fileManager.attributesOfItem(atPath: canonical.path),
           let size = attrs[.size] as? UInt64, size > 1_048_576 {
            guard let handle = try? FileHandle(forReadingFrom: canonical) else { return nil }
            defer { try? handle.close() }
            guard let data = try? handle.read(upToCount: maxBytes) else { return nil }
            return decodeTruncatedUTF8(data)
        }

        if let handle = try? FileHandle(forReadingFrom: canonical) {
            defer { try? handle.close() }
            if let data = try? handle.read(upToCount: maxBytes),
               let str = decodeTruncatedUTF8(data) {
                return str
            }
        }

        // **THE THIRD BRANCH UNDID THE FIRST TWO** (state#108). Both reads
        // above are bounded by `maxBytes`; this one read the WHOLE file with
        // no bound at all, and it was reached whenever the bounded read
        // returned nil -- which includes every file whose first `maxBytes`
        // are not valid UTF-8, i.e. exactly the binaries and device nodes the
        // bound exists for. A rules file that cannot be read as bounded UTF-8
        // is not a rules file.
        return nil
    }
}
