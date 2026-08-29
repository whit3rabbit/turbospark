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

/// Utility for scanning, detecting, and resolving AGENTS.md and CLAUDE.md files in project directories.
public enum ProjectRuleDetector {
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
                let content = readText(at: agentsURL) ?? readText(at: claudeURL) ?? ""
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
            let agentsContent = (readText(at: agentsURL) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let claudeContent = (readText(at: claudeURL) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)

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
        if agentsExists, let content = readText(at: agentsURL) {
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
        if claudeExists, let content = readText(at: claudeURL) {
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
               let content = readText(at: fileURL) {
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

    private static func readText(at url: URL) -> String? {
        let canonical = url.resolvingSymlinksInPath()
        if let data = try? Data(contentsOf: canonical),
           let str = String(data: data, encoding: .utf8) {
            return str
        }
        if let str = try? String(contentsOf: url, encoding: .utf8) {
            return str
        }
        return nil
    }
}
