import Foundation

/// Prompt budgeting and conditional path activation for skills.
///
/// Implements Claude Code's 1% context window character budget rule:
/// skill summaries are constrained so turn-1 cache tokens are not wasted,
/// and skills declaring file path globs are activated dynamically only when
/// matching files are accessed.
extension SkillManager {
    public static let defaultCharBudget = 8000
    public static let maxListingDescChars = 250
    public static let charsPerToken = 4
    public static let skillBudgetContextPercent = 0.01
    private static let minDescLength = 20

    /// Computes character budget for advertised skills based on model context size.
    public func charBudget(for contextWindowTokens: Int? = nil) -> Int {
        if let tokens = contextWindowTokens, tokens > 0 {
            return Int(Double(tokens * Self.charsPerToken) * Self.skillBudgetContextPercent)
        }
        return Self.defaultCharBudget
    }

    /// Truncates a description string cleanly with an ellipsis.
    private func truncateDesc(_ text: String, maxLength: Int) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count <= maxLength { return trimmed }
        let index = trimmed.index(trimmed.startIndex, offsetBy: max(0, maxLength - 3))
        return String(trimmed[..<index]) + "..."
    }

    /// The skills the model may see OFFERED: enabled, not
    /// `disable-model-invocation`, and -- for a skill declaring `paths` --
    /// activated this session by a matching file touch. Claude Code withholds
    /// conditional skills from the listing until the model touches a matching
    /// file; advertising an unactivated one reads as an invitation the tool
    /// then has no reason to honor.
    ///
    /// Every listing surface must resolve eligibility HERE, or two surfaces
    /// advertise two different sets: one that lists a user-disabled skill
    /// next to a tool that refuses it is a promise and a refusal in the same
    /// prompt.
    public func advertisedSkills(
        _ skills: [AppSkill],
        activatedSkillNames: Set<String>? = nil
    ) -> [AppSkill] {
        let activated = activatedSkillNames ?? activatedConditionalSkillNames
        return skills.filter { skill in
            guard skill.isEnabled else { return false }
            guard !(skill.manifest.disableModelInvocation ?? false) else { return false }
            return isSkillEligible(skill, activatedSkillNames: activated)
        }
    }

    /// One listing entry: `- name [Scope]: description - when_to_use`, the
    /// description truncated cleanly at `maxLength`.
    private func listingEntry(_ skill: AppSkill, maxLength: Int) -> String {
        var text = skill.skillDescription
        if let whenToUse = skill.manifest.whenToUse,
            !whenToUse.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            text += " - \(whenToUse)"
        }
        let desc = truncateDesc(text, maxLength: maxLength)
        return "- \(skill.name) [\(skill.scope.scopeTag)]: \(desc)"
    }

    /// Formats active skills into a concise list within the token/character budget.
    public func formatSkillsWithinBudget(
        _ skills: [AppSkill],
        contextWindowTokens: Int? = nil,
        activatedSkillNames: Set<String>? = nil
    ) -> String {
        // The advertised set, not merely the enabled set: conditional skills
        // stay hidden until activated, exactly as they are everywhere else.
        let eligible = advertisedSkills(skills, activatedSkillNames: activatedSkillNames)
        guard !eligible.isEmpty else { return "" }

        let budget = charBudget(for: contextWindowTokens)

        // Try full descriptions first
        let fullEntries = eligible.map { skill -> (skill: AppSkill, entry: String) in
            (skill, listingEntry(skill, maxLength: Self.maxListingDescChars))
        }

        let fullTotal = fullEntries.reduce(0) { $0 + $1.entry.count + 1 }
        if fullTotal <= budget {
            return fullEntries.map(\.entry).joined(separator: "\n")
        }

        // Over budget: preserve bundled skills, truncate user/project skills
        var bundledIndices = Set<Int>()
        var restIndices: [Int] = []

        for (idx, item) in fullEntries.enumerated() {
            if case .bundled = item.skill.scope {
                bundledIndices.insert(idx)
            } else {
                restIndices.append(idx)
            }
        }

        let bundledChars = fullEntries.enumerated()
            .filter { bundledIndices.contains($0.offset) }
            .reduce(0) { $0 + $1.element.entry.count + 1 }

        let remainingBudget = max(0, budget - bundledChars)

        if restIndices.isEmpty {
            return fullEntries.map(\.entry).joined(separator: "\n")
        }

        let restNameOverhead = restIndices.reduce(0) { sum, idx in
            sum + fullEntries[idx].skill.name.count + 4
        }
        let availableForDescs = remainingBudget - restNameOverhead
        let maxDescLen = availableForDescs / restIndices.count

        if maxDescLen < Self.minDescLength {
            // Extreme squeeze: output names only for non-bundled skills
            return eligible.enumerated().map { idx, skill in
                if bundledIndices.contains(idx) {
                    return fullEntries[idx].entry
                }
                return "- \(skill.name)"
            }.joined(separator: "\n")
        }

        // Truncate descriptions to maxDescLen
        return eligible.enumerated().map { idx, skill in
            if bundledIndices.contains(idx) {
                return fullEntries[idx].entry
            }
            return listingEntry(skill, maxLength: maxDescLen)
        }.joined(separator: "\n")
    }

    // MARK: - Conditional Path Activation

    /// Checks if a skill is unconditionally active or if its path pattern has been triggered.
    public func isSkillEligible(
        _ skill: AppSkill,
        activatedSkillNames: Set<String>
    ) -> Bool {
        // Skills without paths frontmatter are unconditional
        if skill.manifest.paths.isEmpty { return true }
        // Otherwise only eligible if previously activated
        return activatedSkillNames.contains(skill.name.lowercased())
    }

    /// Evaluates accessed file paths against conditional skills, returning newly activated skills.
    public func evaluateConditionalSkills(
        filePaths: [String],
        projectURL: URL?,
        currentlyActivated: Set<String>
    ) -> [AppSkill] {
        guard !filePaths.isEmpty else { return [] }
        let allSkills = resolveEffectiveSkills(projectURL: projectURL)
        var newlyActivated: [AppSkill] = []

        for skill in allSkills where !skill.manifest.paths.isEmpty {
            let key = skill.name.lowercased()
            if currentlyActivated.contains(key) { continue }

            for path in filePaths {
                if matchesPath(skill: skill, filePath: path) {
                    newlyActivated.append(skill)
                    break
                }
            }
        }

        return newlyActivated
    }

    // MARK: - Session Conditional Activation State

    private static let sessionLock = NSLock()
    private static var sessionActivatedSkills: Set<String> = []

    /// Records touched file paths and dynamically activates any matching conditional skills.
    public func notePathTouched(_ path: String, projectURL: URL?) {
        let clean = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }

        Self.sessionLock.lock()
        let current = Self.sessionActivatedSkills
        Self.sessionLock.unlock()

        let activated = evaluateConditionalSkills(filePaths: [clean], projectURL: projectURL, currentlyActivated: current)
        if !activated.isEmpty {
            Self.sessionLock.lock()
            for s in activated {
                Self.sessionActivatedSkills.insert(s.name.lowercased())
            }
            Self.sessionLock.unlock()
        }
    }

    /// Returns the currently activated conditional skill names.
    public var activatedConditionalSkillNames: Set<String> {
        Self.sessionLock.lock()
        defer { Self.sessionLock.unlock() }
        return Self.sessionActivatedSkills
    }

    /// Clears session-activated skills (called on new session or project change).
    public func clearActivatedConditionalSkills() {
        Self.sessionLock.lock()
        defer { Self.sessionLock.unlock() }
        Self.sessionActivatedSkills.removeAll()
    }
}
