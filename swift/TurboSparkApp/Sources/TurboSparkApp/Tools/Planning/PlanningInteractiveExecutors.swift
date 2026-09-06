import Foundation

// MARK: - AskUserQuestion Executor

public enum AskUserQuestionExecutor {
    /// Handler invoked when user questions are asked by a model.
    public static var onQuestionsAsked: (@Sendable (UUID?, [UserQuestionItem]) -> Void)?

    /// Active pending answers by chat ID or session.
    private static let lock = NSLock()
    private static var pendingQuestions: [String: [UserQuestionItem]] = [:]

    public static func parseQuestions(from arguments: [String: String]) throws -> [UserQuestionItem] {
        if let jsonRaw = arguments["questions"], let data = jsonRaw.data(using: .utf8) {
            let decoder = JSONDecoder()
            if let list = try? decoder.decode([UserQuestionItem].self, from: data) {
                return list
            }
        }
        if let jsonStr = arguments["questions_json"], let data = jsonStr.data(using: .utf8) {
            let decoder = JSONDecoder()
            if let list = try? decoder.decode([UserQuestionItem].self, from: data) {
                return list
            }
        }
        // Direct single question shorthand
        if let singleQ = arguments["question"] {
            let header = arguments["header"] ?? "Question"
            var options: [UserQuestionOption] = []
            if let optsRaw = arguments["options"], let data = optsRaw.data(using: .utf8) {
                let decoder = JSONDecoder()
                if let decodedOpts = try? decoder.decode([UserQuestionOption].self, from: data) {
                    options = decodedOpts
                } else if let strOpts = try? decoder.decode([String].self, from: data) {
                    options = strOpts.map { UserQuestionOption(label: $0, description: $0) }
                }
            }
            if options.isEmpty {
                options = [
                    UserQuestionOption(label: "Yes", description: "Confirm and proceed"),
                    UserQuestionOption(label: "No", description: "Decline and adjust")
                ]
            }
            let multi = (arguments["multiSelect"]?.lowercased() == "true" || arguments["multi_select"]?.lowercased() == "true")
            return [UserQuestionItem(question: singleQ, header: header, options: options, multiSelect: multi)]
        }
        throw NSError(
            domain: "TurboSparkTool",
            code: 30,
            userInfo: [NSLocalizedDescriptionKey: "Missing or invalid 'questions' parameter for AskUserQuestion."]
        )
    }

    public static func execute(arguments: [String: String], chatID: UUID?) throws -> String {
        let items = try parseQuestions(from: arguments)
        guard !items.isEmpty else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 30,
                userInfo: [NSLocalizedDescriptionKey: "No questions provided in AskUserQuestion call."]
            )
        }

        let key = chatID?.uuidString ?? "default"
        lock.lock()
        pendingQuestions[key] = items
        lock.unlock()

        onQuestionsAsked?(chatID, items)

        var rendered = "Presented \(items.count) question(s) to the user:\n"
        for (i, q) in items.enumerated() {
            rendered += "\n\(i + 1). [\(q.header)] \(q.question)"
            for opt in q.options {
                rendered += "\n   - \(opt.label): \(opt.description)"
            }
        }
        return rendered
    }
}

// MARK: - PlanMode Executor

public enum PlanModeExecutor {
    public static var onPlanModeChanged: (@Sendable (UUID?, Bool, String?) -> Void)?

    private static let lock = NSLock()
    private static var activePlanModes: Set<String> = []
    private static var savedPlans: [String: String] = [:]

    public static func isPlanModeActive(for chatID: UUID?) -> Bool {
        let key = chatID?.uuidString ?? "default"
        lock.lock()
        defer { lock.unlock() }
        return activePlanModes.contains(key)
    }

    public static func enter(arguments: [String: String], chatID: UUID?) -> String {
        let key = chatID?.uuidString ?? "default"
        lock.lock()
        activePlanModes.insert(key)
        lock.unlock()

        onPlanModeChanged?(chatID, true, nil)
        return "Entered plan mode. In this mode, explore the codebase and formulate an architectural plan before modifying files."
    }

    public static func exit(arguments: [String: String], chatID: UUID?) -> String {
        let key = chatID?.uuidString ?? "default"
        let plan = arguments["plan"] ?? arguments["summary"] ?? arguments["content"]

        lock.lock()
        activePlanModes.remove(key)
        if let plan {
            savedPlans[key] = plan
        }
        lock.unlock()

        onPlanModeChanged?(chatID, false, plan)
        if let plan, !plan.isEmpty {
            return "Exited plan mode. Finalized plan submitted for review:\n\n\(plan)"
        }
        return "Exited plan mode. Ready to proceed with execution."
    }
}

// MARK: - ReportFindings Executor

public enum ReportFindingsExecutor {
    public static func parseFindings(from arguments: [String: String]) throws -> (level: String, findings: [CodeFindingItem]) {
        let level = arguments["level"] ?? arguments["effort_level"] ?? "medium"
        var items: [CodeFindingItem] = []

        if let jsonRaw = arguments["findings"], let data = jsonRaw.data(using: .utf8) {
            let decoder = JSONDecoder()
            if let decoded = try? decoder.decode([CodeFindingItem].self, from: data) {
                items = decoded
            }
        }

        if items.isEmpty {
            if let summary = arguments["summary"] ?? arguments["description"] ?? arguments["findings"] {
                let file = arguments["file"] ?? arguments["file_path"] ?? "workspace"
                let line = Int(arguments["line"] ?? "")
                let scenario = arguments["failure_scenario"] ?? arguments["scenario"] ?? "Identified during review."
                let cat = arguments["category"]
                let verdict = arguments["verdict"]
                items.append(CodeFindingItem(file: file, line: line, summary: summary, failureScenario: scenario, category: cat, verdict: verdict))
            }
        }

        guard !items.isEmpty else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 31,
                userInfo: [NSLocalizedDescriptionKey: "Missing or invalid 'findings' array in ReportFindings call."]
            )
        }
        return (level, items)
    }

    public static func execute(arguments: [String: String]) throws -> String {
        let (level, items) = try parseFindings(from: arguments)
        var out = "Reported \(items.count) finding(s) (effort level: \(level)):\n"
        for (i, f) in items.enumerated() {
            let lineStr = f.line.map { ":\($0)" } ?? ""
            out += "\n\(i + 1). [\(f.category ?? "Defect")] \(f.file)\(lineStr)"
            out += "\n   Summary: \(f.summary)"
            out += "\n   Failure Scenario: \(f.failureScenario)"
            if let v = f.verdict {
                out += "\n   Verdict: \(v)"
            }
        }
        return out
    }
}

// MARK: - ProposeSkills Executor

public enum ProposeSkillsExecutor {
    public static func execute(arguments: [String: String], projectRootURL: URL?) throws -> String {
        var proposals: [SkillProposalItem] = []
        if let jsonRaw = arguments["proposals"], let data = jsonRaw.data(using: .utf8) {
            let decoder = JSONDecoder()
            if let decoded = try? decoder.decode([SkillProposalItem].self, from: data) {
                proposals = decoded
            }
        }
        if proposals.isEmpty, let name = arguments["name"], let skillMd = arguments["skillMd"] ?? arguments["skill_md"] ?? arguments["content"] ?? arguments["rules"] ?? arguments["description"] {
            let desc = arguments["description"] ?? "Proposed skill"
            let kind = arguments["kind"] ?? "new"
            proposals.append(SkillProposalItem(name: name, kind: kind, description: desc, skillMd: skillMd))
        }

        guard !proposals.isEmpty else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 32,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'proposals' or skill draft content in ProposeSkills."]
            )
        }

        let isUserScope = (arguments["scope"]?.lowercased() == "user") || (projectRootURL == nil) || (projectRootURL?.path == "/")
        let skillsDir: URL
        let scopeLabel: String
        if isUserScope {
            skillsDir = SkillManager.shared.defaultUserSkillsDirectory
            // The Default profile's user scope is the shared
            // ~/.turbospark/skills; another profile's is inside its own
            // folder, so the label reports the actual directory.
            scopeLabel = "user scope (\(skillsDir.path)/)"
        } else {
            guard let root = projectRootURL else {
                throw NSError(domain: "TurboSparkTool", code: 32, userInfo: [NSLocalizedDescriptionKey: "Project root required for project-scoped skill."])
            }
            skillsDir = root.appendingPathComponent(".turbospark/skills", isDirectory: true)
            scopeLabel = "project scope (.turbospark/skills/)"
        }

        try FileManager.default.createDirectory(at: skillsDir, withIntermediateDirectories: true)

        var savedCount = 0
        for p in proposals {
            let sanitizedName = p.name.trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "/", with: "-")
                .replacingOccurrences(of: "\\", with: "-")
                .lowercased()
            guard !sanitizedName.isEmpty, sanitizedName != ".", sanitizedName != ".." else { continue }
            let skillFolder = skillsDir.appendingPathComponent(sanitizedName, isDirectory: true)
            try FileManager.default.createDirectory(at: skillFolder, withIntermediateDirectories: true)
            let mdFile = skillFolder.appendingPathComponent("SKILL.md")
            try p.skillMd.write(to: mdFile, atomically: true, encoding: .utf8)
            savedCount += 1
        }

        SkillManager.shared.invalidateResolutionCache()
        return "Successfully saved \(savedCount) proposed skill(s) to \(scopeLabel)."
    }
}

// MARK: - ProposeGoal Executor

public enum ProposeGoalExecutor {
    public static func execute(arguments: [String: String]) throws -> String {
        guard let condition = arguments["condition"] ?? arguments["goal"] ?? arguments["target"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 33,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'condition' parameter for ProposeGoal."]
            )
        }
        let askUser = (arguments["ask_user"]?.lowercased() != "false")
        return "Proposed goal: \"\(condition)\" (Requires confirmation: \(askUser ? "Yes" : "No"))."
    }
}

// MARK: - SendFeedback Executor

public enum SendFeedbackExecutor {
    public static func execute(arguments: [String: String]) throws -> String {
        let type = arguments["type"] ?? arguments["category"] ?? "general"
        let title = arguments["title"] ?? arguments["summary"] ?? arguments["feedback"] ?? "Feedback"
        let details = arguments["details"] ?? arguments["description"] ?? arguments["feedback"] ?? title
        let area = arguments["area"] ?? "general"
        return "Diagnostic feedback recorded: [\(type.uppercased())] \(title) (Area: \(area))\nDetails: \(details)"
    }
}
