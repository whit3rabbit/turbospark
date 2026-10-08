import Foundation

// MARK: - AskUserQuestion Executor

public enum AskUserQuestionExecutor {
    /// Handler invoked when user questions are asked by a model. The third
    /// argument is the tool-call id the questions came from, what the
    /// transcript card matches its interactive controls against.
    public static var onQuestionsAsked: (@Sendable (UUID?, [UserQuestionItem], UUID?) -> Void)?

    /// The app-side answer surface. Installed at startup by the app model;
    /// when nil (tests, subagent context without a host) the questions are
    /// recorded and the tool returns immediately, exactly as it did before
    /// interactive answering existed. The third argument is the tool-call id
    /// the questions came from, what the transcript card matches against.
    public static var answerWaiter:
        (@Sendable (UUID?, [UserQuestionItem], UUID?) async -> String)?

    /// Active pending answers by chat ID or session.
    private static let lock = NSLock()
    private static var pendingQuestions: [String: [UserQuestionItem]] = [:]
    /// One parked execution per chat key. `submitAnswer` resumes it with the
    /// user's reply; cancellation resumes it with the dismissed note.
    private static var answerContinuations: [String: CheckedContinuation<String, Never>] = [:]

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
            // The dispatch gate flattens JSON true to "1", so accept both spellings.
            let truthy: Set<String> = ["true", "1"]
            let multi = truthy.contains(arguments["multiSelect"]?.lowercased() ?? "")
                || truthy.contains(arguments["multi_select"]?.lowercased() ?? "")
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

        onQuestionsAsked?(chatID, items, nil)

        var rendered = "Presented \(items.count) question(s) to the user:\n"
        for (i, q) in items.enumerated() {
            rendered += "\n\(i + 1). [\(q.header)] \(q.question)"
            for opt in q.options {
                rendered += "\n   - \(opt.label): \(opt.description)"
            }
        }
        return rendered
    }

    /// The interactive path: like `execute`, but the tool result does not
    /// exist until the user answers. The registry routes here whenever the
    /// app installed `answerWaiter`, which is what makes the questions in
    /// the transcript tappable (qwen-code's AskUserQuestion card). A second
    /// call for the same chat while one is parked replaces the PENDING
    /// continuation bookkeeping only after the old one has been retired --
    /// which cannot happen here, because the registry executes one call per
    /// chat at a time.
    public static func executeAwaitingAnswer(
        arguments: [String: String], chatID: UUID?, toolCallID: UUID? = nil
    ) async throws -> String {
        let items = try parseQuestions(from: arguments)
        guard !items.isEmpty else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 30,
                userInfo: [NSLocalizedDescriptionKey: "No questions provided in AskUserQuestion call."]
            )
        }
        guard let waiter = answerWaiter else {
            return try execute(arguments: arguments, chatID: chatID)
        }

        let key = chatID?.uuidString ?? "default"
        lock.withLock {
            pendingQuestions[key] = items
        }

        let answer = await withTaskCancellationHandler {
            await waiter(chatID, items, toolCallID)
        } onCancel: {
            // Stop during a parked question dismisses it: the loop must
            // never wait on a card the user asked to cancel.
            submitAnswer(chatID: chatID, answers: [:], dismissed: true)
        }
        lock.withLock {
            pendingQuestions[key] = nil
        }
        return answer
    }

    /// Delivers the user's reply (or a dismissal) to the parked execution.
    /// Idempotent: without a parked continuation this is a no-op, so a
    /// double-click or a Stop race cannot crash a second resume.
    public static func submitAnswer(
        chatID: UUID?, answers: [String: String], dismissed: Bool = false
    ) {
        let key = chatID?.uuidString ?? "default"
        lock.lock()
        let continuation = answerContinuations.removeValue(forKey: key)
        lock.unlock()
        guard let continuation else { return }
        if dismissed {
            continuation.resume(returning: "The user dismissed the questions without answering.")
            return
        }
        var rendered = "The user answered:\n"
        for (question, answer) in answers {
            rendered += "\n\(question): \(answer)"
        }
        continuation.resume(returning: rendered)
    }

    /// Parks the calling task until `submitAnswer` fires for this chat. The
    /// executor hands this to `answerWaiter`'s caller, which presents the
    /// questions and comes back through here.
    public static func waitForAnswer(
        chatID: UUID?, continuation: CheckedContinuation<String, Never>
    ) {
        let key = chatID?.uuidString ?? "default"
        lock.lock()
        let previous = answerContinuations.updateValue(continuation, forKey: key)
        lock.unlock()
        // A stale continuation for the same key means an earlier question
        // was never retired; resuming it with a note keeps its task alive
        // instead of leaking it forever.
        previous?.resume(
            returning: "The user dismissed the questions without answering.")
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
        var skippedExisting: [String] = []
        for p in proposals {
            let sanitizedName = p.name.trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "/", with: "-")
                .replacingOccurrences(of: "\\", with: "-")
                .lowercased()
            guard !sanitizedName.isEmpty, sanitizedName != ".", sanitizedName != ".." else { continue }
            let skillFolder = skillsDir.appendingPathComponent(sanitizedName, isDirectory: true)
            let mdFile = skillFolder.appendingPathComponent("SKILL.md")
            // A proposal must never silently replace a skill the user already
            // has: a prompt-injected proposal named after a trusted skill
            // would otherwise swap its instructions.
            if FileManager.default.fileExists(atPath: mdFile.path) {
                skippedExisting.append(sanitizedName)
                continue
            }
            try FileManager.default.createDirectory(at: skillFolder, withIntermediateDirectories: true)
            try p.skillMd.write(to: mdFile, atomically: true, encoding: .utf8)
            savedCount += 1
        }

        SkillManager.shared.invalidateResolutionCache()
        var message = "Successfully saved \(savedCount) proposed skill(s) to \(scopeLabel)."
        if !skippedExisting.isEmpty {
            message += " Not saved because a skill with that name already exists: "
                + skippedExisting.joined(separator: ", ")
                + ". Existing skills are never overwritten by a proposal."
        }
        return message
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
        // No confirmation UI or goal wiring exists behind this tool, and a
        // "proposed, awaiting confirmation" reply made the model wait on, or
        // claim, a goal that was never created. Goals are set by the user
        // with /goal.
        throw NSError(
            domain: "TurboSparkTool",
            code: 34,
            userInfo: [NSLocalizedDescriptionKey:
                "ProposeGoal is not implemented: no goal was proposed or set (condition: \"\(condition)\"). "
                + "Ask the user to run /goal themselves."]
        )
    }
}

// MARK: - SendFeedback Executor

public enum SendFeedbackExecutor {
    public static func execute(arguments: [String: String]) throws -> String {
        // Nothing receives feedback in this client; saying "recorded" would
        // tell the model a report was filed when it went nowhere.
        throw NSError(
            domain: "TurboSparkTool",
            code: 35,
            userInfo: [NSLocalizedDescriptionKey:
                "SendFeedback is not implemented: nothing was recorded or sent. "
                + "Tell the user in your reply instead."]
        )
    }
}
