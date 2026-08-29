import Foundation

// MARK: - AskUserQuestion Tool

public struct UserQuestionOption: Codable, Sendable, Equatable {
    public var label: String
    public var description: String
    public var preview: String?

    public init(label: String, description: String, preview: String? = nil) {
        self.label = label
        self.description = description
        self.preview = preview
    }
}

public struct UserQuestionItem: Codable, Sendable, Equatable {
    public var question: String
    public var header: String
    public var options: [UserQuestionOption]
    public var multiSelect: Bool

    public init(
        question: String,
        header: String,
        options: [UserQuestionOption],
        multiSelect: Bool = false
    ) {
        self.question = question
        self.header = header
        self.options = options
        self.multiSelect = multiSelect
    }
}

public struct AskUserQuestionInput: Codable, Sendable, Equatable {
    public var questions: [UserQuestionItem]

    public init(questions: [UserQuestionItem]) {
        self.questions = questions
    }
}

public struct AskUserQuestionOutput: Codable, Sendable, Equatable {
    public var questions: [UserQuestionItem]
    public var answers: [String: String]
    public var response: String?

    public init(
        questions: [UserQuestionItem] = [],
        answers: [String: String] = [:],
        response: String? = nil
    ) {
        self.questions = questions
        self.answers = answers
        self.response = response
    }
}

// MARK: - Plan Mode Tools

public struct EnterPlanModeInput: Codable, Sendable, Equatable {
    public init() {}
}

public struct EnterPlanModeOutput: Codable, Sendable, Equatable {
    public var message: String

    public init(message: String = "Entered plan mode.") {
        self.message = message
    }
}

public struct ExitPlanModeInput: Codable, Sendable, Equatable {
    public init() {}
}

public struct ExitPlanModeOutput: Codable, Sendable, Equatable {
    public var plan: String?
    public var isAgent: Bool
    public var filePath: String?

    public init(plan: String? = nil, isAgent: Bool = false, filePath: String? = nil) {
        self.plan = plan
        self.isAgent = isAgent
        self.filePath = filePath
    }
}

// MARK: - ReportFindings Tool

public struct CodeFindingItem: Codable, Sendable, Equatable {
    public var file: String
    public var line: Int?
    public var summary: String
    public var shortSummary: String?
    public var failureScenario: String
    public var category: String?
    public var verdict: String?
    public var outcome: String?

    enum CodingKeys: String, CodingKey {
        case file
        case line
        case summary
        case shortSummary = "short_summary"
        case failureScenario = "failure_scenario"
        case category
        case verdict
        case outcome
    }

    public init(
        file: String,
        line: Int? = nil,
        summary: String,
        shortSummary: String? = nil,
        failureScenario: String,
        category: String? = nil,
        verdict: String? = nil,
        outcome: String? = nil
    ) {
        self.file = file
        self.line = line
        self.summary = summary
        self.shortSummary = shortSummary
        self.failureScenario = failureScenario
        self.category = category
        self.verdict = verdict
        self.outcome = outcome
    }
}

public struct ReportFindingsInput: Codable, Sendable, Equatable {
    public var level: String?
    public var findings: [CodeFindingItem]

    public init(level: String? = "medium", findings: [CodeFindingItem] = []) {
        self.level = level
        self.findings = findings
    }
}

public struct ReportFindingsOutput: Codable, Sendable, Equatable {
    public var count: Int
    public var level: String?
    public var findings: [CodeFindingItem]

    public init(count: Int, level: String? = nil, findings: [CodeFindingItem] = []) {
        self.count = count
        self.level = level
        self.findings = findings
    }
}

// MARK: - Skills and Goals

public struct SkillProposalItem: Codable, Sendable, Equatable {
    public var name: String
    public var kind: String
    public var target: String?
    public var description: String
    public var evidence: [String]?
    public var skillMd: String

    public init(
        name: String,
        kind: String = "new",
        target: String? = nil,
        description: String,
        evidence: [String]? = nil,
        skillMd: String
    ) {
        self.name = name
        self.kind = kind
        self.target = target
        self.description = description
        self.evidence = evidence
        self.skillMd = skillMd
    }
}

public struct ProposeSkillsInput: Codable, Sendable, Equatable {
    public var proposals: [SkillProposalItem]

    public init(proposals: [SkillProposalItem] = []) {
        self.proposals = proposals
    }
}

public struct ProposeSkillsOutput: Codable, Sendable, Equatable {
    public var proposalCount: Int

    public init(proposalCount: Int) {
        self.proposalCount = proposalCount
    }
}

public struct ProposeGoalInput: Codable, Sendable, Equatable {
    public var condition: String
    public var askUser: Bool?

    enum CodingKeys: String, CodingKey {
        case condition
        case askUser = "ask_user"
    }

    public init(condition: String, askUser: Bool? = true) {
        self.condition = condition
        self.askUser = askUser
    }
}

public struct ProposeGoalOutput: Codable, Sendable, Equatable {
    public var condition: String
    public var askUser: Bool

    public init(condition: String, askUser: Bool = true) {
        self.condition = condition
        self.askUser = askUser
    }
}

// MARK: - Feedback Tool

public struct SendFeedbackInput: Codable, Sendable, Equatable {
    public var type: String
    public var title: String
    public var details: String
    public var area: String?
    public var failureMode: String?
    public var taskCategory: String?

    enum CodingKeys: String, CodingKey {
        case type
        case title
        case details
        case area
        case failureMode = "failure_mode"
        case taskCategory = "task_category"
    }

    public init(
        type: String,
        title: String,
        details: String,
        area: String? = nil,
        failureMode: String? = nil,
        taskCategory: String? = nil
    ) {
        self.type = type
        self.title = title
        self.details = details
        self.area = area
        self.failureMode = failureMode
        self.taskCategory = taskCategory
    }
}

public struct SendFeedbackOutput: Codable, Sendable, Equatable {
    public var success: Bool
    public var message: String

    public init(success: Bool, message: String) {
        self.success = success
        self.message = message
    }
}

// MARK: - OpenAI Tool Definitions for Planning and Interactions

public enum PlanningInteractiveToolDefinitions {
    public static let askUserQuestion = OpenAITool.function(
        name: "AskUserQuestion",
        description: "Prompt the user with 1-4 multiple-choice or interactive questions to clarify requirements.",
        parameters: .object(
            properties: [
                "questions": .array(
                    items: .object(
                        properties: [
                            "question": .string(description: "Question ending with a question mark."),
                            "header": .string(description: "Short chip/tag label (max 12 chars)."),
                            "options": .array(
                                items: .object(
                                    properties: [
                                        "label": .string(description: "Concise display text for choice."),
                                        "description": .string(description: "Explanation of implications."),
                                        "preview": .string(description: "Optional code snippet or preview mockup.")
                                    ],
                                    required: ["label", "description"]
                                ),
                                description: "Array of 2-4 distinct choices."
                            ),
                            "multiSelect": .boolean(description: "Whether multiple selections are allowed.")
                        ],
                        required: ["question", "header", "options", "multiSelect"]
                    ),
                    description: "Array of questions."
                )
            ],
            required: ["questions"]
        )
    )

    public static let enterPlanMode = OpenAITool.function(
        name: "EnterPlanMode",
        description: "Switch into dedicated architectural planning mode before making codebase edits.",
        parameters: .emptyObject()
    )

    public static let exitPlanMode = OpenAITool.function(
        name: "ExitPlanMode",
        description: "Exit planning mode and present the finalized plan to the user for approval.",
        parameters: .emptyObject()
    )

    public static let reportFindings = OpenAITool.function(
        name: "ReportFindings",
        description: "Report structured defects, code review issues, or architectural findings.",
        parameters: .object(
            properties: [
                "level": .string(description: "Effort level ('low', 'medium', 'high')."),
                "findings": .array(
                    items: .object(
                        properties: [
                            "file": .string(description: "File path containing the issue."),
                            "line": .integer(description: "Line number of defect."),
                            "summary": .string(description: "One-sentence defect description."),
                            "failure_scenario": .string(description: "Concrete inputs leading to failure.")
                        ],
                        required: ["file", "summary", "failure_scenario"]
                    ),
                    description: "List of findings."
                )
            ],
            required: ["findings"]
        )
    )

    public static let proposeSkills = OpenAITool.function(
        name: "ProposeSkills",
        description: "Propose reusable agent skills derived from conversation patterns.",
        parameters: .object(
            properties: [
                "proposals": .array(
                    items: .object(
                        properties: [
                            "name": .string(description: "Kebab-case skill name."),
                            "kind": .string(description: "'new' or 'improvement'."),
                            "description": .string(description: "Summary of skill capability."),
                            "skillMd": .string(description: "Complete SKILL.md draft content.")
                        ],
                        required: ["name", "kind", "description", "skillMd"]
                    ),
                    description: "List of skill proposals."
                )
            ],
            required: ["proposals"]
        )
    )

    public static let proposeGoal = OpenAITool.function(
        name: "ProposeGoal",
        description: "Propose a verifiable completion condition for the current session goal.",
        parameters: .object(
            properties: [
                "condition": .string(description: "Evaluator-verifiable condition text."),
                "ask_user": .boolean(description: "Whether to request user approval dialog.")
            ],
            required: ["condition"]
        )
    )

    public static let sendFeedback = OpenAITool.function(
        name: "SendFeedback",
        description: "Submit diagnostic feedback, bug reports, or feature ideas.",
        parameters: .object(
            properties: [
                "type": .string(description: "'bug', 'idea', or 'missing_capability'."),
                "title": .string(description: "Short one-line summary."),
                "details": .string(description: "Structured bullet points with repro and evidence.")
            ],
            required: ["type", "title", "details"]
        )
    )

    public static let all: [OpenAITool] = [
        askUserQuestion, enterPlanMode, exitPlanMode, reportFindings,
        proposeSkills, proposeGoal, sendFeedback
    ]
}
