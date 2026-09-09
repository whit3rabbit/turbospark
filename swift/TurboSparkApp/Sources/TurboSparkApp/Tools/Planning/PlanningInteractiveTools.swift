import Foundation

// MARK: - AskUserQuestion Tool

/// A selectable option presented to the user in an interactive question.
public struct UserQuestionOption: Codable, Sendable, Equatable {
    public var label: String
    public var description: String
    public var preview: String?

    /// Initializes a user question option with label, description, and optional preview.
    public init(label: String, description: String, preview: String? = nil) {
        self.label = label
        self.description = description
        self.preview = preview
    }
}

/// An individual question item in an interactive question prompt.
public struct UserQuestionItem: Codable, Sendable, Equatable {
    public var question: String
    public var header: String
    public var options: [UserQuestionOption]
    public var multiSelect: Bool

    /// Initializes a user question item with question text, short header, options, and selection mode.
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

    /// `multiSelect` decodes as optional, defaulting false: the upstream
    /// schema marks it optional, and a model that omits the key would
    /// otherwise fail the WHOLE questions payload rather than lose one bit
    /// of selection mode.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        question = try container.decode(String.self, forKey: .question)
        header = try container.decode(String.self, forKey: .header)
        options = try container.decode([UserQuestionOption].self, forKey: .options)
        multiSelect = try container.decodeIfPresent(Bool.self, forKey: .multiSelect) ?? false
    }
}

/// Input payload for AskUserQuestion containing interactive questions.
public struct AskUserQuestionInput: Codable, Sendable, Equatable {
    public var questions: [UserQuestionItem]

    /// Initializes the input payload with questions.
    public init(questions: [UserQuestionItem]) {
        self.questions = questions
    }
}

/// Output payload returned by AskUserQuestion containing answered choices.
public struct AskUserQuestionOutput: Codable, Sendable, Equatable {
    public var questions: [UserQuestionItem]
    public var answers: [String: String]
    public var response: String?

    /// Initializes the output payload with questions, answers, and optional response text.
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

/// Input payload for entering planning mode.
public struct EnterPlanModeInput: Codable, Sendable, Equatable {
    /// Initializes an empty enter plan mode input payload.
    public init() {}
}

/// Output payload returned upon entering planning mode.
public struct EnterPlanModeOutput: Codable, Sendable, Equatable {
    public var message: String

    /// Initializes the enter plan mode output payload with a status message.
    public init(message: String = "Entered plan mode.") {
        self.message = message
    }
}

/// Input payload for exiting planning mode.
public struct ExitPlanModeInput: Codable, Sendable, Equatable {
    /// Initializes an empty exit plan mode input payload.
    public init() {}
}

/// Output payload returned upon exiting planning mode with the finalized plan.
public struct ExitPlanModeOutput: Codable, Sendable, Equatable {
    public var plan: String?
    public var isAgent: Bool
    public var filePath: String?

    /// Initializes the exit plan mode output with plan text, agent flag, and optional plan file path.
    public init(plan: String? = nil, isAgent: Bool = false, filePath: String? = nil) {
        self.plan = plan
        self.isAgent = isAgent
        self.filePath = filePath
    }
}

// MARK: - ReportFindings Tool

/// Structured code defect or review finding.
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

    /// Initializes a code finding item with location, summary, and failure scenario details.
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

/// Input payload for reporting code findings.
public struct ReportFindingsInput: Codable, Sendable, Equatable {
    public var level: String?
    public var findings: [CodeFindingItem]

    /// Initializes the report findings input payload with an optional effort level and findings.
    public init(level: String? = "medium", findings: [CodeFindingItem] = []) {
        self.level = level
        self.findings = findings
    }
}

/// Output summary returned after reporting findings.
public struct ReportFindingsOutput: Codable, Sendable, Equatable {
    public var count: Int
    public var level: String?
    public var findings: [CodeFindingItem]

    /// Initializes the report findings output summary with count, effort level, and findings.
    public init(count: Int, level: String? = nil, findings: [CodeFindingItem] = []) {
        self.count = count
        self.level = level
        self.findings = findings
    }
}

// MARK: - Skills and Goals

/// Proposed reusable agent skill specification.
public struct SkillProposalItem: Codable, Sendable, Equatable {
    public var name: String
    public var kind: String
    public var target: String?
    public var description: String
    public var evidence: [String]?
    public var skillMd: String

    /// Initializes a skill proposal item with metadata and SKILL.md draft content.
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

/// Input payload for proposing reusable agent skills.
public struct ProposeSkillsInput: Codable, Sendable, Equatable {
    public var proposals: [SkillProposalItem]

    /// Initializes the propose skills input payload with candidate skill proposals.
    public init(proposals: [SkillProposalItem] = []) {
        self.proposals = proposals
    }
}

/// Output summary returned after proposing skills.
public struct ProposeSkillsOutput: Codable, Sendable, Equatable {
    public var proposalCount: Int

    /// Initializes the propose skills output payload with the count of accepted proposals.
    public init(proposalCount: Int) {
        self.proposalCount = proposalCount
    }
}

/// Input payload for proposing a verifiable session goal.
public struct ProposeGoalInput: Codable, Sendable, Equatable {
    public var condition: String
    public var askUser: Bool?

    enum CodingKeys: String, CodingKey {
        case condition
        case askUser = "ask_user"
    }

    /// Initializes the propose goal input with condition text and confirmation flag.
    public init(condition: String, askUser: Bool? = true) {
        self.condition = condition
        self.askUser = askUser
    }
}

/// Output returned after recording a proposed goal.
public struct ProposeGoalOutput: Codable, Sendable, Equatable {
    public var condition: String
    public var askUser: Bool

    /// Initializes the propose goal output confirmation.
    public init(condition: String, askUser: Bool = true) {
        self.condition = condition
        self.askUser = askUser
    }
}

// MARK: - Feedback Tool

/// Input payload for submitting diagnostic feedback or bug reports.
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

    /// Initializes feedback payload with category, title, details, and optional failure mode.
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

/// Output returned after submitting diagnostic feedback.
public struct SendFeedbackOutput: Codable, Sendable, Equatable {
    public var success: Bool
    public var message: String

    /// Initializes the feedback output response.
    public init(success: Bool, message: String) {
        self.success = success
        self.message = message
    }
}

// MARK: - OpenAI Tool Definitions for Planning and Interactions

/// OpenAI tool schema definitions for interactive user questions and planning tools.
public enum PlanningInteractiveToolDefinitions {
    /// Tool schema definition for AskUserQuestion.
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

    /// Tool schema definition for EnterPlanMode.
    public static let enterPlanMode = OpenAITool.function(
        name: "EnterPlanMode",
        description: "Switch into dedicated architectural planning mode before making codebase edits.",
        parameters: .emptyObject()
    )

    /// Tool schema definition for ExitPlanMode.
    public static let exitPlanMode = OpenAITool.function(
        name: "ExitPlanMode",
        description: "Exit planning mode and present the finalized plan to the user for approval.",
        parameters: .emptyObject()
    )

    /// Tool schema definition for ReportFindings.
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

    /// Tool schema definition for ProposeSkills.
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

    /// Tool schema definition for ProposeGoal.
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

    /// Tool schema definition for SendFeedback.
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

    /// Complete list of planning and interactive tool definitions.
    public static let all: [OpenAITool] = [
        askUserQuestion, enterPlanMode, exitPlanMode, reportFindings,
        proposeSkills, proposeGoal, sendFeedback
    ]
}
