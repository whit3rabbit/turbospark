import Foundation

/// The four agents this app ships, and the prompts that define them.
///
/// Data rather than logic, and moved out for that reason: it is a third of
/// `AgentManager` by line count and none of it participates in discovery,
/// precedence or the disabled list. The one behavioural fact worth keeping
/// next to it is that a project agent taking one of these names is
/// CONSTRAINED to it rather than replacing it -- its tool ceiling
/// (state#22) and, since state#95, its turn budget.
extension AgentManager {
    // MARK: - Built-in Agents

    public static let explorePrompt = """
    You are a fast, read-only file search and codebase exploration specialist.
    Your role is EXCLUSIVELY to search, read, and analyze code.
    You are strictly prohibited from creating or modifying files.
    Use glob, read_file, search_code, and read-only shell commands to explore the codebase.
    Report your findings clearly, concisely, and directly.
    """

    public static let planPrompt = """
    You are a software architecture and implementation planning specialist.
    Analyze requirements, inspect existing codebase design and patterns, and construct detailed,
    step-by-step implementation plans, identifying potential edge cases, verification steps, and trade-offs.
    """

    public static let generalPurposePrompt = """
    You are a versatile, autonomous coding assistant capable of codebase exploration,
    file editing, command execution, and problem solving.
    """

    public static let reviewerPrompt = """
    You are a code review and security verification specialist.
    Analyze code changes, identify bugs, edge cases, performance bottlenecks, and security hazards.
    Provide constructive feedback and specific recommendations.
    """

    public var builtInAgents: [AppAgentDefinition] {
        [
            AppAgentDefinition(
                name: "explore",
                displayName: "Codebase Explorer",
                agentDescription: "Fast read-only agent specialized for exploring codebases and answering questions without modifying files.",
                systemPrompt: Self.explorePrompt,
                disallowedTools: [
                    "write_file", "save_file", "filewrite", "write",
                    "edit_file", "fileedit", "edit",
                    "apply_patch", "applypatch",
                    "agent", "subagent"
                ],
                maxTurns: 5,
                sourceAgent: .turboSpark,
                scope: .builtIn,
                isEnabled: !isAgentDisabled(name: "explore")
            ),
            AppAgentDefinition(
                name: "plan",
                displayName: "Architect & Planner",
                agentDescription: "Specialized planning agent for designing architectures, researching requirements, and structuring implementation steps.",
                systemPrompt: Self.planPrompt,
                disallowedTools: [
                    "write_file", "save_file", "filewrite", "write",
                    "edit_file", "fileedit", "edit",
                    "apply_patch", "applypatch"
                ],
                maxTurns: 5,
                sourceAgent: .turboSpark,
                scope: .builtIn,
                isEnabled: !isAgentDisabled(name: "plan")
            ),
            AppAgentDefinition(
                name: "general-purpose",
                displayName: "General Purpose",
                agentDescription: "Full autonomous assistant capable of reading, searching, editing files, and running commands.",
                systemPrompt: Self.generalPurposePrompt,
                maxTurns: 6,
                sourceAgent: .turboSpark,
                scope: .builtIn,
                isEnabled: !isAgentDisabled(name: "general-purpose")
            ),
            AppAgentDefinition(
                name: "reviewer",
                displayName: "Code Reviewer",
                agentDescription: "Code review and security audit specialist for verifying correctness and code quality.",
                systemPrompt: Self.reviewerPrompt,
                disallowedTools: [
                    "write_file", "save_file", "filewrite", "write",
                    "edit_file", "fileedit", "edit",
                    "apply_patch", "applypatch"
                ],
                maxTurns: 4,
                sourceAgent: .turboSpark,
                scope: .builtIn,
                isEnabled: !isAgentDisabled(name: "reviewer")
            )
        ]
    }
}
