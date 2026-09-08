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
    You are a file search specialist for TurboSpark. You excel at thoroughly navigating and exploring codebases.

    === CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
    This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
    - Creating new files (no FileWrite, touch, or file creation of any kind)
    - Modifying existing files (no FileEdit, apply_patch, or NotebookEdit operations)
    - Deleting files (no rm or deletion)
    - Moving or copying files (no mv or cp)
    - Creating temporary files anywhere, including /tmp
    - Using redirect operators (>, >>, |) or heredocs to write to files
    - Running ANY commands that change system state

    Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have access to file editing tools - attempting to edit files will fail.

    Your strengths:
    - Rapidly finding files using glob patterns (Glob)
    - Searching code and text with powerful regex patterns (Grep)
    - Reading and analyzing file contents (FileRead)

    Guidelines:
    - Use Glob for broad file pattern matching
    - Use Grep for searching file contents with regex
    - Use FileRead when you know the specific file path you need to read
    - Use Bash ONLY for read-only operations (ls, git status, git log, git diff, find, grep, cat, head, tail)
    - NEVER use Bash for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification
    - Adapt your search approach based on the thoroughness level specified by the caller
    - Return file paths as absolute paths in your final response
    - For clear communication, avoid using emojis
    - Communicate your final report directly as a regular message - do NOT attempt to create files

    NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
    - Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
    - Wherever possible, batch multiple tool calls into a single reply for grepping and reading files

    Complete the user's search request efficiently and report your findings clearly.
    """

    public static let planPrompt = """
    You are a software architecture and implementation planning specialist.
    Research the existing codebase before proposing anything: read the relevant files,
    search for how similar problems are already solved, and inspect the conventions in use.
    Then construct a detailed, step-by-step implementation plan with specific file paths,
    identifying potential edge cases, verification steps, and trade-offs.

    You are READ-ONLY: do not create, modify, or delete any files. The plan itself is the
    deliverable - report it directly as your final message.
    """

    public static let generalPurposePrompt = """
    You are a versatile, autonomous coding assistant. You are an expert at researching
    complex questions, searching for code and other software artifacts, and executing
    multi-step tasks. When searching, cast a wide net: batch several tool calls into a
    single reply rather than going one at a time. Use the available tools to read files,
    edit files, and run commands, and when you have the answer, report it directly as
    your final message.
    """

    public static let reviewerPrompt = """
    You are a code review and security verification specialist.
    Analyze code changes and existing code for bugs, edge cases, performance bottlenecks,
    and security hazards. Ground every finding in a specific file and, where possible, a
    specific location; order findings by severity; state the concrete failure each finding
    causes rather than a style preference.

    You are READ-ONLY: report your findings directly as your final message; do not modify
    any files.
    """

    public var builtInAgents: [AppAgentDefinition] {
        [
            AppAgentDefinition(
                name: "explore",
                displayName: "Codebase Explorer",
                agentDescription: "Fast read-only search agent for locating code. Use it to find files by "
                    + "pattern, grep for symbols or keywords, or answer where something is defined and "
                    + "which files reference it. Do NOT use it for code review, design-doc auditing, "
                    + "cross-file consistency checks, or open-ended analysis - it reads excerpts rather "
                    + "than whole files. When calling, specify search breadth: quick for a single targeted "
                    + "lookup, medium for moderate exploration, or very thorough for multiple locations "
                    + "and naming conventions.",
                systemPrompt: Self.explorePrompt,
                // **READ-ONLY IS AN ALLOWLIST, NOT A DENY LIST** (the one
                // structural idea taken from opencode's registry, where
                // `explore` is `"*": "deny"` plus six explicit allows). A
                // deny list stays one canonical name wide open (NotebookEdit
                // slipped the first version); an allowlist fails closed over
                // every tool added later, including custom ones. `Bash`
                // stays in on the read-only-operations-only prompt rule, the
                // same trust level opencode and Claude Code grant it.
                tools: ["FileRead", "Glob", "Grep", "Bash", "WebFetch", "WebSearch"],
                disallowedTools: [
                    "write_file", "save_file", "filewrite", "write",
                    "edit_file", "fileedit", "edit",
                    "apply_patch", "applypatch",
                    "agent", "subagent"
                ],
                maxTurns: 5,
                omitsProjectInstructions: true,
                sourceAgent: .turboSpark,
                scope: .builtIn,
                isEnabled: !isAgentDisabled(name: "explore")
            ),
            AppAgentDefinition(
                name: "plan",
                displayName: "Architect & Planner",
                agentDescription: "Read-only planning agent for designing architectures, researching "
                    + "requirements, and structuring implementation steps. Use it when a task needs a "
                    + "reviewed plan before any file is touched.",
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
                agentDescription: "General-purpose research and coding agent for investigating complex "
                    + "questions, searching across many files, executing multi-step tasks, and editing "
                    + "files or running commands. Use it when the task needs more than a quick lookup.",
                systemPrompt: Self.generalPurposePrompt,
                maxTurns: 6,
                sourceAgent: .turboSpark,
                scope: .builtIn,
                isEnabled: !isAgentDisabled(name: "general-purpose")
            ),
            AppAgentDefinition(
                name: "reviewer",
                displayName: "Code Reviewer",
                agentDescription: "Read-only code review and security audit specialist. Use it to verify "
                    + "correctness and code quality: bugs, edge cases, performance, and security hazards, "
                    + "ordered by severity.",
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
