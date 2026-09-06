import Foundation

// MARK: - REPL Tool

/// Input parameters for evaluating code in a persistent interactive REPL session.
public struct REPLInput: Codable, Sendable, Equatable {
    /// Source code snippet to evaluate.
    public var code: String
    /// Optional summary of what the code performs.
    public var description: String?
    /// Execution timeout in milliseconds.
    public var timeout: Int?

    public init(code: String, description: String? = nil, timeout: Int? = nil) {
        self.code = code
        self.description = description
        self.timeout = timeout
    }
}

/// Output returned from interactive REPL code evaluation.
public struct REPLOutput: Codable, Sendable, Equatable {
    /// Code snippet evaluated.
    public var code: String
    /// Standard output from execution.
    public var stdout: String
    /// Standard error output.
    public var stderr: String
    /// Error message string if execution failed.
    public var error: String?
    /// Whether execution was dispatched asynchronously.
    public var asyncDispatched: Bool?
    /// Any dynamic tools registered during evaluation.
    public var registeredTools: [String]?

    public init(
        code: String,
        stdout: String = "",
        stderr: String = "",
        error: String? = nil,
        asyncDispatched: Bool? = nil,
        registeredTools: [String]? = nil
    ) {
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.asyncDispatched = asyncDispatched
        self.registeredTools = registeredTools
    }
}

// MARK: - OpenAI Tool Definitions for Terminal Operations

/// OpenAI tool definition schemas for terminal operations.
public enum TerminalToolDefinitions {
    public static let bash = OpenAITool.function(
        name: "Bash",
        description: """
            Execute a shell command in the project workspace.

            Runs under zsh with the project root as the working directory. The working \
            directory persists between calls within the project, so "cd build && ninja" \
            is still in effect on the next call; a command that moves the shell outside \
            the project directory resets it back to the root with a note.

            Output combines stdout and stderr in the order they were written, with ANSI \
            escape sequences removed, and is truncated to 30,000 characters (head and \
            tail kept) before being returned.

            The timeout parameter is in milliseconds: default 120000, maximum 600000. A \
            command that exceeds it is terminated and reported as an error with whatever \
            output it produced.

            For long-running commands (dev servers, large builds, watchers) set \
            run_in_background to true: the call returns immediately with a shell ID, \
            the command keeps running with no timeout, and you retrieve its output with \
            the BashOutput tool and stop it with the KillShell tool. Shell IDs are valid \
            only in the conversation that started them.
            """,
        parameters: .object(
            properties: [
                "command": .string(description: "The shell command to execute."),
                "timeout": .integer(description: "Optional execution timeout in milliseconds. Default 120000, maximum 600000."),
                "description": .string(description: "Concise active-voice explanation of what the command does."),
                "run_in_background": .boolean(description: "Set to true to run the command asynchronously in the background. Returns a shell ID immediately; retrieve output later with BashOutput.")
            ],
            required: ["command"]
        )
    )

    public static let bashOutput = OpenAITool.function(
        name: "BashOutput",
        description: "Retrieve the output of a background shell started with Bash and run_in_background: true.",
        parameters: .object(
            properties: [
                "task_id": .string(description: "The background shell ID returned by the Bash tool."),
                "wait_seconds": .integer(description: "Optional. Seconds to wait for completion before returning the current output. Default 30, maximum 120. Use 0 to poll without waiting.")
            ],
            required: ["task_id"]
        )
    )

    public static let killShell = OpenAITool.function(
        name: "KillShell",
        description: "Stop a background shell started with Bash and run_in_background: true.",
        parameters: .object(
            properties: [
                "task_id": .string(description: "The background shell ID returned by the Bash tool.")
            ],
            required: ["task_id"]
        )
    )

    public static let repl = OpenAITool.function(
        name: "REPL",
        description: "Execute an interactive code snippet in a persistent runtime session.",
        parameters: .object(
            properties: [
                "code": .string(description: "The code snippet to evaluate."),
                "description": .string(description: "Short description of the script's purpose."),
                "timeout": .integer(description: "Optional timeout in milliseconds.")
            ],
            required: ["code"]
        )
    )

    public static let all: [OpenAITool] = [
        bash, bashOutput, killShell, repl
    ]
}
