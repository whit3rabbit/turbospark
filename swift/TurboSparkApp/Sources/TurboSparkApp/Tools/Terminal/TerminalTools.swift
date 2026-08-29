import Foundation

// MARK: - Bash / Terminal Command Tool

/// Input parameters for executing a command in the local shell.
public struct BashInput: Codable, Sendable, Equatable {
    /// Shell command string to execute.
    public var command: String
    /// Optional execution timeout in milliseconds.
    public var timeout: Int?
    /// Concise explanation of why the command is being run.
    public var description: String?
    /// Whether the process should be launched asynchronously as a background task.
    public var runInBackground: Bool?
    /// Whether sandboxing restrictions should be bypassed.
    public var dangerouslyDisableSandbox: Bool?

    enum CodingKeys: String, CodingKey {
        case command
        case timeout
        case description
        case runInBackground = "run_in_background"
        case dangerouslyDisableSandbox
    }

    public init(
        command: String,
        timeout: Int? = nil,
        description: String? = nil,
        runInBackground: Bool? = nil,
        dangerouslyDisableSandbox: Bool? = nil
    ) {
        self.command = command
        self.timeout = timeout
        self.description = description
        self.runInBackground = runInBackground
        self.dangerouslyDisableSandbox = dangerouslyDisableSandbox
    }
}

/// Output returned upon completion or interruption of a shell command.
public struct BashOutput: Codable, Sendable, Equatable {
    /// Standard output text.
    public var stdout: String
    /// Standard error text.
    public var stderr: String
    /// File path where full raw output was persisted if large.
    public var rawOutputPath: String?
    /// Whether the command execution was cancelled or interrupted.
    public var interrupted: Bool
    /// Whether stdout contains binary image data.
    public var isImage: Bool?
    /// Task identifier if dispatched as a background task.
    public var backgroundTaskId: String?
    /// Timeout duration reached if timed out.
    public var timedOutAfterMs: Int?
    /// Human-readable explanation of non-zero exit code if applicable.
    public var returnCodeInterpretation: String?
    /// Process numeric exit status code.
    public var exitCode: Int?

    public init(
        stdout: String,
        stderr: String = "",
        rawOutputPath: String? = nil,
        interrupted: Bool = false,
        isImage: Bool? = nil,
        backgroundTaskId: String? = nil,
        timedOutAfterMs: Int? = nil,
        returnCodeInterpretation: String? = nil,
        exitCode: Int? = nil
    ) {
        self.stdout = stdout
        self.stderr = stderr
        self.rawOutputPath = rawOutputPath
        self.interrupted = interrupted
        self.isImage = isImage
        self.backgroundTaskId = backgroundTaskId
        self.timedOutAfterMs = timedOutAfterMs
        self.returnCodeInterpretation = returnCodeInterpretation
        self.exitCode = exitCode
    }
}

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

/// OpenAI tool definition schemas for Bash and REPL execution.
public enum TerminalToolDefinitions {
    public static let bash = OpenAITool.function(
        name: "Bash",
        description: "Execute a shell command inside the project environment or workspace terminal.",
        parameters: .object(
            properties: [
                "command": .string(description: "The shell command to execute."),
                "timeout": .integer(description: "Optional execution timeout in milliseconds."),
                "description": .string(description: "Concise active-voice explanation of what the command does."),
                "run_in_background": .boolean(description: "Set to true to run the command asynchronously in the background.")
            ],
            required: ["command"]
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
        bash, repl
    ]
}
