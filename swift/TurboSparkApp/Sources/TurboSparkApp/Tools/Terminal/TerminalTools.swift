import Foundation

// MARK: - Bash / Terminal Command Tool

public struct BashInput: Codable, Sendable, Equatable {
    public var command: String
    public var timeout: Int?
    public var description: String?
    public var runInBackground: Bool?
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

public struct BashOutput: Codable, Sendable, Equatable {
    public var stdout: String
    public var stderr: String
    public var rawOutputPath: String?
    public var interrupted: Bool
    public var isImage: Bool?
    public var backgroundTaskId: String?
    public var timedOutAfterMs: Int?
    public var returnCodeInterpretation: String?
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

public struct REPLInput: Codable, Sendable, Equatable {
    public var code: String
    public var description: String?
    public var timeout: Int?

    public init(code: String, description: String? = nil, timeout: Int? = nil) {
        self.code = code
        self.description = description
        self.timeout = timeout
    }
}

public struct REPLOutput: Codable, Sendable, Equatable {
    public var code: String
    public var stdout: String
    public var stderr: String
    public var error: String?
    public var asyncDispatched: Bool?
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
