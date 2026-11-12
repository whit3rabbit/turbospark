import Foundation

/// Schema definitions for the batch execution tool.
public enum BatchToolDefinitions {
    public static let batch = OpenAITool.function(
        name: "batch",
        description: """
            Execute multiple independent tool calls in one request (up to 25 calls).

            Accepts an array of tool calls, each specifying a tool name and its arguments. \
            Read-only calls run concurrently; file-write and terminal calls run one at a time \
            so edits to one path cannot lose a write. Partial failure is handled gracefully: if one \
            tool call fails, the remaining calls continue and complete. Recursive batching \
            (calling batch within batch) is rejected.

            Returns a structured execution report detailing total, successful, and failed \
            operations alongside individual tool outputs.
            """,
        parameters: .object(
            properties: [
                "tool_calls": .array(
                    items: .object(
                        properties: [
                            "tool": .string(description: "Name of the tool to execute."),
                            "parameters": .object(
                                properties: [:],
                                description: "Dictionary of arguments to pass to the tool."
                            )
                        ],
                        required: ["tool", "parameters"],
                        description: "An individual tool invocation specification."
                    ),
                    description: "List of tool calls to execute (maximum 25). Read-only calls run in parallel; file-write and terminal calls are serialized."
                ),
                "tool_calls_json": .string(
                    description: "Optional JSON string encoding the tool_calls array if the model formats it as text."
                )
            ]
        )
    )

    public static let all: [OpenAITool] = [batch]
}
