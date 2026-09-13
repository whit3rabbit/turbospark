import Foundation

/// Bounded, byte-offset recall for archived tool observations.
enum ToolObservationToolDefinitions {
    static let recall = OpenAITool.function(
        name: "recall_tool_output",
        description: "Read an exact bounded byte range from archived tool output. Use the observation id and byte count in an archived-output placeholder.",
        parameters: .object(
            properties: [
                "observation_id": .string(description: "The archived observation UUID."),
                "offset_bytes": .integer(description: "Zero-based byte offset to start reading."),
                "max_bytes": .integer(description: "Maximum number of bytes to return, capped by the archive."),
            ],
            required: ["observation_id", "offset_bytes", "max_bytes"]
        )
    )

    static let all: [OpenAITool] = [recall]
}
