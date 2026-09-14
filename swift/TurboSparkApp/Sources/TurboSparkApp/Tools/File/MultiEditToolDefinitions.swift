import Foundation

/// Schema definitions for atomic multi-file editing tool.
public enum MultiEditToolDefinitions {
    public static let multiedit = OpenAITool.function(
        name: "multiedit",
        description: """
            Perform coordinated edits across multiple files in a single atomic transaction.

            Validates all target files, paths, and replacement patterns up front. If any \
            single edit fails or any file has been modified externally since last read, the \
            entire transaction aborts and all modified files are automatically rolled back to \
            their pre-edit state.

            Accepts an array of edit specifications. Each edit contains file_path, old_string, \
            new_string, and optional replace_all.
            """,
        parameters: .object(
            properties: [
                "edits": .array(
                    items: .object(
                        properties: [
                            "file_path": .string(description: "Path to the file to edit (relative to project root)."),
                            "old_string": .string(description: "Exact text segment to find and replace."),
                            "new_string": .string(description: "Replacement text segment."),
                            "replace_all": .boolean(description: "Optional. True to replace all occurrences; false for first match.")
                        ],
                        required: ["file_path", "old_string", "new_string"],
                        description: "An individual file edit specification."
                    ),
                    description: "List of file edits to apply atomically."
                ),
                "edits_json": .string(
                    description: "Optional JSON string encoding the edits array if the model formats it as text."
                )
            ]
        )
    )

    public static let all: [OpenAITool] = [multiedit]
}
