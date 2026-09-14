import Foundation

/// Schema definitions for code and API documentation search tool.
public enum CodeSearchToolDefinitions {
    public static let codesearch = OpenAITool.function(
        name: "codesearch",
        description: """
            Search for technical code documentation, API references, library examples, and SDK signatures.

            Specialized for programming languages, frameworks, and developer documentation. \
            Returns LLM-optimized code snippets, type definitions, and reference documentation.

            Parameters include the search query, optional token limit, and optional target framework or language.
            """,
        parameters: .object(
            properties: [
                "query": .string(description: "Technical search query (e.g. 'Swift CheckedContinuation usage', 'React 19 useActionState')."),
                "tokens_num": .integer(description: "Optional maximum context tokens to return (default: 5000, max: 20000)."),
                "framework": .string(description: "Optional framework or language hint (e.g. 'swift', 'rust', 'react', 'python')."),
                "provider": .string(description: "Optional search provider ('auto', 'tavily', 'exa', 'brave'). Default is 'auto'.")
            ],
            required: ["query"]
        )
    )

    public static let all: [OpenAITool] = [codesearch]
}
