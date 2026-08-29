import Foundation

// MARK: - Projects Tool

/// Input payload for querying and managing project documents, instructions, and search indexes.
public struct ProjectsInput: Codable, Sendable, Equatable {
    /// Operation method ("project_info", "project_read", "project_search", "project_write", "project_delete").
    public var method: String
    /// Relative path of the document inside the project's knowledge store.
    public var path: String?
    /// Inline document text content for write operations.
    public var content: String?
    /// Local filesystem path for importing/syncing external files.
    public var localPath: String?
    /// Whether the document should be highlighted as a primary user deliverable.
    public var presentToUser: Bool?
    /// Search query string for semantic/keyword retrieval over project documents.
    public var query: String?
    /// Maximum count of search hits to return.
    public var n: Int?

    enum CodingKeys: String, CodingKey {
        case method
        case path
        case content
        case localPath = "local_path"
        case presentToUser = "present_to_user"
        case query
        case n
    }

    public init(
        method: String,
        path: String? = nil,
        content: String? = nil,
        localPath: String? = nil,
        presentToUser: Bool? = nil,
        query: String? = nil,
        n: Int? = nil
    ) {
        self.method = method
        self.path = path
        self.content = content
        self.localPath = localPath
        self.presentToUser = presentToUser
        self.query = query
        self.n = n
    }
}

/// Metadata description of a document stored in project knowledge.
public struct ProjectDocInfo: Codable, Sendable, Equatable {
    /// Relative document path.
    public var path: String
    /// ISO timestamp when the document was added.
    public var createdAt: String?

    enum CodingKeys: String, CodingKey {
        case path
        case createdAt = "created_at"
    }

    public init(path: String, createdAt: String? = nil) {
        self.path = path
        self.createdAt = createdAt
    }
}

/// Search hit returned from project knowledge retrieval.
public struct ProjectSearchHit: Codable, Sendable, Equatable {
    /// Document file name or title.
    public var name: String?
    /// Unique document ID.
    public var docUuid: String?
    /// Matching text excerpt from the document.
    public var text: String?

    enum CodingKeys: String, CodingKey {
        case name
        case docUuid = "doc_uuid"
        case text
    }

    public init(name: String? = nil, docUuid: String? = nil, text: String? = nil) {
        self.name = name
        self.docUuid = docUuid
        self.text = text
    }
}

/// Output payload from a project document or knowledge base operation.
public struct ProjectsOutput: Codable, Sendable, Equatable {
    /// Executed method name.
    public var method: String
    /// Informational or warning notice text.
    public var notice: String?
    /// Project display name.
    public var name: String?
    /// Project summary description.
    public var description: String?
    /// Project custom instructions and guidelines.
    public var instructions: String?
    /// Discovered document listing.
    public var docs: [ProjectDocInfo]?
    /// Document relative path.
    public var path: String?
    /// Document textual content.
    public var content: String?
    /// Local filesystem file path.
    public var localPath: String?
    /// Whether RAG indexing is active for this document.
    public var rag: Bool?
    /// Search result hits if method was project_search.
    public var hits: [ProjectSearchHit]?
    /// Document unique ID.
    public var docUuid: String?
    /// Whether an existing document was overwritten.
    public var replaced: Bool?
    /// Whether the document was deleted.
    public var deleted: Bool?

    enum CodingKeys: String, CodingKey {
        case method
        case notice
        case name
        case description
        case instructions
        case docs
        case path
        case content
        case localPath = "local_path"
        case rag
        case hits
        case docUuid = "doc_uuid"
        case replaced
        case deleted
    }

    public init(
        method: String,
        notice: String? = nil,
        name: String? = nil,
        description: String? = nil,
        instructions: String? = nil,
        docs: [ProjectDocInfo]? = nil,
        path: String? = nil,
        content: String? = nil,
        localPath: String? = nil,
        rag: Bool? = nil,
        hits: [ProjectSearchHit]? = nil,
        docUuid: String? = nil,
        replaced: Bool? = nil,
        deleted: Bool? = nil
    ) {
        self.method = method
        self.notice = notice
        self.name = name
        self.description = description
        self.instructions = instructions
        self.docs = docs
        self.path = path
        self.content = content
        self.localPath = localPath
        self.rag = rag
        self.hits = hits
        self.docUuid = docUuid
        self.replaced = replaced
        self.deleted = deleted
    }
}

// MARK: - OpenAI Tool Definitions for Project Documents

/// OpenAI tool definition schemas for project document knowledge operations.
public enum ProjectDocDefinitions {
    public static let projects = OpenAITool.function(
        name: "Projects",
        description: "Manage project knowledge base, documents, instructions, and search index.",
        parameters: .object(
            properties: [
                "method": .string(
                    description: "Operation: 'project_info', 'project_read', 'project_search', 'project_write', 'project_delete'."
                ),
                "path": .string(description: "Document relative path in project knowledge."),
                "content": .string(description: "Inline document content for project_write."),
                "local_path": .string(description: "Local file path to upload into project."),
                "present_to_user": .boolean(description: "Mark as primary deliverable document."),
                "query": .string(description: "Search query for project knowledge."),
                "n": .integer(description: "Number of search results to return.")
            ],
            required: ["method"]
        )
    )

    public static let all: [OpenAITool] = [projects]
}
