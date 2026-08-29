import Foundation

// MARK: - Projects Tool

public struct ProjectsInput: Codable, Sendable, Equatable {
    public var method: String
    public var path: String?
    public var content: String?
    public var localPath: String?
    public var presentToUser: Bool?
    public var query: String?
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

public struct ProjectDocInfo: Codable, Sendable, Equatable {
    public var path: String
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

public struct ProjectSearchHit: Codable, Sendable, Equatable {
    public var name: String?
    public var docUuid: String?
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

public struct ProjectsOutput: Codable, Sendable, Equatable {
    public var method: String
    public var notice: String?
    public var name: String?
    public var description: String?
    public var instructions: String?
    public var docs: [ProjectDocInfo]?
    public var path: String?
    public var content: String?
    public var localPath: String?
    public var rag: Bool?
    public var hits: [ProjectSearchHit]?
    public var docUuid: String?
    public var replaced: Bool?
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
