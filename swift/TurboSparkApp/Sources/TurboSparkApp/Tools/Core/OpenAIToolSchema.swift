import Foundation

// MARK: - OpenAI Function Calling Schema Models

/// A tool specification adhering to the OpenAI function calling schema.
public struct OpenAITool: Codable, Sendable, Equatable {
    public var type: String
    public var function: OpenAIFunction

    public init(type: String = "function", function: OpenAIFunction) {
        self.type = type
        self.function = function
    }

    public static func function(
        name: String,
        description: String,
        parameters: JSONSchema
    ) -> OpenAITool {
        OpenAITool(
            type: "function",
            function: OpenAIFunction(
                name: name,
                description: description,
                parameters: parameters
            )
        )
    }
}

/// A function definition inside an OpenAI tool specification.
public struct OpenAIFunction: Codable, Sendable, Equatable {
    public var name: String
    public var description: String
    public var parameters: JSONSchema

    public init(name: String, description: String, parameters: JSONSchema) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// JSON Schema representation for OpenAI function parameters.
public struct JSONSchema: Codable, Sendable, Equatable {
    public var type: String
    public var description: String?
    public var properties: [String: JSONSchemaProperty]?
    public var required: [String]?
    public var enumValues: [String]?
    public var items: JSONSchemaProperty?
    public var additionalProperties: Bool?

    enum CodingKeys: String, CodingKey {
        case type
        case description
        case properties
        case required
        case enumValues = "enum"
        case items
        case additionalProperties
    }

    public init(
        type: String = "object",
        description: String? = nil,
        properties: [String: JSONSchemaProperty]? = nil,
        required: [String]? = nil,
        enumValues: [String]? = nil,
        items: JSONSchemaProperty? = nil,
        additionalProperties: Bool? = nil
    ) {
        self.type = type
        self.description = description
        self.properties = properties
        self.required = required
        self.enumValues = enumValues
        self.items = items
        self.additionalProperties = additionalProperties
    }

    public static func object(
        properties: [String: JSONSchemaProperty],
        required: [String] = [],
        description: String? = nil
    ) -> JSONSchema {
        JSONSchema(
            type: "object",
            description: description,
            properties: properties,
            required: required.isEmpty ? nil : required
        )
    }

    public static func emptyObject(description: String? = nil) -> JSONSchema {
        JSONSchema(
            type: "object",
            description: description,
            properties: [:],
            required: nil
        )
    }
}

/// Individual property specification inside a JSON schema object or array.
public struct JSONSchemaProperty: Codable, Sendable, Equatable {
    public var type: String
    public var description: String?
    public var enumValues: [String]?
    public var items: IndirectWrapper<JSONSchemaProperty>?
    public var properties: [String: JSONSchemaProperty]?
    public var required: [String]?
    public var defaultVal: String?

    enum CodingKeys: String, CodingKey {
        case type
        case description
        case enumValues = "enum"
        case items
        case properties
        case required
        case defaultVal = "default"
    }

    public init(
        type: String,
        description: String? = nil,
        enumValues: [String]? = nil,
        items: JSONSchemaProperty? = nil,
        properties: [String: JSONSchemaProperty]? = nil,
        required: [String]? = nil,
        defaultVal: String? = nil
    ) {
        self.type = type
        self.description = description
        self.enumValues = enumValues
        self.items = items.map { IndirectWrapper($0) }
        self.properties = properties
        self.required = required
        self.defaultVal = defaultVal
    }

    public static func string(
        description: String? = nil,
        enumValues: [String]? = nil,
        defaultVal: String? = nil
    ) -> JSONSchemaProperty {
        JSONSchemaProperty(
            type: "string",
            description: description,
            enumValues: enumValues,
            defaultVal: defaultVal
        )
    }

    public static func integer(
        description: String? = nil,
        defaultVal: String? = nil
    ) -> JSONSchemaProperty {
        JSONSchemaProperty(
            type: "integer",
            description: description,
            defaultVal: defaultVal
        )
    }

    public static func number(
        description: String? = nil,
        defaultVal: String? = nil
    ) -> JSONSchemaProperty {
        JSONSchemaProperty(
            type: "number",
            description: description,
            defaultVal: defaultVal
        )
    }

    public static func boolean(
        description: String? = nil,
        defaultVal: String? = nil
    ) -> JSONSchemaProperty {
        JSONSchemaProperty(
            type: "boolean",
            description: description,
            defaultVal: defaultVal
        )
    }

    public static func array(
        items: JSONSchemaProperty,
        description: String? = nil
    ) -> JSONSchemaProperty {
        JSONSchemaProperty(
            type: "array",
            description: description,
            items: items
        )
    }

    public static func object(
        properties: [String: JSONSchemaProperty],
        required: [String] = [],
        description: String? = nil
    ) -> JSONSchemaProperty {
        JSONSchemaProperty(
            type: "object",
            description: description,
            properties: properties,
            required: required.isEmpty ? nil : required
        )
    }
}

/// Helper wrapper for recursive data structures in Codable structs.
public final class IndirectWrapper<T: Codable & Sendable & Equatable>: Codable, Sendable, Equatable {
    public let value: T

    public init(_ value: T) {
        self.value = value
    }

    public required init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.value = try container.decode(T.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }

    public static func == (lhs: IndirectWrapper<T>, rhs: IndirectWrapper<T>) -> Bool {
        lhs.value == rhs.value
    }
}

// MARK: - Schema Serializer Helper

public enum OpenAIToolSerializer {
    /// Encodes a list of OpenAI tools to a formatted JSON data buffer.
    public static func encodeJSON(_ tools: [OpenAITool]) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(tools)
    }

    /// Encodes a list of OpenAI tools to a JSON UTF-8 string.
    public static func encodeJSONString(_ tools: [OpenAITool]) -> String {
        guard let data = try? encodeJSON(tools),
              let string = String(data: data, encoding: .utf8) else {
            return "[]"
        }
        return string
    }
}
