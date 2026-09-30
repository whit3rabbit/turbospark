import Foundation

/// JSON values retained in their original structural and scalar types until
/// a completed model tool call has passed schema validation.
public indirect enum ToolCallJSONValue: Codable, Equatable, Sendable {
    case object([String: ToolCallJSONValue])
    case array([ToolCallJSONValue])
    case string(String)
    case number(Decimal)
    case boolean(Bool)
    case null

    public init(from decoder: Decoder) throws {
        let single = try decoder.singleValueContainer()
        if single.decodeNil() {
            self = .null
        } else if let value = try? single.decode(String.self) {
            self = .string(value)
        } else if let value = try? single.decode(Bool.self) {
            self = .boolean(value)
        } else if let value = try? single.decode(Decimal.self) {
            self = .number(value)
        } else if let values = try? decoder.container(keyedBy: ToolCallCodingKey.self) {
            var object: [String: ToolCallJSONValue] = [:]
            for key in values.allKeys {
                object[key.stringValue] = try values.decode(ToolCallJSONValue.self, forKey: key)
            }
            self = .object(object)
        } else if var values = try? decoder.unkeyedContainer() {
            var array: [ToolCallJSONValue] = []
            while !values.isAtEnd {
                array.append(try values.decode(ToolCallJSONValue.self))
            }
            self = .array(array)
        } else {
            throw DecodingError.dataCorruptedError(
                in: single, debugDescription: "Unsupported JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .object(let values):
            var container = encoder.container(keyedBy: ToolCallCodingKey.self)
            for (key, value) in values {
                guard let codingKey = ToolCallCodingKey(stringValue: key) else {
                    throw EncodingError.invalidValue(
                        key,
                        EncodingError.Context(
                            codingPath: encoder.codingPath,
                            debugDescription: "Invalid JSON object key"))
                }
                try container.encode(value, forKey: codingKey)
            }
        case .array(let values):
            var container = encoder.unkeyedContainer()
            for value in values { try container.encode(value) }
        case .string(let value):
            var container = encoder.singleValueContainer()
            try container.encode(value)
        case .number(let value):
            var container = encoder.singleValueContainer()
            try container.encode(value)
        case .boolean(let value):
            var container = encoder.singleValueContainer()
            try container.encode(value)
        case .null:
            var container = encoder.singleValueContainer()
            try container.encodeNil()
        }
    }

    fileprivate var isInteger: Bool {
        guard case .number(let value) = self else { return false }
        var input = value
        var rounded = Decimal()
        NSDecimalRound(&rounded, &input, 0, .down)
        return rounded == value
    }
}

private struct ToolCallCodingKey: CodingKey, Hashable {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}

public enum ToolCallValidationRefusal: Equatable, Sendable {
    case unavailableTool(name: String)
    case malformedJSON
    case nonObjectArguments
    case unsupportedSchema(toolName: String, path: String, reason: String)
    case invalidArguments(path: String, reason: String)
}

public enum ToolCallValidationResult: Equatable, Sendable {
    case validated(tool: OpenAITool, arguments: [String: ToolCallJSONValue])
    case refused(ToolCallValidationRefusal)
}

/// Immutable definitions and deferred MCP descriptors captured for one
/// generation request after the app applies its project and runtime filters.
public struct TurnAvailableTools: Equatable, Sendable {
    /// Exact structured definitions in the captured available set, including
    /// discovered MCP definitions used by direct MCP-call surfaces.
    public let definitions: [OpenAITool]

    /// Definitions rendered in the ordinary `Available Tools` prompt block.
    /// Dynamic MCP tools remain in the deferred listing to preserve the current
    /// progressive-disclosure prompt contract.
    public let promptDefinitions: [OpenAITool]

    /// Deferred MCP names, descriptions, and raw schemas from the same catalog
    /// snapshot used to build the prompt listing.
    public let deferredMcpTools: [DeferredToolDescriptor]

    public init(
        definitions: [OpenAITool],
        promptDefinitions: [OpenAITool]? = nil,
        deferredMcpTools: [DeferredToolDescriptor] = []
    ) {
        self.definitions = definitions
        self.promptDefinitions = promptDefinitions ?? definitions
        self.deferredMcpTools = deferredMcpTools
    }

    public func validate(toolName: String, argumentsJSON: String) -> ToolCallValidationResult {
        guard let tool = definitions.first(where: { $0.function.name == toolName }) else {
            return .refused(.unavailableTool(name: toolName))
        }
        guard let data = argumentsJSON.data(using: .utf8),
              let value = try? JSONDecoder().decode(ToolCallJSONValue.self, from: data) else {
            return .refused(.malformedJSON)
        }
        guard case .object(let arguments) = value else {
            return .refused(.nonObjectArguments)
        }
        if let issue = unsupportedSchemaIssue(tool.function.parameters) {
            return .refused(.unsupportedSchema(
                toolName: toolName, path: issue.path, reason: issue.reason))
        }
        if let refusal = validateObject(
            arguments,
            properties: tool.function.parameters.properties ?? [:],
            required: tool.function.parameters.required ?? [],
            additionalProperties: tool.function.parameters.additionalProperties,
            toolName: toolName,
            path: "$") {
            return .refused(refusal)
        }
        return .validated(tool: tool, arguments: arguments)
    }

    private func unsupportedSchemaIssue(
        _ schema: JSONSchema
    ) -> (path: String, reason: String)? {
        if let issue = schema.validationIssue { return ("$", issue) }
        guard schema.type == "object" else {
            return ("$", "parameters must use an object schema")
        }
        if schema.items != nil || schema.enumValues != nil {
            return ("$", "unsupported keyword on object parameters")
        }
        for (key, property) in (schema.properties ?? [:]).sorted(by: { $0.key < $1.key }) {
            if let issue = unsupportedSchemaIssue(property, path: "$.\(key)") {
                return issue
            }
        }
        return nil
    }

    private func unsupportedSchemaIssue(
        _ schema: JSONSchemaProperty, path: String
    ) -> (path: String, reason: String)? {
        if let issue = schema.validationIssue { return (path, issue) }
        let supportedTypes: Set<String> = ["object", "array", "string", "integer", "number", "boolean", "null"]
        guard supportedTypes.contains(schema.type) else { return (path, "unsupported type \(schema.type)") }
        if schema.enumValues != nil && schema.type != "string" {
            return (path, "only string enums are represented")
        }
        switch schema.type {
        case "object":
            if schema.items != nil { return (path, "items is only supported for arrays") }
            for (key, property) in (schema.properties ?? [:]).sorted(by: { $0.key < $1.key }) {
                if let issue = unsupportedSchemaIssue(property, path: "\(path).\(key)") {
                    return issue
                }
            }
        case "array":
            if schema.properties != nil || schema.required != nil || schema.additionalProperties != nil {
                return (path, "object keywords are not supported on arrays")
            }
            if let item = schema.items?.value,
               let issue = unsupportedSchemaIssue(item, path: "\(path)[]") {
                return issue
            }
        default:
            if schema.items != nil || schema.properties != nil || schema.required != nil
                || schema.additionalProperties != nil {
                return (path, "container keywords are not supported on scalar types")
            }
        }
        return nil
    }

    private func validateObject(
        _ object: [String: ToolCallJSONValue],
        properties: [String: JSONSchemaProperty],
        required: [String],
        additionalProperties: Bool?,
        toolName: String,
        path: String
    ) -> ToolCallValidationRefusal? {
        for name in required where object[name] == nil {
            return .invalidArguments(path: "\(path).\(name)", reason: "required")
        }
        for name in object.keys.sorted() {
            let childPath = "\(path).\(name)"
            guard let value = object[name] else { continue }
            guard let schema = properties[name] else {
                if additionalProperties == false {
                    return .invalidArguments(path: childPath, reason: "additionalProperty")
                }
                continue
            }
            if let refusal = validate(value, against: schema, toolName: toolName, path: childPath) {
                return refusal
            }
        }
        return nil
    }

    private func validate(
        _ value: ToolCallJSONValue,
        against schema: JSONSchemaProperty,
        toolName: String,
        path: String
    ) -> ToolCallValidationRefusal? {
        switch schema.type {
        case "object":
            guard case .object(let object) = value else {
                return .invalidArguments(path: path, reason: "type")
            }
            return validateObject(
                object,
                properties: schema.properties ?? [:],
                required: schema.required ?? [],
                additionalProperties: schema.additionalProperties,
                toolName: toolName,
                path: path)
        case "array":
            guard case .array(let values) = value else {
                return .invalidArguments(path: path, reason: "type")
            }
            if let item = schema.items?.value {
                for (index, element) in values.enumerated() {
                    if let refusal = validate(
                        element, against: item, toolName: toolName, path: "\(path)[\(index)]") {
                        return refusal
                    }
                }
            }
        case "string":
            guard case .string(let string) = value else {
                return .invalidArguments(path: path, reason: "type")
            }
            if let allowed = schema.enumValues, !allowed.contains(string) {
                return .invalidArguments(path: path, reason: "enum")
            }
        case "integer":
            guard value.isInteger else { return .invalidArguments(path: path, reason: "type") }
        case "number":
            guard case .number = value else { return .invalidArguments(path: path, reason: "type") }
        case "boolean":
            guard case .boolean = value else { return .invalidArguments(path: path, reason: "type") }
        case "null":
            guard case .null = value else { return .invalidArguments(path: path, reason: "type") }
        default:
            return .unsupportedSchema(
                toolName: toolName, path: path, reason: "unsupported type \(schema.type)")
        }
        return nil
    }
}
