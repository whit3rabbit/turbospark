import Foundation

/// Small JSON helpers shared by the two family clients.
enum RemoteJSON {
    static func object(_ text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8),
            let parsed = try? JSONSerialization.jsonObject(with: data)
        else { return nil }
        return parsed as? [String: Any]
    }

    static func data(_ object: Any) throws -> Data {
        guard JSONSerialization.isValidJSONObject(object) else {
            throw RemoteProviderError.requestRefused(status: 0, detail: "request is not valid JSON")
        }
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    static func string(_ object: Any) -> String {
        guard JSONSerialization.isValidJSONObject(object),
            let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }

    /// A tool call's accumulated argument text must be one JSON object. An
    /// empty buffer is the empty object (a call with no parameters).
    static func validatedArguments(_ buffer: String) throws -> String {
        let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "{}" }
        guard object(trimmed) != nil else {
            throw RemoteProviderError.badStream("tool call arguments were cut off or not an object")
        }
        return trimmed
    }

    static func toolSchema(_ spec: RemoteToolSpec) throws -> [String: Any] {
        guard let schema = object(spec.parametersJSON) else {
            throw RemoteProviderError.requestRefused(status: 0, detail: "tool schema for \(spec.name) is not an object")
        }
        return schema
    }
}
