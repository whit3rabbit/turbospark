import Foundation

/// Renders a tool's JSON Schema `inputSchema` as a compact TypeScript
/// parameter declaration for the `## Deferred MCP Tools` listing.
///
/// Why this exists: a script that calls a tool with a wrong argument name
/// fails, and the model then spends a turn on `tool_describe` to find out,
/// which is exactly the round trip codemode is meant to remove. With the
/// shape in the prompt the first script is usually right. pi's codemode does
/// the same with `renderDeclarations`.
///
/// This is deliberately a lossy summary, not a validator: unions are capped,
/// `$ref` expansion is bounded, nesting is bounded, and anything it cannot
/// express becomes `unknown`. A schema too large to inline degrades to
/// `Record<string, unknown>` and the model falls back to `tool_describe`.
enum CodemodeSchemaRenderer {
    /// Longest `args` declaration kept on one listing line, in characters.
    static let defaultMaximumCharacters = 700

    private static let maximumDepth = 4
    private static let maximumRefExpansions = 32
    private static let maximumEnumMembers = 12
    private static let maximumUnionMembers = 6
    private static let maximumTupleMembers = 6

    /// The `args` parameter of a tool, for example
    /// `args: { path: string; limit?: number }`. The parameter is optional
    /// (`args?:`) when nothing in the schema is required.
    static func argumentsDeclaration(
        schemaJSON: String, maximumCharacters: Int = defaultMaximumCharacters
    ) -> String {
        guard let data = schemaJSON.data(using: .utf8),
              let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              !root.isEmpty
        else { return "args?: Record<string, unknown>" }

        let optional = (root["required"] as? [String] ?? []).isEmpty
        // Full first; if that is too long, collapse nested objects to
        // `object` and try once more.
        for shallow in [false, true] {
            let body = Renderer(root: root, shallow: shallow).render(root, depth: 0)
            let text = "args\(optional ? "?" : ""): \(body)"
            if text.count <= maximumCharacters { return text }
        }
        return "args: Record<string, unknown>"
    }

    // MARK: Rendering

    private final class Renderer {
        let root: [String: Any]
        let shallow: Bool
        private var refExpansions = 0
        private var activeRefs: Set<String> = []

        init(root: [String: Any], shallow: Bool) {
            self.root = root
            self.shallow = shallow
        }

        func render(_ schema: Any, depth: Int) -> String {
            guard let schema = schema as? [String: Any] else { return "unknown" }

            if let reference = schema["$ref"] as? String {
                return resolve(reference, depth: depth)
            }
            if let constant = schema["const"], let literal = Self.literal(constant) {
                return literal
            }
            if let values = schema["enum"] as? [Any], !values.isEmpty,
               values.count <= CodemodeSchemaRenderer.maximumEnumMembers
            {
                let literals = values.compactMap(Self.literal)
                if literals.count == values.count { return Self.unique(literals).joined(separator: " | ") }
            }
            for key in ["oneOf", "anyOf"] {
                if let members = schema[key] as? [Any], !members.isEmpty {
                    return union(members.map { render($0, depth: depth) })
                }
            }
            if let members = schema["allOf"] as? [Any], !members.isEmpty {
                let rendered = Self.unique(members.map { render($0, depth: depth) })
                return rendered.count == 1 ? rendered[0] : rendered.joined(separator: " & ")
            }

            var result: String
            if let names = schema["type"] as? [String] {
                result = union(names.map { base($0, schema, depth: depth) })
            } else if let name = schema["type"] as? String {
                result = base(name, schema, depth: depth)
            } else if schema["properties"] != nil {
                result = object(schema, depth: depth)
            } else if schema["items"] != nil {
                result = array(schema, depth: depth)
            } else {
                result = "unknown"
            }
            // OpenAPI-style nullable, which some servers emit.
            if (schema["nullable"] as? Bool) == true, !result.contains("null") {
                result += " | null"
            }
            return result
        }

        private func base(_ name: String, _ schema: [String: Any], depth: Int) -> String {
            switch name {
            case "string": return "string"
            case "number", "integer": return "number"
            case "boolean": return "boolean"
            case "null": return "null"
            case "array": return array(schema, depth: depth)
            case "object": return object(schema, depth: depth)
            default: return "unknown"
            }
        }

        private func array(_ schema: [String: Any], depth: Int) -> String {
            if let tuple = schema["items"] as? [Any] {
                let members = tuple.prefix(CodemodeSchemaRenderer.maximumTupleMembers)
                    .map { render($0, depth: depth + 1) }
                return "[" + members.joined(separator: ", ") + "]"
            }
            guard let items = schema["items"] else { return "unknown[]" }
            let element = render(items, depth: depth + 1)
            // A union or intersection needs parentheses before `[]`.
            let wrapped = element.contains(" | ") || element.contains(" & ") ? "(\(element))" : element
            return wrapped + "[]"
        }

        private func object(_ schema: [String: Any], depth: Int) -> String {
            // Past the depth bound, or inside any nested level of a shallow
            // pass, an object is just `object`: the model can `tool_describe`.
            if depth >= CodemodeSchemaRenderer.maximumDepth || (shallow && depth >= 1) {
                return "object"
            }
            let properties = schema["properties"] as? [String: Any] ?? [:]
            let required = Set(schema["required"] as? [String] ?? [])
            if properties.isEmpty {
                if let additional = schema["additionalProperties"] as? [String: Any] {
                    return "Record<string, \(render(additional, depth: depth + 1))>"
                }
                if (schema["additionalProperties"] as? Bool) == true {
                    return "Record<string, unknown>"
                }
                return "{}"
            }
            // JSON object order does not survive parsing, so order is
            // deterministic instead: required first, then by name.
            let ordered = properties.keys.sorted {
                (required.contains($0) ? 0 : 1, $0) < (required.contains($1) ? 0 : 1, $1)
            }
            let members = ordered.map { key -> String in
                let mark = required.contains(key) ? "" : "?"
                return "\(Self.propertyName(key))\(mark): \(render(properties[key] ?? [:], depth: depth + 1))"
            }
            return "{ " + members.joined(separator: "; ") + " }"
        }

        private func union(_ members: [String]) -> String {
            let unique = Self.unique(members)
            if unique.contains("unknown") || unique.count > CodemodeSchemaRenderer.maximumUnionMembers {
                return "unknown"
            }
            return unique.joined(separator: " | ")
        }

        /// Follows a local `#/...` reference. Bounded in total expansions and
        /// guarded against a cycle, so a self-referential or heavily shared
        /// definition cannot blow the output up.
        private func resolve(_ reference: String, depth: Int) -> String {
            guard reference.hasPrefix("#/"),
                  !activeRefs.contains(reference),
                  refExpansions < CodemodeSchemaRenderer.maximumRefExpansions,
                  let target = pointer(reference)
            else { return "unknown" }
            refExpansions += 1
            activeRefs.insert(reference)
            defer { activeRefs.remove(reference) }
            return render(target, depth: depth)
        }

        private func pointer(_ reference: String) -> Any? {
            var node: Any = root
            for rawPart in reference.dropFirst(2).split(separator: "/", omittingEmptySubsequences: false) {
                let part = rawPart.replacingOccurrences(of: "~1", with: "/")
                    .replacingOccurrences(of: "~0", with: "~")
                if let dictionary = node as? [String: Any], let next = dictionary[part] {
                    node = next
                } else if let array = node as? [Any], let index = Int(part), array.indices.contains(index) {
                    node = array[index]
                } else {
                    return nil
                }
            }
            return node
        }

        // MARK: Literals and names

        static func literal(_ value: Any) -> String? {
            if value is NSNull { return "null" }
            if let string = value as? String { return quoted(string) }
            guard let number = value as? NSNumber else { return nil }
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "true" : "false" }
            let double = number.doubleValue
            if double == double.rounded(), abs(double) < 1e15 { return String(Int64(double)) }
            return "\(double)"
        }

        static func propertyName(_ key: String) -> String {
            var first = true
            for scalar in key.unicodeScalars {
                let letter = (scalar.value >= 65 && scalar.value <= 90)
                    || (scalar.value >= 97 && scalar.value <= 122)
                    || scalar == "_" || scalar == "$"
                let digit = scalar.value >= 48 && scalar.value <= 57
                guard letter || (digit && !first) else { return quoted(key) }
                first = false
            }
            return key.isEmpty ? quoted(key) : key
        }

        private static func quoted(_ string: String) -> String {
            guard let data = try? JSONEncoder().encode(string),
                  let text = String(data: data, encoding: .utf8)
            else { return "\"\"" }
            return text
        }

        static func unique(_ values: [String]) -> [String] {
            var seen: Set<String> = []
            return values.filter { seen.insert($0).inserted }
        }
    }
}
