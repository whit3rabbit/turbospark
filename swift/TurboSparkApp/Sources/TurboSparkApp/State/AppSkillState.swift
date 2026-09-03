import Foundation

/// SKILL.state: the bounded execution state an agent carries between steps,
/// instead of an append-only transcript. See `docs/SKILL_STATE.md` for the
/// measurement that justifies this existing (arXiv 2608.26263).
///
/// The semantics here MIRROR `scripts/skill_state_probe.py` deliberately: that
/// script is the evidence, so a divergence between the two makes the measured
/// numbers stop describing this code. In particular the patch is an RFC 7386
/// JSON Merge Patch (arrays replace wholesale, null deletes), and extraction
/// strips markdown fences, which the probe found was the ONLY formatting
/// hazard in practice and is entirely dialect-dependent.

/// A JSON value, because the state is a document rather than a fixed record and
/// a merge patch has to be able to carry an explicit null.
public enum AppJSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([AppJSONValue])
    case object([String: AppJSONValue])

    public init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let v = try? c.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? c.decode(Double.self) {
            self = .number(v)
        } else if let v = try? c.decode(String.self) {
            self = .string(v)
        } else if let v = try? c.decode([AppJSONValue].self) {
            self = .array(v)
        } else if let v = try? c.decode([String: AppJSONValue].self) {
            self = .object(v)
        } else {
            throw DecodingError.dataCorruptedError(
                in: c, debugDescription: "unsupported JSON value")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }

    /// The `Any` tree `JSONSerialization` wants, for rendering.
    var foundationValue: Any {
        switch self {
        case .null: return NSNull()
        case .bool(let v): return v
        case .number(let v):
            // `Int(v)` TRAPS above `Int.max`, and this is reached for a
            // persisted `skillState` as well as for validated model output --
            // a number from disk has been through no validator at all.
            guard v.isFinite, v >= -9_007_199_254_740_992, v <= 9_007_199_254_740_992 else {
                return v
            }
            return v == v.rounded() ? Int(v) : v
        case .string(let v): return v
        case .array(let v): return v.map(\.foundationValue)
        case .object(let v): return v.mapValues(\.foundationValue)
        }
    }

    static func from(_ any: Any) -> AppJSONValue {
        switch any {
        case is NSNull: return .null
        case let v as Bool where type(of: any) == type(of: NSNumber(value: true)):
            return .bool(v)
        case let v as NSNumber:
            if CFGetTypeID(v) == CFBooleanGetTypeID() { return .bool(v.boolValue) }
            return .number(v.doubleValue)
        case let v as String: return .string(v)
        case let v as [Any]: return .array(v.map(AppJSONValue.from))
        case let v as [String: Any]: return .object(v.mapValues(AppJSONValue.from))
        default: return .null
        }
    }
}

/// What a field of the generic coding-agent state is allowed to hold.
public enum AppSkillStateFieldKind: Sendable {
    /// A plain string, such as the current goal.
    case text
    /// An ordered list of short strings, such as facts learned.
    case textList
    /// A map of key to short string, such as path to what is known about it.
    case textMap
}

/// The generic coding-agent schema.
///
/// The paper authors a schema per domain and names "no fixed schema known in
/// advance" as its first limitation, which a general assistant with a shell is
/// arguably an instance of. This is the compromise: one schema covering what a
/// coding agent actually carries between steps, close in shape to the paper's
/// own five-field InterCode CTF schema.
public enum AppSkillStateSchema {
    public static let fields: [(name: String, kind: AppSkillStateFieldKind, doc: String)] = [
        ("goal", .text, "the objective in one sentence, restated as it sharpens"),
        ("files", .textMap, "path -> what is known about it or what was changed"),
        ("facts", .textList, "durable findings that later steps depend on"),
        ("commands", .textList, "command run -> its one-line outcome"),
        ("next", .textList, "open questions and remaining steps"),
    ]

    static func kind(of field: String) -> AppSkillStateFieldKind? {
        fields.first { $0.name == field }?.kind
    }

    /// The schema as the model is told it, which is also what the runtime
    /// enforces. One source, so a drift between prompt and validator is not
    /// expressible.
    public static var promptDescription: String {
        let body = fields.map { field -> String in
            let shape: String
            switch field.kind {
            case .text: shape = "\"<text>\""
            case .textList: shape = "[\"<text>\", ...]"
            case .textMap: shape = "{\"<key>\": \"<text>\", ...}"
            }
            return "  \"\(field.name)\": \(shape),   // \(field.doc)"
        }.joined(separator: "\n")
        return "{\n\(body)\n}"
    }
}

/// The execution state itself: a validated JSON object over the schema above.
public struct AppSkillState: Codable, Equatable, Sendable {
    public var fields: [String: AppJSONValue]

    public init(fields: [String: AppJSONValue] = [:]) {
        self.fields = fields
    }

    /// Tolerant decode (state#45): this hangs off `AppChat.skillState`, so a
    /// synthesized decoder here can quarantine every conversation over one
    /// added field. A state that will not decode is bookkeeping for one agent
    /// run, and losing it costs a step; losing the archive costs everything.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        fields = try container.decodeIfPresent([String: AppJSONValue].self, forKey: .fields) ?? [:]
    }

    public var isEmpty: Bool { fields.isEmpty }

    /// Canonical rendering: sorted keys and stable formatting, so an unchanged
    /// field never moves. That is what keeps the prompt's shared prefix as long
    /// as it can be (docs/SKILL_STATE.md, "Interaction with prefix KV reuse").
    public var rendered: String {
        guard !fields.isEmpty else { return "{}" }
        let tree = fields.mapValues(\.foundationValue)
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: tree, options: [.prettyPrinted, .sortedKeys]),
            let text = String(data: data, encoding: .utf8)
        else { return "{}" }
        return text
    }

    /// Applies an RFC 7386 JSON Merge Patch. Arrays replace wholesale; an
    /// explicit null deletes the key.
    public mutating func apply(patch: [String: AppJSONValue]) {
        for (key, value) in patch {
            if case .null = value {
                fields.removeValue(forKey: key)
            } else if case .object(let sub) = value,
                case .object(var existing)? = fields[key] {
                for (k, v) in sub {
                    if case .null = v { existing.removeValue(forKey: k) } else { existing[k] = v }
                }
                fields[key] = .object(existing)
            } else {
                fields[key] = value
            }
        }
    }

    public func applying(patch: [String: AppJSONValue]) -> AppSkillState {
        var copy = self
        copy.apply(patch: patch)
        return copy
    }
}

/// Parsing and validation of a model-proposed patch.
public enum AppSkillStatePatch {
    /// The bounds a "bounded execution state" actually has (state#58).
    ///
    /// **THE VALIDATOR CHECKED TYPES AND NOTHING ELSE**, so the whole point
    /// of this feature -- a prompt that is O(1) in step count
    /// (`docs/SKILL_STATE.md`) -- rested on the model choosing to be brief.
    /// A run that keeps appending to `facts` grows the state without limit
    /// and reproduces exactly the context exhaustion the append-only path was
    /// measured to hit, with the toggle on and no sign of why.
    ///
    /// Generous rather than tight: these are the size at which a state has
    /// stopped being a summary, not a budget anyone should feel.
    public static let maxStringLength = 4_000
    public static let maxCollectionCount = 64
    public static let maxRenderedBytes = 32_000

    /// Schema violations in a proposed patch. Empty means valid.
    public static func validate(_ patch: [String: AppJSONValue]) -> [String] {
        var errors: [String] = []
        for (key, value) in patch {
            guard let kind = AppSkillStateSchema.kind(of: key) else {
                errors.append("unknown field '\(key)'")
                continue
            }
            if case .null = value { continue }  // deletion is legal for any field
            switch kind {
            case .text:
                guard case .string(let text) = value else {
                    errors.append("'\(key)' must be a string")
                    continue
                }
                if text.count > maxStringLength {
                    errors.append(
                        "'\(key)' is \(text.count) characters; the limit is \(maxStringLength)")
                }
            case .textList:
                guard case .array(let items) = value else {
                    errors.append("'\(key)' must be an array of strings")
                    continue
                }
                if items.contains(where: { if case .string = $0 { return false } else { return true } }) {
                    errors.append("'\(key)' must contain only strings")
                }
                if items.count > maxCollectionCount {
                    errors.append(
                        "'\(key)' has \(items.count) entries; the limit is \(maxCollectionCount). "
                        + "Summarize or drop the ones you no longer need.")
                }
                for case .string(let text) in items where text.count > maxStringLength {
                    errors.append(
                        "an entry of '\(key)' is \(text.count) characters; the limit is "
                        + "\(maxStringLength)")
                    break
                }
            case .textMap:
                guard case .object(let entries) = value else {
                    errors.append("'\(key)' must be an object of strings")
                    continue
                }
                for (k, v) in entries {
                    if case .string(let text) = v {
                        if text.count > maxStringLength {
                            errors.append(
                                "'\(key).\(k)' is \(text.count) characters; the limit is "
                                + "\(maxStringLength)")
                        }
                        continue
                    }
                    if case .null = v { continue }  // deletes one entry
                    errors.append("'\(key).\(k)' must be a string")
                }
                if entries.count > maxCollectionCount {
                    errors.append(
                        "'\(key)' has \(entries.count) entries; the limit is "
                        + "\(maxCollectionCount)")
                }
            }
        }
        return errors
    }

    /// Pulls a patch out of a model turn.
    ///
    /// Looks for `<state_patch>...</state_patch>` first, then falls back to the
    /// first balanced JSON object in the text. The fallback is what handles
    /// markdown fences, which the probe measured as the only real formatting
    /// hazard: gemma4 fenced 92% of its replies and gptoss none, and rescue
    /// took all three installs to a 1.00 valid-patch rate.
    ///
    /// Returns nil when no patch is present, which is NOT an error: a step that
    /// changes nothing correctly carries no patch.
    public static func extract(from text: String) -> [String: AppJSONValue]? {
        let body = taggedBody(in: text) ?? text
        guard let object = firstJSONObject(in: body) else { return nil }
        if let inner = object["patch"], case .object(let patch) = inner { return patch }
        // A bare object is only a patch when it came from the tag; otherwise it
        // is far more likely to be a tool call or an example the model wrote.
        return taggedBody(in: text) != nil ? object : nil
    }

    static func taggedBody(in text: String) -> String? {
        guard let open = text.range(of: "<state_patch>"),
            let close = text.range(of: "</state_patch>", range: open.upperBound..<text.endIndex)
        else { return nil }
        return String(text[open.upperBound..<close.lowerBound])
    }

    /// First balanced `{...}` in `text`, tracking string literals so a brace
    /// inside a quoted value cannot end the scan early.
    static func firstJSONObject(in text: String) -> [String: AppJSONValue]? {
        var depth = 0
        var start: String.Index?
        var inString = false
        var escaped = false
        for index in text.indices {
            let ch = text[index]
            if inString {
                if escaped { escaped = false } else if ch == "\\" { escaped = true } else if ch == "\"" { inString = false }
                continue
            }
            switch ch {
            case "\"": inString = true
            case "{":
                if depth == 0 { start = index }
                depth += 1
            case "}":
                // Clamped at zero: a stray closing brace before any opening
                // one drove `depth` negative, and every later `{` then
                // incremented from there without ever reaching the `depth ==
                // 0` that starts an object -- so ONE unbalanced `}` disabled
                // the scanner for the whole rest of the string.
                depth = max(0, depth - 1)
                if depth == 0, let from = start {
                    let slice = String(text[from...index])
                    if let data = slice.data(using: .utf8),
                        let parsed = try? JSONDecoder().decode(
                            [String: AppJSONValue].self, from: data) {
                        return parsed
                    }
                    start = nil
                }
            default: break
            }
        }
        return nil
    }
}
