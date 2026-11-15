import Foundation

// Gen UI (ZP-06): the model describes a small interactive card as DATA and the
// app draws it with its own native views. There is no HTML, no script and no
// web view anywhere on this path: the component set below is closed, every
// field is validated, and an unknown type or key rejects the whole spec with
// a named path so the model can fix it. A spec cannot call a tool, touch a
// file or reach the network; its only outward effect is a follow-up DRAFT the
// person reads and sends themselves.

enum GenUiLimits {
    static let maxSpecBytes = 16 * 1024
    static let maxComponents = 40
    static let maxLabelLength = 120
    static let maxTextLength = 1_000
    static let maxOptions = 12
    static let maxTableRows = 20
    static let maxTableColumns = 8
    static let maxBars = 24
    /// (max - min) / step, so a slider cannot ask for millions of steps.
    static let maxSteps = 10_000
    static let maxFollowUpLength = 2_000
}

enum GenUiError: Error, Equatable {
    case tooLarge
    case notJSON
    case invalid(String)

    var message: String {
        switch self {
        case .tooLarge: return "The spec is over \(GenUiLimits.maxSpecBytes) bytes."
        case .notJSON: return "The spec is not a JSON object."
        case .invalid(let detail): return detail
        }
    }
}

struct GenUiNumberControl: Equatable {
    let id: String
    let label: String
    let min: Double
    let max: Double
    let step: Double
    let value: Double
    let unit: String
}

enum GenUiTextStyle: String, Equatable {
    case heading, body, caption
}

struct GenUiBar: Equatable {
    let label: String
    let value: Double
}

enum GenUiComponent: Equatable {
    case text(String, GenUiTextStyle)
    case slider(GenUiNumberControl)
    case stepper(GenUiNumberControl)
    case toggle(id: String, label: String, value: Bool)
    case picker(id: String, label: String, options: [String], selected: Int)
    case readout(label: String, bind: String, unit: String)
    case table(header: [String], rows: [[String]])
    case bars(title: String, items: [GenUiBar])
    case followUp(label: String, template: String)
}

struct GenUiSpec: Equatable {
    let title: String
    let components: [GenUiComponent]

    /// The ids of controls that carry a value, with their kinds, for bind and
    /// template checks.
    var controlIDs: [String] {
        components.compactMap {
            switch $0 {
            case .slider(let c), .stepper(let c): return c.id
            case .toggle(let id, _, _): return id
            case .picker(let id, _, _, _): return id
            default: return nil
            }
        }
    }
}

enum GenUiParser {
    private static let idPattern = "^[a-z][a-z0-9_]{0,31}$"

    static func parse(_ json: String) -> Result<GenUiSpec, GenUiError> {
        guard json.utf8.count <= GenUiLimits.maxSpecBytes else { return .failure(.tooLarge) }
        guard let data = json.data(using: .utf8),
            let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return .failure(.notJSON) }
        do {
            return .success(try build(root))
        } catch let error as GenUiError {
            return .failure(error)
        } catch {
            return .failure(.notJSON)
        }
    }

    private static func fail(_ path: String, _ problem: String) -> GenUiError {
        .invalid("\(path): \(problem)")
    }

    /// Control characters are dropped; the text is only ever drawn verbatim.
    static func clean(_ text: String, limit: Int) -> String {
        let scalars = text.unicodeScalars.filter { $0.value == 0x0a || ($0.value >= 0x20 && $0.value != 0x7f) }
        return String(String.UnicodeScalarView(scalars)).prefix(limit).description
    }

    private static func requireKeys(_ dict: [String: Any], allowed: Set<String>, path: String) throws {
        for key in dict.keys where !allowed.contains(key) {
            throw fail(path, "unknown field '\(key.prefix(40))'")
        }
    }

    private static func string(_ dict: [String: Any], _ key: String, path: String, limit: Int, required: Bool = true) throws -> String {
        guard let raw = dict[key] else {
            if required { throw fail(path, "'\(key)' is required") }
            return ""
        }
        guard let value = raw as? String else { throw fail(path, "'\(key)' must be a string") }
        guard value.count <= limit else { throw fail(path, "'\(key)' is longer than \(limit) characters") }
        return clean(value, limit: limit)
    }

    private static func number(_ dict: [String: Any], _ key: String, path: String) throws -> Double {
        guard let raw = dict[key] as? NSNumber, CFGetTypeID(raw) != CFBooleanGetTypeID() else {
            throw fail(path, "'\(key)' must be a number")
        }
        let value = raw.doubleValue
        guard value.isFinite, abs(value) <= 1e12 else { throw fail(path, "'\(key)' is out of range") }
        return value
    }

    private static func id(_ dict: [String: Any], path: String) throws -> String {
        guard let value = dict["id"] as? String, value.range(of: idPattern, options: .regularExpression) != nil else {
            throw fail(path, "'id' must match [a-z][a-z0-9_]{0,31}")
        }
        return value
    }

    private static func control(_ dict: [String: Any], path: String) throws -> GenUiNumberControl {
        try requireKeys(dict, allowed: ["type", "id", "label", "min", "max", "step", "value", "unit"], path: path)
        let min = try number(dict, "min", path: path)
        let max = try number(dict, "max", path: path)
        let step = try number(dict, "step", path: path)
        let value = try number(dict, "value", path: path)
        guard min < max else { throw fail(path, "'min' must be below 'max'") }
        guard step > 0 else { throw fail(path, "'step' must be above 0") }
        guard (max - min) / step <= Double(GenUiLimits.maxSteps) else {
            throw fail(path, "more than \(GenUiLimits.maxSteps) steps between min and max")
        }
        guard value >= min, value <= max else { throw fail(path, "'value' is outside min and max") }
        return GenUiNumberControl(
            id: try id(dict, path: path), label: try string(dict, "label", path: path, limit: GenUiLimits.maxLabelLength),
            min: min, max: max, step: step, value: value,
            unit: try string(dict, "unit", path: path, limit: 12, required: false))
    }

    private static func build(_ root: [String: Any]) throws -> GenUiSpec {
        try requireKeys(root, allowed: ["title", "components"], path: "spec")
        let title = try string(root, "title", path: "spec", limit: GenUiLimits.maxLabelLength)
        guard let raw = root["components"] as? [Any], !raw.isEmpty else {
            throw fail("spec", "'components' must be a non-empty array")
        }
        guard raw.count <= GenUiLimits.maxComponents else {
            throw fail("spec", "more than \(GenUiLimits.maxComponents) components")
        }
        var components: [GenUiComponent] = []
        var ids = Set<String>()
        func claim(_ id: String, _ path: String) throws {
            guard ids.insert(id).inserted else { throw fail(path, "duplicate id '\(id)'") }
        }
        for (index, element) in raw.enumerated() {
            let path = "components[\(index)]"
            guard let dict = element as? [String: Any], let type = dict["type"] as? String else {
                throw fail(path, "must be an object with a 'type'")
            }
            switch type {
            case "text":
                try requireKeys(dict, allowed: ["type", "text", "style"], path: path)
                let style: GenUiTextStyle
                if let rawStyle = dict["style"] {
                    guard let name = rawStyle as? String, let parsed = GenUiTextStyle(rawValue: name) else {
                        throw fail(path, "'style' must be heading, body or caption")
                    }
                    style = parsed
                } else { style = .body }
                components.append(.text(try string(dict, "text", path: path, limit: GenUiLimits.maxTextLength), style))
            case "slider", "stepper":
                let c = try control(dict, path: path)
                try claim(c.id, path)
                components.append(type == "slider" ? .slider(c) : .stepper(c))
            case "toggle":
                try requireKeys(dict, allowed: ["type", "id", "label", "value"], path: path)
                let toggleID = try id(dict, path: path)
                try claim(toggleID, path)
                guard let value = dict["value"] as? Bool else { throw fail(path, "'value' must be true or false") }
                components.append(.toggle(id: toggleID, label: try string(dict, "label", path: path, limit: GenUiLimits.maxLabelLength), value: value))
            case "picker":
                try requireKeys(dict, allowed: ["type", "id", "label", "options", "value"], path: path)
                let pickerID = try id(dict, path: path)
                try claim(pickerID, path)
                guard let rawOptions = dict["options"] as? [Any], (2...GenUiLimits.maxOptions).contains(rawOptions.count) else {
                    throw fail(path, "'options' must hold 2 to \(GenUiLimits.maxOptions) strings")
                }
                var options: [String] = []
                for option in rawOptions {
                    guard let text = option as? String, !text.isEmpty, text.count <= GenUiLimits.maxLabelLength else {
                        throw fail(path, "an option is not a short non-empty string")
                    }
                    options.append(clean(text, limit: GenUiLimits.maxLabelLength))
                }
                let selected = (dict["value"] as? Int) ?? 0
                guard options.indices.contains(selected) else { throw fail(path, "'value' is not an option index") }
                components.append(.picker(id: pickerID, label: try string(dict, "label", path: path, limit: GenUiLimits.maxLabelLength), options: options, selected: selected))
            case "readout":
                try requireKeys(dict, allowed: ["type", "label", "bind", "unit"], path: path)
                components.append(.readout(
                    label: try string(dict, "label", path: path, limit: GenUiLimits.maxLabelLength),
                    bind: try string(dict, "bind", path: path, limit: 32),
                    unit: try string(dict, "unit", path: path, limit: 12, required: false)))
            case "table":
                try requireKeys(dict, allowed: ["type", "header", "rows"], path: path)
                guard let header = dict["header"] as? [String], (1...GenUiLimits.maxTableColumns).contains(header.count),
                    header.allSatisfy({ $0.count <= GenUiLimits.maxLabelLength })
                else { throw fail(path, "'header' must hold 1 to \(GenUiLimits.maxTableColumns) short strings") }
                guard let rawRows = dict["rows"] as? [[Any]], rawRows.count <= GenUiLimits.maxTableRows else {
                    throw fail(path, "'rows' must be at most \(GenUiLimits.maxTableRows) arrays")
                }
                var rows: [[String]] = []
                for row in rawRows {
                    guard row.count == header.count else { throw fail(path, "every row needs \(header.count) cells") }
                    rows.append(try row.map { cell in
                        if let text = cell as? String, text.count <= GenUiLimits.maxLabelLength { return clean(text, limit: GenUiLimits.maxLabelLength) }
                        if let n = cell as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite { return formatted(n.doubleValue, step: 0.0001) }
                        throw fail(path, "a cell is not a short string or a number")
                    })
                }
                components.append(.table(header: header.map { clean($0, limit: GenUiLimits.maxLabelLength) }, rows: rows))
            case "bars":
                try requireKeys(dict, allowed: ["type", "title", "items"], path: path)
                guard let rawItems = dict["items"] as? [[String: Any]], (1...GenUiLimits.maxBars).contains(rawItems.count) else {
                    throw fail(path, "'items' must hold 1 to \(GenUiLimits.maxBars) objects")
                }
                var items: [GenUiBar] = []
                for (i, item) in rawItems.enumerated() {
                    let itemPath = "\(path).items[\(i)]"
                    try requireKeys(item, allowed: ["label", "value"], path: itemPath)
                    let value = try number(item, "value", path: itemPath)
                    guard value >= 0 else { throw fail(itemPath, "'value' must be 0 or more") }
                    items.append(GenUiBar(label: try string(item, "label", path: itemPath, limit: 40), value: value))
                }
                components.append(.bars(title: try string(dict, "title", path: path, limit: GenUiLimits.maxLabelLength, required: false), items: items))
            case "follow_up":
                try requireKeys(dict, allowed: ["type", "label", "template"], path: path)
                components.append(.followUp(
                    label: try string(dict, "label", path: path, limit: 60),
                    template: try string(dict, "template", path: path, limit: GenUiLimits.maxFollowUpLength)))
            default:
                throw fail(path, "unknown component type '\(type.prefix(40))'")
            }
        }
        // Binds must name a numeric or toggle control that exists.
        let controls = Set(components.compactMap { c -> String? in
            switch c {
            case .slider(let n), .stepper(let n): return n.id
            case .toggle(let id, _, _): return id
            case .picker(let id, _, _, _): return id
            default: return nil
            }
        })
        for (index, component) in components.enumerated() {
            if case .readout(_, let bind, _) = component, !controls.contains(bind) {
                throw fail("components[\(index)]", "'bind' names no control id")
            }
        }
        return GenUiSpec(title: title, components: components)
    }

    /// Decimal places implied by a step, at most 4.
    static func formatted(_ value: Double, step: Double) -> String {
        var places = 0
        var scaled = step
        while places < 4, scaled.rounded() != scaled {
            scaled *= 10
            places += 1
        }
        return String(format: "%.\(places)f", value)
    }
}

/// Current values of a card's controls, and the follow-up they expand into.
struct GenUiValues: Equatable {
    var numbers: [String: Double] = [:]
    var toggles: [String: Bool] = [:]
    var picks: [String: Int] = [:]

    init(_ spec: GenUiSpec) {
        for component in spec.components {
            switch component {
            case .slider(let c), .stepper(let c): numbers[c.id] = c.value
            case .toggle(let id, _, let value): toggles[id] = value
            case .picker(let id, _, _, let selected): picks[id] = selected
            default: break
            }
        }
    }

    /// Replaces `{id}` with the control's current value. Unknown ids stay
    /// literal. The result is clipped. Nothing is evaluated: it is plain
    /// substitution into text that becomes a composer DRAFT.
    func expand(_ template: String, in spec: GenUiSpec) -> String {
        var text = template
        for component in spec.components {
            switch component {
            case .slider(let c), .stepper(let c):
                text = text.replacingOccurrences(of: "{\(c.id)}", with: GenUiParser.formatted(numbers[c.id] ?? c.value, step: c.step))
            case .toggle(let id, _, _):
                text = text.replacingOccurrences(of: "{\(id)}", with: (toggles[id] ?? false) ? "on" : "off")
            case .picker(let id, _, let options, _):
                let index = picks[id] ?? 0
                text = text.replacingOccurrences(of: "{\(id)}", with: options.indices.contains(index) ? options[index] : "")
            default: break
            }
        }
        return String(text.prefix(GenUiLimits.maxFollowUpLength))
    }
}
