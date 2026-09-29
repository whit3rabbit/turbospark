import Foundation
import OpenKind

enum TypeSafeStateMode: String, CaseIterable {
    case text, json
}

enum TypeSafeQuestionKind: String, CaseIterable {
    case noul, choice, score
}

struct TypeSafeChoiceOption: Identifiable {
    let id = UUID()
    var key: String
    var detail: String
}

struct TypeSafeQuestionDraft: Identifiable {
    let id = UUID()
    var name: String
    var kind: TypeSafeQuestionKind
    var instructions: String
    var trueCriteria = ""
    var falseCriteria = ""
    var options: [TypeSafeChoiceOption] = []
    var levels: [String] = []

    static func new(_ kind: TypeSafeQuestionKind, index: Int) -> Self {
        switch kind {
        case .noul:
            return .init(name: "question_\(index)", kind: kind, instructions: "")
        case .choice:
            return .init(
                name: "question_\(index)", kind: kind, instructions: "",
                options: [
                    .init(key: "option_a", detail: ""),
                    .init(key: "option_b", detail: ""),
                    .init(key: "__none__", detail: "None of the listed options applies.")
                ])
        case .score:
            return .init(name: "question_\(index)", kind: kind, instructions: "", levels: ["Low", "High"])
        }
    }
}

struct TypeSafePlaygroundError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

struct TypeSafePlaygroundDraft {
    var model = "mock"
    var stateMode: TypeSafeStateMode = .text
    var state = "A package arrived late and the customer needs a response today."
    var questions = [TypeSafeQuestionDraft(
        name: "urgent", kind: .noul,
        instructions: "Does this need immediate attention?")]

    func request() throws -> SystemRequest {
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else { throw TypeSafePlaygroundError(message: "Select a loaded model.") }
        let stateValue: JSONValue
        switch stateMode {
        case .text:
            guard !state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw TypeSafePlaygroundError(message: "State must not be empty.")
            }
            stateValue = .string(state)
        case .json:
            let parsed = try JSONDecoder().decode(JSONValue.self, from: Data(state.utf8))
            switch parsed {
            case .string(let value) where !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty:
                stateValue = parsed
            case .object, .array:
                stateValue = parsed
            default:
                throw TypeSafePlaygroundError(message: "State JSON must be a nonempty string, object, or array.")
            }
        }

        guard !questions.isEmpty else { throw TypeSafePlaygroundError(message: "Add at least one question.") }
        var result: [String: Question] = [:]
        for item in questions {
            let name = item.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { throw TypeSafePlaygroundError(message: "Every question needs an ID.") }
            guard result[name] == nil else {
                throw TypeSafePlaygroundError(message: "Question IDs must be unique: \(name).")
            }
            let instructions = item.instructions.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !instructions.isEmpty else {
                throw TypeSafePlaygroundError(message: "Question \(name) needs instructions.")
            }
            switch item.kind {
            case .noul:
                let trueValue = item.trueCriteria.trimmingCharacters(in: .whitespacesAndNewlines)
                let falseValue = item.falseCriteria.trimmingCharacters(in: .whitespacesAndNewlines)
                guard trueValue.isEmpty == falseValue.isEmpty else {
                    throw TypeSafePlaygroundError(message: "Question \(name) needs both true and false criteria.")
                }
                let criteria = trueValue.isEmpty ? nil : NoulCriteria(true: trueValue, false: falseValue)
                result[name] = .noul(instructions: .string(instructions), criteria: criteria)
            case .choice:
                guard !item.options.isEmpty else {
                    throw TypeSafePlaygroundError(message: "Choice question \(name) needs options.")
                }
                var criteria: [String: String?] = [:]
                for option in item.options {
                    let key = option.key.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !key.isEmpty, criteria[key] == nil else {
                        throw TypeSafePlaygroundError(message: "Choice question \(name) has an empty or repeated option key.")
                    }
                    let detail = option.detail.trimmingCharacters(in: .whitespacesAndNewlines)
                    criteria[key] = detail.isEmpty ? .some(nil) : .some(detail)
                }
                result[name] = .choice(instructions: .string(instructions), criteria: criteria)
            case .score:
                let levels = item.levels.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                guard levels.count >= 2, levels.allSatisfy({ !$0.isEmpty }) else {
                    throw TypeSafePlaygroundError(message: "Score question \(name) needs at least two named levels.")
                }
                result[name] = .score(instructions: .string(instructions), criteria: levels)
            }
        }
        return SystemRequest(state: stateValue, model: model, questions: result)
    }

    static func from(_ request: SystemRequest) throws -> Self {
        var draft = Self()
        draft.model = request.model
        switch request.state {
        case .string(let value):
            draft.stateMode = .text
            draft.state = value
        case .object, .array:
            draft.stateMode = .json
            draft.state = try Self.pretty(request.state)
        default:
            throw TypeSafePlaygroundError(message: "The form supports string, object, or array state.")
        }
        draft.questions = try request.questions.sorted { $0.key < $1.key }.map { name, question in
            let instructions: JSONValue
            switch question {
            case .noul(let value, _), .choice(let value, _), .score(let value, _): instructions = value
            }
            guard case .string(let text) = instructions else {
                throw TypeSafePlaygroundError(message: "The form supports string instructions. Keep this request in Raw JSON.")
            }
            switch question {
            case .noul(_, let criteria):
                return TypeSafeQuestionDraft(
                    name: name, kind: .noul, instructions: text,
                    trueCriteria: criteria?.true ?? "", falseCriteria: criteria?.false ?? "")
            case .choice(_, let criteria):
                return TypeSafeQuestionDraft(
                    name: name, kind: .choice, instructions: text,
                    options: criteria.sorted { $0.key < $1.key }.map {
                        TypeSafeChoiceOption(key: $0.key, detail: $0.value ?? "")
                    })
            case .score(_, let levels):
                return TypeSafeQuestionDraft(name: name, kind: .score, instructions: text, levels: levels)
            }
        }
        return draft
    }

    static func pretty<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

enum TypeSafePlaygroundPreset: String, CaseIterable, Identifiable {
    case urgency = "noul"
    case triage = "choice"
    case frustration = "score"
    case mixed = "mixed"
    case structured = "JSON state"
    case semanticNone = "semantic none"

    var id: String { rawValue }

    func draft(model: String) -> TypeSafePlaygroundDraft {
        var draft = TypeSafePlaygroundDraft()
        draft.model = model
        draft.state = "Help! My payouts have been failing for 3 days."
        switch self {
        case .urgency:
            var question = TypeSafeQuestionDraft.new(.noul, index: 1)
            question.name = "is_urgent"
            question.instructions = "Does this convey urgency?"
            question.trueCriteria = "Explicitly time-sensitive"
            question.falseCriteria = "No urgency expressed"
            draft.questions = [question]
        case .triage, .semanticNone:
            var question = TypeSafeQuestionDraft.new(.choice, index: 1)
            question.name = "department"
            question.instructions = "Which team should handle this?"
            question.options = [
                .init(key: "billing", detail: "Payments, invoicing, refunds"),
                .init(key: "technical", detail: "Bugs, outages, integrations"),
                .init(key: "sales", detail: "Pricing, upgrades, new accounts"),
                .init(key: "__none__", detail: "None of the listed options applies.")
            ]
            if self == .semanticNone {
                draft.state = "My card was charged twice and I need this fixed today."
                question.name = "route"
                question.instructions = "Which workflow owns this issue?"
            }
            draft.questions = [question]
        case .frustration:
            var question = TypeSafeQuestionDraft.new(.score, index: 1)
            question.name = "frustration"
            question.instructions = "How frustrated is the customer?"
            question.levels = ["Calm", "Frustrated", "Very angry"]
            draft.questions = [question]
        case .mixed:
            draft.questions = [
                TypeSafePlaygroundPreset.urgency.draft(model: model).questions[0],
                TypeSafePlaygroundPreset.triage.draft(model: model).questions[0],
                TypeSafePlaygroundPreset.frustration.draft(model: model).questions[0]
            ]
        case .structured:
            draft.stateMode = .json
            draft.state = """
                {"user":{"id":123,"messages":[]},"logs":[],"diff":"pending"}
                """
            draft.questions = TypeSafePlaygroundPreset.triage.draft(model: model).questions
        }
        return draft
    }
}

struct TypeSafeLatencySummary {
    let count: Int
    let mean: Double
    let p50: Double
    let p95: Double
    let minimum: Double
    let maximum: Double

    init?(_ samples: [Double]) {
        guard !samples.isEmpty else { return nil }
        let sorted = samples.sorted()
        count = sorted.count
        mean = sorted.reduce(0, +) / Double(sorted.count)
        p50 = sorted[max(0, Int(ceil(0.5 * Double(sorted.count))) - 1)]
        p95 = sorted[max(0, Int(ceil(0.95 * Double(sorted.count))) - 1)]
        minimum = sorted[0]
        maximum = sorted[sorted.count - 1]
    }
}
