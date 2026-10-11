import OpenKind
import SwiftUI

/// One question of the TypeSafe playground form, and one answer of its
/// output. Split from TypeSafePlaygroundView to keep both files readable.
struct TypeSafeQuestionCard: View {
    @Binding var question: TypeSafeQuestionDraft
    let remove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                TextField(text: $question.name, prompt: Text("Question ID", bundle: .module)) {
                    Text("Question ID", bundle: .module)
                }
                Picker(selection: $question.kind) {
                    ForEach(TypeSafeQuestionKind.allCases, id: \.self) { kind in
                        Text(verbatim: kind.rawValue).tag(kind)
                    }
                } label: { Text("Question type", bundle: .module) }
                .labelsHidden()
                .frame(minWidth: 100, idealWidth: 135, maxWidth: 160)
                Button(role: .destructive, action: remove) { Text("Remove", bundle: .module) }
                    .appAction(.destructive, size: APIControlSize.row)
            }
            TextField(text: $question.instructions, prompt: Text("Instructions", bundle: .module)) {
                Text("Instructions", bundle: .module)
            }
            switch question.kind {
            case .noul:
                // The placeholders are the schema's own outcome names, so they
                // stay as written; the accessible names say what each holds.
                HStack {
                    TextField(text: $question.trueCriteria, prompt: Text(verbatim: "true")) {
                        Text("Criteria for true", bundle: .module)
                    }
                    TextField(text: $question.falseCriteria, prompt: Text(verbatim: "false")) {
                        Text("Criteria for false", bundle: .module)
                    }
                }
            case .choice:
                Text("Options", bundle: .module).themedFont(.small, weight: .medium)
                ForEach($question.options) { $option in
                    HStack {
                        TextField(text: $option.key, prompt: Text("Key", bundle: .module)) {
                            Text("Option key", bundle: .module)
                        }
                        .disabled(option.key == "__none__")
                        .frame(minWidth: 100, idealWidth: 135, maxWidth: 160)
                        TextField(text: $option.detail, prompt: Text("Description", bundle: .module)) {
                            Text("Option description", bundle: .module)
                        }
                        Button(role: .destructive) {
                            question.options.removeAll { $0.id == option.id }
                        } label: { Text("Remove", bundle: .module) }
                        .appAction(.destructive, size: APIControlSize.row)
                    }
                }
                HStack {
                    Button {
                        question.options.append(.init(key: "option_\(question.options.count + 1)", detail: ""))
                    } label: { Text("Add", bundle: .module) }
                    .appAction(.secondary, size: APIControlSize.row)
                    if !question.options.contains(where: { $0.key == "__none__" }) {
                        // The detail is request content sent to the model,
                        // so it stays in English with the schema keyword.
                        Button {
                            question.options.append(.init(key: "__none__", detail: "None of the listed options applies."))
                        } label: { Text(verbatim: "+ __none__") }
                        .appAction(.secondary, size: APIControlSize.row)
                    }
                }
            case .score:
                ForEach(question.levels.indices, id: \.self) { index in
                    HStack {
                        Text(verbatim: "\(index)").frame(width: 25)
                        // A bounds-checked binding: ForEach over indices keeps
                        // this row alive for one pass after Remove shrinks the
                        // array, and a direct `$question.levels[index]` then
                        // traps with "Index out of range".
                        TextField(text: Binding(
                            get: { question.levels.indices.contains(index) ? question.levels[index] : "" },
                            set: { if question.levels.indices.contains(index) { question.levels[index] = $0 } }),
                            prompt: Text("Level", bundle: .module)
                        ) {
                            Text("Level \(index)", bundle: .module)
                        }
                        Button(role: .destructive) {
                            if question.levels.count > 2 { question.levels.remove(at: index) }
                        } label: { Text("Remove", bundle: .module) }
                        .appAction(.destructive, size: APIControlSize.row)
                        .disabled(question.levels.count <= 2)
                    }
                }
                Button { question.levels.append("") } label: { Text("Add", bundle: .module) }
                    .appAction(.secondary, size: APIControlSize.row)
            }
        }
        .padding(10)
        .background(.appPage, in: RoundedRectangle(cornerRadius: 8))
        .onChange(of: question.kind) { _, kind in
            let defaults = TypeSafeQuestionDraft.new(kind, index: 1)
            if kind == .choice && question.options.isEmpty { question.options = defaults.options }
            if kind == .score && question.levels.count < 2 { question.levels = defaults.levels }
        }
    }
}

struct TypeSafeAnswerCard: View {
    let name: String
    let answer: Answer
    let question: Question?

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text(verbatim: name).themedFont(.callout, weight: .semibold)
                Spacer()
                Text(verbatim: kind).themedCode(.small).foregroundStyle(.appSecondary)
            }
            switch answer {
            case .noul(let value):
                Text(verbatim: String(format: "%.3f", value)).themedFont(.title2, weight: .semibold)
                ProgressView(value: value, total: 1)
                    .accessibilityLabel(Text("Probability of true", bundle: .module))
                if case .noul(_, let criteria) = question, let criteria {
                    Text(verbatim: "true: \(criteria.true)  false: \(criteria.false)")
                        .themedFont(.small).foregroundStyle(.appSecondary)
                }
            case .choice(let selected, let probabilities, let confidence):
                Group {
                    if selected == "__none__" {
                        Text("None of the listed options", bundle: .module)
                    } else {
                        Text(verbatim: selected)
                    }
                }
                .themedFont(.title3, weight: .semibold)
                ForEach(probabilities.sorted(by: { $0.value > $1.value }), id: \.key) { key, value in
                    probabilityRow(label: key, value: value, selected: key == selected)
                }
                Text("Confidence \(String(format: "%.3f", confidence))", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            case .score(let value, let legend, let probabilities, let confidence):
                Text(
                    "Expected level \(String(format: "%.2f", value)), confidence \(String(format: "%.3f", confidence))",
                    bundle: .module)
                    .themedFont(.callout, weight: .semibold)
                ForEach(probabilities.sorted(by: { (Int($0.key) ?? 0) < (Int($1.key) ?? 0) }), id: \.key) { key, probability in
                    probabilityRow(label: "\(key)  \(legend[key] ?? "")", value: probability, selected: false)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appPage, in: RoundedRectangle(cornerRadius: 8))
    }

    private var kind: String {
        switch answer {
        case .noul: return "noul"
        case .choice: return "choice"
        case .score: return "score"
        }
    }

    private func probabilityRow(label: String, value: Double, selected: Bool) -> some View {
        HStack(spacing: 8) {
            Text(verbatim: label).frame(width: 145, alignment: .leading)
            ProgressView(value: value, total: 1)
            Text(verbatim: String(format: "%.1f%%", value * 100)).frame(width: 55, alignment: .trailing)
            if selected { Text("Selected", bundle: .module).themedFont(.tiny) }
        }
        .accessibilityElement(children: .combine)
    }
}
