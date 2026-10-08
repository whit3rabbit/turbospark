import AppKit
import OpenKind
import SwiftUI

@MainActor
struct TypeSafePlaygroundView: View {
    @ObservedObject var model: AppModel
    @State private var draft = TypeSafePlaygroundDraft()
    @State private var rawJSON = ""
    @State private var mode: EditorMode = .form
    @State private var page: Page = .playground
    @State private var response: SystemResponse?
    @State private var responseJSON = ""
    @State private var lastRequest: SystemRequest?
    @State private var requestID: String?
    @State private var latencyMS: Double?
    @State private var errorMessage: String?
    @State private var isRunning = false
    @State private var runTask: Task<Void, Never>?

    private enum EditorMode: String, CaseIterable { case form, raw }
    private enum Page: String, CaseIterable { case playground, benchmark }

    private var loadedModels: [String] {
        model.typeSafeModels.filter(\.loaded).map(\.name).sorted()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("TypeSafe", bundle: .module).themedFont(.title2, weight: .semibold)
                Spacer()
                Picker(selection: $page) {
                    Text("Playground", bundle: .module).tag(Page.playground)
                    Text("Benchmark", bundle: .module).tag(Page.benchmark)
                } label: { Text("TypeSafe", bundle: .module) }
                .pickerStyle(.segmented)
                .frame(width: 260)
                .disabled(isRunning)
            }
            if page == .playground {
                playgroundContent
            } else {
                TypeSafeBenchmarkView(
                    service: model.typeSafeServer,
                    loadedModels: loadedModels,
                    request: currentRequest)
            }
        }
        .onAppear { chooseLoadedModel() }
        .onChange(of: loadedModels) { _, _ in chooseLoadedModel() }
        .onChange(of: model.typeSafeServer != nil) { _, running in
            if !running { runTask?.cancel() }
        }
        .onDisappear { runTask?.cancel() }
    }

    private var playgroundContent: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 14) {
                examplePanel.frame(width: 180)
                requestPanel.frame(maxWidth: .infinity)
            }
            outputPanel
        }
    }

    private var examplePanel: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text("Examples", bundle: .module).themedFont(.callout, weight: .semibold)
            ForEach(TypeSafePlaygroundPreset.allCases) { preset in
                Button {
                    draft = preset.draft(model: draft.model)
                    mode = .form
                    clearResult()
                } label: {
                    Text(verbatim: preset.rawValue)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.bordered)
                .accessibilityHint(Text("Request", bundle: .module))
            }
        }
        .padding(14)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
    }

    private var requestPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Request", bundle: .module).themedFont(.callout, weight: .semibold)
                Spacer()
                Picker(selection: Binding(
                    get: { mode },
                    set: { changeMode(to: $0) }
                )) {
                    Text("Form", bundle: .module).tag(EditorMode.form)
                    Text("Raw JSON", bundle: .module).tag(EditorMode.raw)
                } label: { Text("Request", bundle: .module) }
                .pickerStyle(.segmented)
                .frame(width: 210)
            }

            if mode == .form { formEditor } else { rawEditor }

            if let errorMessage {
                Text(errorMessage).foregroundStyle(.red).textSelection(.enabled)
                    .accessibilityAddTraits(.isStaticText)
            }
            HStack {
                Button {
                    runTask = Task { await runRequest() }
                } label: { Text("Run", bundle: .module) }
                .buttonStyle(.borderedProminent)
                .disabled(isRunning || model.typeSafeServer == nil)
                if isRunning {
                    Button { runTask?.cancel() } label: { Text("Stop", bundle: .module) }
                    ProgressView().controlSize(.small)
                }
                Spacer()
                Button { copyRequest() } label: { Text("Copy", bundle: .module) }
            }
        }
        .padding(14)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
    }

    private var formEditor: some View {
        VStack(alignment: .leading, spacing: 10) {
            if loadedModels.isEmpty {
                Text("Select a model to load", bundle: .module)
                    .foregroundStyle(.secondary)
            } else {
                Picker(selection: $draft.model) {
                    ForEach(loadedModels, id: \.self) { name in Text(name).tag(name) }
                } label: { Text("Model", bundle: .module) }
            }

            HStack {
                Text("State", bundle: .module).themedFont(.small, weight: .medium)
                Spacer()
                Picker(selection: $draft.stateMode) {
                    Text("Text", bundle: .module).tag(TypeSafeStateMode.text)
                    Text(verbatim: "JSON").tag(TypeSafeStateMode.json)
                } label: { Text("State", bundle: .module) }
                .pickerStyle(.segmented)
                .frame(width: 155)
            }
            TextEditor(text: $draft.state)
                .themedCode(.small)
                .frame(minHeight: 95)
                .border(.appBorder)
                .accessibilityLabel(Text("State", bundle: .module))

            HStack {
                Text("Questions", bundle: .module).themedFont(.small, weight: .medium)
                Spacer()
                Button {
                    draft.questions.append(.new(.noul, index: draft.questions.count + 1))
                } label: { Text("Add", bundle: .module) }
            }
            ForEach($draft.questions) { $question in
                TypeSafeQuestionCard(question: $question) {
                    draft.questions.removeAll { $0.id == question.id }
                }
            }
        }
    }

    private var rawEditor: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextEditor(text: $rawJSON)
                .themedCode(.small)
                .frame(minHeight: 310)
                .border(.appBorder)
                .accessibilityLabel(Text("Raw JSON", bundle: .module))
            HStack {
                Button {
                    do {
                        let value = try JSONDecoder().decode(JSONValue.self, from: Data(rawJSON.utf8))
                        rawJSON = try TypeSafePlaygroundDraft.pretty(value)
                        errorMessage = nil
                    } catch { errorMessage = error.localizedDescription }
                } label: { Text("Format", bundle: .module) }
                Button { copy(rawJSON) } label: { Text("Copy", bundle: .module) }
            }
        }
    }

    private var outputPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Output", bundle: .module).themedFont(.callout, weight: .semibold)
                Spacer()
                if !responseJSON.isEmpty {
                    Button { copy(responseJSON) } label: { Text("Copy", bundle: .module) }
                }
            }
            if let latencyMS {
                HStack(spacing: 14) {
                    Text(verbatim: String(format: "%.1f ms", latencyMS))
                    if let response { Text(verbatim: response.model) }
                    if let requestID { Text(verbatim: "Request ID: \(requestID)").textSelection(.enabled) }
                }
                .themedFont(.small)
                .foregroundStyle(.secondary)
            }
            if let response {
                let names = response.answers.keys.sorted()
                Text(verbatim: "Input \(response.usage.input_tokens), output \(response.usage.output_tokens) tokens")
                    .themedFont(.small).foregroundStyle(.secondary)
                ForEach(names, id: \.self) { name in
                    if let answer = response.answers[name] {
                        TypeSafeAnswerCard(name: name, answer: answer, question: lastRequest?.questions[name])
                    }
                }
                DisclosureGroup {
                    Text(responseJSON).themedCode(.small).textSelection(.enabled)
                } label: { Text("Raw JSON", bundle: .module) }
            } else if let errorMessage {
                Text(errorMessage).foregroundStyle(.red).textSelection(.enabled)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
    }

    private func chooseLoadedModel() {
        if !loadedModels.contains(draft.model), let first = loadedModels.first { draft.model = first }
    }

    private func changeMode(to next: EditorMode) {
        guard next != mode else { return }
        do {
            if next == .raw {
                rawJSON = try TypeSafePlaygroundDraft.pretty(draft.request())
            } else {
                let request = try JSONDecoder().decode(SystemRequest.self, from: Data(rawJSON.utf8))
                draft = try TypeSafePlaygroundDraft.from(request)
            }
            errorMessage = nil
            mode = next
        } catch { errorMessage = error.localizedDescription }
    }

    private func currentRequest() throws -> SystemRequest {
        if mode == .raw {
            return try JSONDecoder().decode(SystemRequest.self, from: Data(rawJSON.utf8))
        }
        return try draft.request()
    }

    private func runRequest() async {
        guard let service = model.typeSafeServer else { return }
        isRunning = true
        clearResult()
        let started = ProcessInfo.processInfo.systemUptime
        defer {
            isRunning = false
            runTask = nil
            latencyMS = (ProcessInfo.processInfo.systemUptime - started) * 1_000
        }
        do {
            let request = try currentRequest()
            let result = try await service.client.evaluate(request)
            try Task.checkCancellation()
            response = result.data
            responseJSON = try TypeSafePlaygroundDraft.pretty(result.data)
            requestID = result.requestID
            lastRequest = request
        } catch is CancellationError {
            errorMessage = "Stopped waiting for the response."
        } catch let error as ApiError {
            requestID = error.requestID
            errorMessage = "HTTP \(error.status): \(error.message)"
        } catch { errorMessage = error.localizedDescription }
    }

    private func clearResult() {
        response = nil
        responseJSON = ""
        lastRequest = nil
        requestID = nil
        latencyMS = nil
        errorMessage = nil
    }

    private func copyRequest() {
        do { copy(try TypeSafePlaygroundDraft.pretty(currentRequest())) }
        catch { errorMessage = error.localizedDescription }
    }

    private func copy(_ value: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
    }
}

private struct TypeSafeQuestionCard: View {
    @Binding var question: TypeSafeQuestionDraft
    let remove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                TextField("Question ID", text: $question.name)
                    .accessibilityLabel(Text("Questions", bundle: .module))
                Picker(selection: $question.kind) {
                    ForEach(TypeSafeQuestionKind.allCases, id: \.self) { kind in
                        Text(verbatim: kind.rawValue).tag(kind)
                    }
                } label: { Text("Questions", bundle: .module) }
                .frame(width: 135)
                Button(action: remove) { Text("Remove", bundle: .module) }
            }
            TextField("Instructions", text: $question.instructions)
                .accessibilityLabel(Text("Instructions", bundle: .module))
            switch question.kind {
            case .noul:
                HStack {
                    TextField("true", text: $question.trueCriteria)
                    TextField("false", text: $question.falseCriteria)
                }
            case .choice:
                Text("Options", bundle: .module).themedFont(.small, weight: .medium)
                ForEach($question.options) { $option in
                    HStack {
                        TextField("key", text: $option.key)
                            .disabled(option.key == "__none__")
                            .frame(width: 135)
                        TextField("description", text: $option.detail)
                        Button {
                            question.options.removeAll { $0.id == option.id }
                        } label: { Text("Remove", bundle: .module) }
                    }
                }
                HStack {
                    Button {
                        question.options.append(.init(key: "option_\(question.options.count + 1)", detail: ""))
                    } label: { Text("Add", bundle: .module) }
                    if !question.options.contains(where: { $0.key == "__none__" }) {
                        Button {
                            question.options.append(.init(key: "__none__", detail: "None of the listed options applies."))
                        } label: { Text(verbatim: "+ __none__") }
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
                        TextField("level", text: Binding(
                            get: { question.levels.indices.contains(index) ? question.levels[index] : "" },
                            set: { if question.levels.indices.contains(index) { question.levels[index] = $0 } }))
                        Button {
                            if question.levels.count > 2 { question.levels.remove(at: index) }
                        } label: { Text("Remove", bundle: .module) }
                        .disabled(question.levels.count <= 2)
                    }
                }
                Button { question.levels.append("") } label: { Text("Add", bundle: .module) }
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

private struct TypeSafeAnswerCard: View {
    let name: String
    let answer: Answer
    let question: Question?

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text(name).themedFont(.callout, weight: .semibold)
                Spacer()
                Text(verbatim: kind).themedCode(.small).foregroundStyle(.secondary)
            }
            switch answer {
            case .noul(let value):
                Text(verbatim: String(format: "%.3f", value)).themedFont(.title2, weight: .semibold)
                ProgressView(value: value, total: 1)
                    .accessibilityLabel(Text(verbatim: "Probability of true"))
                if case .noul(_, let criteria) = question, let criteria {
                    Text(verbatim: "true: \(criteria.true)  false: \(criteria.false)")
                        .themedFont(.small).foregroundStyle(.secondary)
                }
            case .choice(let selected, let probabilities, let confidence):
                Text(verbatim: selected == "__none__" ? "None of the listed options" : selected)
                    .themedFont(.title3, weight: .semibold)
                ForEach(probabilities.sorted(by: { $0.value > $1.value }), id: \.key) { key, value in
                    probabilityRow(label: key, value: value, selected: key == selected)
                }
                Text(verbatim: String(format: "confidence %.3f", confidence))
                    .themedFont(.small).foregroundStyle(.secondary)
            case .score(let value, let legend, let probabilities, let confidence):
                Text(verbatim: String(format: "expected level %.2f, confidence %.3f", value, confidence))
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
