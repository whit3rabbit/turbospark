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
            AppAdaptiveHeader {
                AppSectionHeader(
                    title: Text("Test TypeSafe request", bundle: .module),
                    subtitle: Text("Requests go to the TypeSafe service, not the TurboSpark server.", bundle: .module))
            } actions: {
                Picker(selection: $page) {
                    Text("Playground", bundle: .module).tag(Page.playground)
                    Text("Benchmark", bundle: .module).tag(Page.benchmark)
                } label: { Text("TypeSafe", bundle: .module) }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(minWidth: 200, idealWidth: 260, maxWidth: 320)
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
            // Side by side when there is room; stacked when a narrow window
            // or a large text size would squeeze the request editor.
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 14) {
                    examplePanel.frame(width: 180)
                    requestPanel.frame(minWidth: 380, maxWidth: .infinity)
                }
                VStack(alignment: .leading, spacing: 14) {
                    examplePanel
                    requestPanel
                }
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
                .appAction(.secondary, size: APIControlSize.row)
                .accessibilityHint(Text("Replaces the request with this example", bundle: .module))
            }
        }
        .apiCard()
    }

    private var requestPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            AppAdaptiveHeader {
                Text("Request", bundle: .module).themedFont(.callout, weight: .semibold)
            } actions: {
                Picker(selection: Binding(
                    get: { mode },
                    set: { changeMode(to: $0) }
                )) {
                    Text("Form", bundle: .module).tag(EditorMode.form)
                    Text("Raw JSON", bundle: .module).tag(EditorMode.raw)
                } label: { Text("Request", bundle: .module) }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(minWidth: 160, idealWidth: 210, maxWidth: 260)
            }

            if mode == .form { formEditor } else { rawEditor }

            if let errorMessage {
                Text(verbatim: errorMessage).foregroundStyle(.red).textSelection(.enabled)
                    .accessibilityAddTraits(.isStaticText)
            }
            HStack(spacing: AppMetrics.actionSpacing) {
                Button {
                    runTask = Task { await runRequest() }
                } label: { Text("Run", bundle: .module) }
                .appAction(.primary, size: APIControlSize.panel)
                .disabled(isRunning || model.typeSafeServer == nil)
                if isRunning {
                    Button { runTask?.cancel() } label: { Text("Stop", bundle: .module) }
                        .appAction(.secondary, size: APIControlSize.panel)
                }
                APIRunIndicator(running: isRunning, elapsed: nil)
                Spacer()
                Button { copyRequest() } label: { Text("Copy", bundle: .module) }
                    .appAction(.secondary, size: APIControlSize.panel)
            }
        }
        .apiCard()
    }

    private var formEditor: some View {
        VStack(alignment: .leading, spacing: 10) {
            if loadedModels.isEmpty {
                Text("Select a model to load", bundle: .module)
                    .foregroundStyle(.appSecondary)
            } else {
                Picker(selection: $draft.model) {
                    ForEach(loadedModels, id: \.self) { name in Text(verbatim: name).tag(name) }
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
                .labelsHidden()
                .frame(minWidth: 120, idealWidth: 155, maxWidth: 200)
            }
            TextEditor(text: $draft.state)
                .themedCode(.small)
                .frame(minHeight: 95)
                .appEditorSurface()
                .accessibilityLabel(Text("State", bundle: .module))

            HStack {
                Text("Questions", bundle: .module).themedFont(.small, weight: .medium)
                Spacer()
                Button {
                    draft.questions.append(.new(.noul, index: draft.questions.count + 1))
                } label: { Text("Add", bundle: .module) }
                .appAction(.secondary, size: APIControlSize.row)
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
                .appEditorSurface()
                .accessibilityLabel(Text("Raw JSON", bundle: .module))
            HStack(spacing: AppMetrics.actionSpacing) {
                Button {
                    do {
                        let value = try JSONDecoder().decode(JSONValue.self, from: Data(rawJSON.utf8))
                        rawJSON = try TypeSafePlaygroundDraft.pretty(value)
                        errorMessage = nil
                    } catch { errorMessage = error.localizedDescription }
                } label: { Text("Format", bundle: .module) }
                .appAction(.secondary, size: APIControlSize.panel)
                Button { copy(rawJSON) } label: { Text("Copy", bundle: .module) }
                    .appAction(.secondary, size: APIControlSize.panel)
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
                        .appAction(.secondary, size: APIControlSize.panel)
                }
            }
            if let latencyMS {
                HStack(spacing: 14) {
                    Text(verbatim: String(format: "%.1f ms", latencyMS))
                    if let response { Text(verbatim: response.model) }
                    if let requestID { Text("Request ID: \(requestID)", bundle: .module).textSelection(.enabled) }
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }
            if let response {
                let names = response.answers.keys.sorted()
                Text("Input \(String(response.usage.input_tokens)), output \(String(response.usage.output_tokens)) tokens", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
                ForEach(names, id: \.self) { name in
                    if let answer = response.answers[name] {
                        TypeSafeAnswerCard(name: name, answer: answer, question: lastRequest?.questions[name])
                    }
                }
                DisclosureGroup {
                    APICodeBlock(code: responseJSON)
                } label: { Text("Raw JSON", bundle: .module) }
            } else if let errorMessage {
                Text(verbatim: errorMessage).foregroundStyle(.red).textSelection(.enabled)
            }
        }
        .apiCard()
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
            errorMessage = AppStrings.localized("Stopped waiting for the response.")
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
