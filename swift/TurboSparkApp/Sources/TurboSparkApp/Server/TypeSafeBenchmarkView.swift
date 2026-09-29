import OpenKind
import SwiftUI

@MainActor
struct TypeSafeBenchmarkView: View {
    let service: OpenKindServer?
    let loadedModels: [String]
    let request: () throws -> SystemRequest

    @State private var selectedModels: Set<String> = []
    @State private var repetitions = 10
    @State private var completed = 0
    @State private var total = 0
    @State private var activeModel: String?
    @State private var results: [String: TypeSafeLatencySummary] = [:]
    @State private var errorMessage: String?
    @State private var task: Task<Void, Never>?
    @State private var stopRequested = false

    private var isRunning: Bool { task != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Benchmark", bundle: .module).themedFont(.callout, weight: .semibold)
            HStack {
                Stepper(value: $repetitions, in: 1...200) {
                    HStack {
                        Text("Repetitions", bundle: .module)
                        Text(verbatim: "\(repetitions)")
                    }
                }
                .frame(width: 230)
                Spacer()
                if isRunning {
                    Button { stopRequested = true } label: { Text("Stop", bundle: .module) }
                        .disabled(stopRequested)
                } else {
                    Button { task = Task { await run() } } label: { Text("Run", bundle: .module) }
                        .buttonStyle(.borderedProminent)
                        .disabled(service == nil || selectedModels.isEmpty)
                }
            }
            if loadedModels.isEmpty {
                Text("Select a model to load", bundle: .module).foregroundStyle(.secondary)
            }
            ForEach(loadedModels, id: \.self) { name in
                Toggle(isOn: Binding(
                    get: { selectedModels.contains(name) },
                    set: { selected in
                        if selected { selectedModels.insert(name) }
                        else { selectedModels.remove(name) }
                    }
                )) { Text(name) }
            }
            if isRunning {
                ProgressView(value: Double(completed), total: Double(max(total, 1))) {
                    if let activeModel {
                        Text(verbatim: "\(activeModel): \(completed) / \(total)")
                    }
                }
            }
            if let errorMessage {
                Text(errorMessage).foregroundStyle(.red).textSelection(.enabled)
            }
            ForEach(results.keys.sorted(), id: \.self) { name in
                if let result = results[name] {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(name).themedFont(.callout, weight: .semibold)
                        Text(verbatim: String(format: "n=%d  mean %.1f ms  p50 %.1f ms  p95 %.1f ms  min %.1f ms  max %.1f ms",
                                              result.count, result.mean, result.p50, result.p95,
                                              result.minimum, result.maximum))
                            .themedCode(.small).textSelection(.enabled)
                    }
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
        .onAppear { selectedModels.formUnion(loadedModels) }
        .onChange(of: loadedModels) { _, names in
            selectedModels.formIntersection(names)
            if selectedModels.isEmpty { selectedModels.formUnion(names) }
        }
        .onChange(of: service != nil) { _, running in
            if !running { task?.cancel() }
        }
        .onDisappear { task?.cancel() }
    }

    private func run() async {
        defer {
            activeModel = nil
            task = nil
            stopRequested = false
        }
        guard let service else { return }
        completed = 0
        total = repetitions * selectedModels.count
        results = [:]
        errorMessage = nil
        do {
            let base = try request()
            for name in selectedModels.sorted() {
                if Task.isCancelled || stopRequested { break }
                activeModel = name
                let test = SystemRequest(state: base.state, model: name, questions: base.questions)
                _ = try await service.client.evaluate(test)
                var samples: [Double] = []
                for _ in 0..<repetitions {
                    if Task.isCancelled || stopRequested { break }
                    let started = ProcessInfo.processInfo.systemUptime
                    _ = try await service.client.evaluate(test)
                    samples.append((ProcessInfo.processInfo.systemUptime - started) * 1_000)
                    completed += 1
                }
                results[name] = TypeSafeLatencySummary(samples)
            }
        } catch is CancellationError {
            return
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
