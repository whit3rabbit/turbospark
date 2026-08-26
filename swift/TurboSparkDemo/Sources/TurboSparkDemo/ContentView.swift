import SwiftUI
import TurboSpark

/// Main application window view for the TurboSpark demo chat app.
struct ContentView: View {
    @StateObject private var model = ChatModel()
    @State private var showingInstall = false

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            transcript
            Divider()
            composer
            Divider()
            StatusBar(model: model)
        }
        .frame(minWidth: 720, minHeight: 520)
        .task { model.refreshModels() }
        .sheet(isPresented: $showingInstall) {
            InstallSheet(catalog: model.catalog) { model.refreshModels() }
        }
        .alert(
            "Something went wrong",
            isPresented: Binding(get: { model.error != nil }, set: { if !$0 { model.error = nil } })
        ) {
            Button("OK") { model.error = nil }
        } message: {
            Text(model.error ?? "")
        }
    }

    private var toolbar: some View {
        HStack(spacing: 12) {
            Picker("Model", selection: Binding(
                get: { model.selected?.alias ?? "" },
                set: { alias in
                    guard let m = model.installed.first(where: { $0.alias == alias }) else { return }
                    Task { await model.open(m) }
                }
            )) {
                if model.installed.isEmpty {
                    Text("no models installed").tag("")
                }
                ForEach(model.installed) { m in
                    Text("\(m.alias)  (\(m.family))").tag(m.alias)
                }
            }
            .frame(width: 280)
            .disabled(model.opening || model.generating)

            if model.opening {
                ProgressView().controlSize(.small)
                Text("opening").foregroundStyle(.secondary)
            }

            Spacer()

            Picker("Thinking", selection: $model.reasoning) {
                ForEach(GenerateOptions.Reasoning.allCases, id: \.self) { level in
                    Text(level.rawValue).tag(level)
                }
            }
            .frame(width: 170)
            // Disabled rather than hidden when the checkpoint's template
            // cannot express a level: the control's absence would read as a
            // missing feature rather than a property of the model.
            .disabled(!model.reasoningAvailable || model.generating)
            .help(
                model.reasoningAvailable
                    ? "Levels a checkpoint rejects are reported, not silently dropped."
                    : "This checkpoint ships no chat template, so it cannot be asked to think.")

            Button("Install...") { showingInstall = true }
            Button("Clear") { model.clear() }.disabled(model.turns.isEmpty)
        }
        .padding(10)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(model.turns) { turn in
                        TurnView(turn: turn).id(turn.id)
                    }
                    if let p = model.prefill {
                        HStack(spacing: 8) {
                            ProgressView(value: Double(p.done), total: Double(max(p.total, 1)))
                                .frame(width: 160)
                            Text("reading prompt, \(p.done) of \(p.total) tokens")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        .id("prefill")
                    }
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onChange(of: model.turns.last?.content) { _ in
                withAnimation { proxy.scrollTo(model.turns.last?.id, anchor: .bottom) }
                model.updateTokenEstimate()
            }
            .onChange(of: model.draft) { _ in
                model.updateTokenEstimate()
            }
        }
    }

    private var composer: some View {
        HStack(spacing: 10) {
            TextField("Message", text: $model.draft, axis: .vertical)
                .lineLimit(1...5)
                .textFieldStyle(.roundedBorder)
                .onSubmit { model.send() }
                .disabled(model.session == nil || model.generating)

            if model.generating {
                // The button this whole binding is shaped around. It reaches
                // the engine immediately rather than waiting for the turn.
                Button("Stop", role: .destructive) { model.stop() }
                    .keyboardShortcut(".", modifiers: .command)
            } else {
                Button("Send") { model.send() }
                    .keyboardShortcut(.return, modifiers: [])
                    .disabled(model.session == nil || model.draft.isEmpty)
            }
        }
        .padding(10)
    }
}

/// Renders a single conversation turn with optional reasoning disclosure.
private struct TurnView: View {
    let turn: ChatModel.Turn

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(turn.role == .user ? "You" : "Assistant")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            if !turn.reasoning.isEmpty {
                DisclosureGroup("Thinking") {
                    Text(turn.reasoning)
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .font(.caption)
            }

            Text(turn.content)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Only worth showing when the turn did NOT simply finish, which
            // is the case a reader needs an explanation for.
            if let reason = turn.stopReason, reason != .endOfTurn {
                Text(explain(reason))
                    .font(.caption)
                    .foregroundStyle(reason == .cancelled ? .orange : .secondary)
            }
        }
    }

    private func explain(_ reason: GenerationResult.StopReason) -> String {
        switch reason {
        case .cancelled: return "stopped by you"
        case .maxTokens: return "hit the token budget"
        case .stopString: return "hit a stop string"
        case .eos: return "end of sequence"
        case .toolCalls: return "the model called a tool"
        case .endOfTurn: return ""
        }
    }
}

/// Status footer displaying active context, cache slots, decode speed, and memory footprint.
private struct StatusBar: View {
    @ObservedObject var model: ChatModel

    var body: some View {
        HStack(spacing: 16) {
            if let info = model.info {
                // The RESOLVED values, which under automatic sizing are the
                // only ones that exist. Neither a throughput nor a footprint
                // figure is readable without the slot count beside it.
                label("context", "\(info.maxContext)")
                if info.expertCacheSlots > 0 {
                    label("slots", "\(info.expertCacheSlots)")
                }
                if info.pastTrainedContext {
                    Text("past trained context")
                        .font(.caption).foregroundStyle(.orange)
                        .help("The model runs, but quality degrades past this point.")
                }
                // Shown even when OFF, and carrying the engine's own reason.
                // An install that carries a drafter and decodes one token at
                // a time with nothing said is the failure the feature exists
                // to end -- and this app samples, so the honest label for a
                // drafter that resolved is "greedy only".
                if let block = info.speculation.block {
                    label(
                        "speculative",
                        "\(info.speculation.drafter?.rawValue ?? "on") x\(block), greedy only"
                    )
                    .help("Acceptance is exact only at temperature 0, so a sampled turn "
                        + "decodes sequentially.")
                } else if let reason = info.speculation.reason {
                    label("speculative", "off").help(reason)
                }
                if info.steering.active {
                    label("steering", info.steering.mode ?? "on")
                        .help(info.steering.summary ?? "Directional steering is active.")
                }
                if model.estimatedPromptTokens > 0 {
                    label("draft", "\(model.estimatedPromptTokens) tok")
                }
            } else {
                Text("no model open").font(.caption).foregroundStyle(.secondary)
            }

            Spacer()

            if let t = model.telemetry {
                label("ram", humanBytes(t.physicalMemoryBytes))
                if t.thermalLevel != "nominal" {
                    label("thermal", t.thermalLevel).foregroundStyle(.orange)
                }
            }

            if let r = model.lastResult {
                if let rate = r.tokensPerSecond {
                    label("decode", String(format: "%.1f tok/s", rate))
                }
                label("tokens", "\(r.promptTokens) in / \(r.newTokens) out")
            }
            if let peak = model.peakFootprint {
                label("peak", humanBytes(peak))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(.quaternary.opacity(0.4))
    }

    /// Helper view displaying a labelled key-value metric.
    private func label(_ name: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(name).foregroundStyle(.secondary)
            Text(value).monospacedDigit()
        }
        .font(.caption)
    }
}
