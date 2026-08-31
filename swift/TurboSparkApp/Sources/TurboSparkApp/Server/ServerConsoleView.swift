import AppKit
import SwiftUI
import TurboSpark

/// The request log.
///
/// **ONE ROW PER REQUEST, NOT ONE PER EVENT.** Four events describe a
/// request and a reader wants the request. The raw events are still there
/// underneath -- `AppModel.serverEventLog` -- and are what an expanded row
/// shows, but the default reading is the thing that happened.
struct ServerConsoleView: View {
    @ObservedObject var model: AppModel

    @State private var search = ""
    @State private var showErrorsOnly = false
    @State private var paused = false
    @State private var expanded: Set<UInt64> = []
    @State private var modelFilter: String?

    private var rows: [ServerRequestRecord] {
        var records = model.serverMetrics.records.reversed().map { $0 }
        if showErrorsOnly {
            records = records.filter(\.isError)
        }
        if let modelFilter {
            records = records.filter { $0.servedModel == modelFilter }
        }
        let query = search.trimmingCharacters(in: .whitespaces).lowercased()
        if !query.isEmpty {
            records = records.filter {
                $0.path.lowercased().contains(query)
                    || ($0.servedModel ?? "").lowercased().contains(query)
                    || ($0.requestedModel ?? "").lowercased().contains(query)
                    || "\($0.status ?? 0)".contains(query)
            }
        }
        return records
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            if rows.isEmpty {
                Text(model.serverMetrics.records.isEmpty ? "No traffic yet." : "Nothing matches.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(rows) { record in
                            row(record)
                            Divider()
                        }
                    }
                }
            }
        }
        .background(Color(nsColor: .textBackgroundColor).opacity(0.4))
    }

    private var toolbar: some View {
        HStack(spacing: 10) {
            Text("Console")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)

            TextField("Filter", text: $search)
                .textFieldStyle(.roundedBorder)
                .controlSize(.small)
                .frame(width: 160)
                // macOS draws a TextField's title as a VISIBLE LABEL rather
                // than a placeholder, and an unhidden one gets clipped in a
                // toolbar this narrow (`swift/CLAUDE.md` Gotcha 23's smaller
                // sibling).
                .labelsHidden()

            Toggle("Errors only", isOn: $showErrorsOnly)
                .toggleStyle(.checkbox)
                .font(.system(size: 11))

            if model.serverMetrics.servingModels.count > 1 {
                Picker("Model", selection: $modelFilter) {
                    Text("All models").tag(String?.none)
                    ForEach(model.serverMetrics.servingModels, id: \.self) { name in
                        Text(name).tag(String?.some(name))
                    }
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .labelsHidden()
                .frame(width: 160)
            }

            Spacer()

            if model.serverMetrics.droppedEvents > 0 {
                Label("\(model.serverMetrics.droppedEvents) events dropped", systemImage: "exclamationmark.triangle")
                    .font(.system(size: 10))
                    .foregroundStyle(.orange)
                    .help(
                        "The engine's buffer overran while this pane was not polling. "
                            + "Those rows are gone rather than delayed.")
            }

            Button {
                paused.toggle()
                paused ? model.stopServerPolling() : model.startServerPolling()
            } label: {
                Image(systemName: paused ? "play.fill" : "pause.fill")
            }
            .buttonStyle(.borderless)
            .help(paused ? "Resume" : "Pause. The engine keeps buffering while paused.")

            Button {
                copyAll()
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .buttonStyle(.borderless)
            .help("Copy the visible rows as JSON")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private func row(_ record: ServerRequestRecord) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text(statusText(record))
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(statusColor(record))
                    .frame(width: 34, alignment: .leading)

                Text(record.method)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 38, alignment: .leading)

                Text(record.path)
                    .font(.system(size: 11, design: .monospaced))
                    .lineLimit(1)

                Spacer(minLength: 8)

                if let served = record.servedModel {
                    Text(served)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                if record.stream {
                    Image(systemName: "dot.radiowaves.right")
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                        .help("Streamed")
                }
                if let tokens = record.newTokens, tokens > 0 {
                    Text("\(tokens) tok")
                        .font(.system(size: 10))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                if let duration = record.durationMs {
                    Text("\(duration) ms")
                        .font(.system(size: 10))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .frame(width: 62, alignment: .trailing)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
            .contentShape(Rectangle())
            .onTapGesture {
                if expanded.contains(record.id) {
                    expanded.remove(record.id)
                } else {
                    expanded.insert(record.id)
                }
            }

            if expanded.contains(record.id) {
                detail(record)
            }
        }
    }

    private func detail(_ record: ServerRequestRecord) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            // The pair that matters on a multi-model server: what the client
            // asked for against what answered. They differ on every
            // single-model fallback, which is the common case.
            if let requested = record.requestedModel, requested != record.servedModel {
                detailLine("asked for", requested)
            }
            if let served = record.servedModel { detailLine("served by", served) }
            if let prompt = record.promptTokens { detailLine("prompt", "\(prompt) tokens") }
            if let tokens = record.newTokens { detailLine("generated", "\(tokens) tokens") }
            if let prefill = record.prefillSeconds {
                detailLine("prefill", String(format: "%.3f s", prefill))
            }
            if let decode = record.decodeSeconds {
                detailLine("decode", String(format: "%.3f s", decode))
            }
            if let rate = record.tokensPerSecond {
                detailLine("rate", String(format: "%.1f tok/s", rate))
            }
            if let queued = record.queuedSeconds {
                detailLine(
                    "queued", String(format: "%.3f s", queued),
                    help: "Waiting for the runner. One turn at a time per model.")
            }
            if let reason = record.stopReason { detailLine("stopped", reason) }
            if record.generations > 1 {
                detailLine(
                    "generations", "\(record.generations)",
                    help: "The tool-call guardrails re-asked. Both turns really ran.")
            }
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
        .padding(.leading, 34)
    }

    private func detailLine(_ label: String, _ value: String, help: String? = nil) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
                .frame(width: 78, alignment: .leading)
            Text(value)
                .font(.system(size: 10, design: .monospaced))
                .textSelection(.enabled)
            if let help {
                Image(systemName: "questionmark.circle")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                    .help(help)
            }
        }
    }

    private func statusText(_ record: ServerRequestRecord) -> String {
        record.status.map(String.init) ?? "..."
    }

    private func statusColor(_ record: ServerRequestRecord) -> Color {
        guard let status = record.status else { return .secondary }
        if status >= 500 { return .red }
        if status >= 400 { return .orange }
        return .green
    }

    private func copyAll() {
        let payload = rows.map { record -> [String: Any] in
            var object: [String: Any] = [
                "id": record.id, "method": record.method, "path": record.path,
            ]
            object["status"] = record.status.map { Int($0) }
            object["durationMs"] = record.durationMs.map { Int($0) }
            object["requestedModel"] = record.requestedModel
            object["servedModel"] = record.servedModel
            object["promptTokens"] = record.promptTokens.map { Int($0) }
            object["newTokens"] = record.newTokens.map { Int($0) }
            object["prefillSeconds"] = record.prefillSeconds
            object["decodeSeconds"] = record.decodeSeconds
            object["stopReason"] = record.stopReason
            return object.compactMapValues { $0 }
        }
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(String(decoding: data, as: UTF8.self), forType: .string)
    }
}
