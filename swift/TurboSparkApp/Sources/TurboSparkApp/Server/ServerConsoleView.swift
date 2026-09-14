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
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @State private var search = ""
    @State private var showErrorsOnly = false
    @State private var paused = false
    @State private var expanded: Set<UInt64> = []
    @State private var modelFilter: String?
    @State private var copiedAll = false
    @State private var copiedRecordID: UInt64?

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
                    || ($0.errorMessage ?? "").lowercased().contains(query)
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
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
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
        .background(.appElevated.opacity(0.4))
    }

    private var toolbar: some View {
        HStack(spacing: 10) {
            Text("Console", bundle: .module)
                .themedFont(.tiny, weight: .semibold)
                .foregroundStyle(.appSecondary)

            TextField("Filter", text: $search)
                .textFieldStyle(.roundedBorder)
                .controlSize(.small)
                .frame(width: 160)
                .labelsHidden()

            Toggle(isOn: $showErrorsOnly) {
                Text("Errors only", bundle: .module)
            }
                .toggleStyle(.checkbox)
                .themedFont(.tiny)

            if model.serverMetrics.servingModels.count > 1 {
                Picker(selection: $modelFilter) {
                    Text("All models", bundle: .module).tag(String?.none)
                    ForEach(model.serverMetrics.servingModels, id: \.self) { name in
                        Text(name).tag(String?.some(name))
                    }
                } label: { Text("Model", bundle: .module) }
                .pickerStyle(.menu)
                .controlSize(.small)
                .labelsHidden()
                .frame(width: 160)
            }

            Spacer()

            if model.serverMetrics.droppedEvents > 0 {
                Label("\(model.serverMetrics.droppedEvents) events dropped", systemImage: "exclamationmark.triangle")
                    .themedFont(.tiny)
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
                Image(systemName: copiedAll ? "checkmark" : "doc.on.doc")
                    .foregroundStyle(copiedAll ? .green : .secondary)
            }
            .buttonStyle(.borderless)
            .help(copiedAll ? "Copied visible rows as JSON" : "Copy the visible rows as JSON")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private func row(_ record: ServerRequestRecord) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text(statusText(record))
                    .font(theme.code(.tiny, weight: .semibold))
                    .foregroundStyle(statusColor(record))
                    .frame(width: 34, alignment: .leading)

                Text(record.method)
                    .font(theme.code(.tiny))
                    .foregroundStyle(.appSecondary)
                    .frame(width: 38, alignment: .leading)

                Text(record.path)
                    .font(theme.code(.small))
                    .lineLimit(1)

                if record.isError, let err = record.errorMessage, !err.isEmpty {
                    Text(err)
                        .font(theme.code(.micro))
                        .foregroundStyle(.red.opacity(0.85))
                        .lineLimit(1)
                }

                Spacer(minLength: 8)

                if let served = record.servedModel {
                    Text(served)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                }
                if record.stream {
                    Image(systemName: "dot.radiowaves.right")
                        .themedFont(.micro)
                        .foregroundStyle(.appSecondary)
                        .help("Streamed")
                }
                if let tokens = record.newTokens, tokens > 0 {
                    Text(verbatim: "\(tokens) tok")
                        .themedFont(.tiny)
                        .monospacedDigit()
                        .foregroundStyle(.appSecondary)
                }
                if let duration = record.durationMs {
                    Text(verbatim: "\(duration) ms")
                        .themedFont(.tiny)
                        .monospacedDigit()
                        .foregroundStyle(.appSecondary)
                        .frame(width: 62, alignment: .trailing)
                }

                Button {
                    copyRowAction(record)
                } label: {
                    Image(systemName: copiedRecordID == record.id ? "checkmark" : "doc.on.doc")
                        .themedFont(.micro)
                        .foregroundStyle(copiedRecordID == record.id ? Color.green : Color.secondary)
                }
                .buttonStyle(.borderless)
                .help(record.isError ? "Copy error or request summary" : "Copy request summary")
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
            .contextMenu {
                if let err = record.errorMessage, !err.isEmpty {
                    Button {
                        copyText(err)
                        model.showToast("Copied error message", style: .info)
                    } label: { Text("Copy Error Message", bundle: .module) }
                }
                Button {
                    copyRecordSummary(record)
                    model.showToast("Copied request summary", style: .info)
                } label: { Text("Copy Request Log Line", bundle: .module) }
                Button {
                    copyRecordAsJSON(record)
                    model.showToast("Copied request as JSON", style: .info)
                } label: { Text("Copy Request as JSON", bundle: .module) }
                Button {
                    copyText(record.path)
                    model.showToast("Copied path", style: .info)
                } label: { Text("Copy Path", bundle: .module) }
            }

            if expanded.contains(record.id) {
                detail(record)
            }
        }
        .textSelection(.enabled)
    }

    private func detail(_ record: ServerRequestRecord) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            if record.isError {
                errorBox(record)
            }

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

            let matchingEvents = model.serverEventLog.filter { $0.requestID == record.id }
            if !matchingEvents.isEmpty {
                rawEventsBox(matchingEvents)
            }
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
        .padding(.leading, 34)
    }

    private func errorBox(_ record: ServerRequestRecord) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .imageScale(.small)
                Text("Error Details (HTTP \(statusText(record)))", bundle: .module)
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.red)

                Spacer()

                if let err = record.errorMessage, !err.isEmpty {
                    Button {
                        copyText(err)
                        model.showToast("Copied error message", style: .info)
                    } label: {
                        Image(systemName: "doc.on.doc")
                            .themedFont(.micro)
                    }
                    .buttonStyle(.borderless)
                    .help("Copy error message")
                }
            }

            if let err = record.errorMessage, !err.isEmpty {
                Text(err)
                    .font(theme.code(.tiny))
                    .foregroundStyle(.red.opacity(0.95))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(6)
                    .background(Color.red.opacity(0.08))
                    .cornerRadius(4)
            } else {
                Text("Request failed with HTTP status \(statusText(record)).", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
        }
        .padding(6)
        .background(Color.red.opacity(0.04))
        .cornerRadius(6)
    }

    private func rawEventsBox(_ events: [ServerEvent]) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text("Raw Events (\(events.count))", bundle: .module)
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                Spacer()
                Button {
                    copyEvents(events)
                    model.showToast("Copied \(events.count) events", style: .info)
                } label: {
                    Image(systemName: "doc.on.doc")
                        .themedFont(.micro)
                }
                .buttonStyle(.borderless)
                .help("Copy raw events")
            }
            ForEach(Array(events.enumerated()), id: \.offset) { _, ev in
                Text(eventDescription(ev))
                    .font(theme.code(.micro))
                    .foregroundStyle(.appSecondary)
                    .textSelection(.enabled)
            }
        }
        .padding(6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appSurface.opacity(0.4))
        .cornerRadius(4)
    }

    private func detailLine(_ label: String, _ value: String, help: String? = nil) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .frame(width: 78, alignment: .leading)
            Text(value)
                .font(theme.code(.tiny))
                .textSelection(.enabled)
            if let help {
                Image(systemName: "questionmark.circle")
                    .themedFont(.micro)
                    .foregroundStyle(.appSecondary)
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

    private func recordToDictionary(_ record: ServerRequestRecord) -> [String: Any] {
        var object: [String: Any] = [
            "id": record.id,
            "method": record.method,
            "path": record.path,
        ]
        object["status"] = record.status.map { Int($0) }
        object["durationMs"] = record.durationMs.map { Int($0) }
        object["requestedModel"] = record.requestedModel
        object["servedModel"] = record.servedModel
        object["error"] = record.errorMessage
        object["promptTokens"] = record.promptTokens.map { Int($0) }
        object["newTokens"] = record.newTokens.map { Int($0) }
        object["prefillSeconds"] = record.prefillSeconds
        object["decodeSeconds"] = record.decodeSeconds
        object["stopReason"] = record.stopReason
        return object.compactMapValues { $0 }
    }

    private func copyAll() {
        let payload = rows.map { recordToDictionary($0) }
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        else { return }
        copyText(String(decoding: data, as: UTF8.self))
        withAnimation {
            copiedAll = true
        }
        model.showToast("Copied \(rows.count) requests to clipboard", style: .info)
        Task {
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            withAnimation {
                copiedAll = false
            }
        }
    }

    private func copyRowAction(_ record: ServerRequestRecord) {
        if let err = record.errorMessage, !err.isEmpty {
            copyText(err)
            model.showToast("Copied error message", style: .info)
        } else {
            copyRecordSummary(record)
            model.showToast("Copied request summary", style: .info)
        }
        withAnimation {
            copiedRecordID = record.id
        }
        Task {
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            withAnimation {
                if copiedRecordID == record.id {
                    copiedRecordID = nil
                }
            }
        }
    }

    private func copyRecordAsJSON(_ record: ServerRequestRecord) {
        let dict = recordToDictionary(record)
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: dict, options: [.prettyPrinted, .sortedKeys])
        else { return }
        copyText(String(decoding: data, as: UTF8.self))
    }

    private func copyRecordSummary(_ record: ServerRequestRecord) {
        var summary = "[\(record.method) \(record.path)] Status: \(statusText(record))"
        if let duration = record.durationMs {
            summary += " (\(duration) ms)"
        }
        if let err = record.errorMessage, !err.isEmpty {
            summary += " - Error: \(err)"
        }
        copyText(summary)
    }

    private func copyEvents(_ events: [ServerEvent]) {
        let text = events.map(eventDescription).joined(separator: "\n")
        copyText(text)
    }

    private func eventDescription(_ event: ServerEvent) -> String {
        switch event {
        case let .requestStarted(id, atMs, method, path):
            return "[\(atMs)ms] #\(id) started: \(method) \(path)"
        case let .requestRouted(id, requested, served, stream):
            let asked = requested.map { " (asked \($0))" } ?? ""
            return "#\(id) routed: \(served)\(asked), stream=\(stream)"
        case let .generated(id, model, promptTokens, newTokens, prefill, decode, stopReason, _, _):
            return "#\(id) generated (\(model)): \(promptTokens) in, \(newTokens) out (\(String(format: "%.2fs", prefill)) prefill, \(String(format: "%.2fs", decode)) decode, stop: \(stopReason))"
        case let .requestFinished(id, status, durationMs, error):
            let err = error.map { " - Error: \($0)" } ?? ""
            return "#\(id) finished: status \(status), duration \(durationMs)ms\(err)"
        case let .modelAttached(atMs, model):
            return "[\(atMs)ms] Model attached: \(model)"
        case let .modelDetached(atMs, model):
            return "[\(atMs)ms] Model detached: \(model)"
        case let .unknown(kind):
            return "Unknown event: \(kind)"
        }
    }

    private func copyText(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}
