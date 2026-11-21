import SwiftUI

/// Holds the Files section's staging list while imports run.
///
/// Lives as view state, not on AppModel: it is a report of the last few
/// imports, not data. Closing the section drops it; the attachments the
/// imports produced are on their chats either way.
@MainActor
final class SourceIngestQueue: ObservableObject {
    @Published private(set) var entries: [SourceIngestStaging.Entry] = []

    var summary: SourceIngestStaging.Summary { SourceIngestStaging.summary(entries) }

    /// Runs the existing import for `urls` and returns its outcome so the
    /// caller keeps its success toast.
    func run(_ urls: [URL], into model: AppModel, chatID: UUID?) async -> AttachmentImporter.Outcome {
        entries = SourceIngestStaging.staged(entries, adding: urls, chatID: chatID)
        let outcome = await AttachmentImporter.importDocuments(
            urls, into: model, chatID: chatID,
            onReading: { [weak self] url in
                guard let self else { return }
                self.entries = SourceIngestStaging.markingReading(url, in: self.entries)
            })
        entries = SourceIngestStaging.applying(outcome.fileResults, to: entries)
        // No message to give: the importer returns a result for every URL
        // it was handed, so this only guards against a row left spinning.
        entries = SourceIngestStaging.finishing(urls, in: entries, fallback: "")
        return outcome
    }

    /// Retries one failed file with the same import call and the chat the
    /// original batch targeted.
    func retry(_ id: UUID, into model: AppModel) async -> AttachmentImporter.Outcome? {
        guard let entry = entries.first(where: { $0.id == id }),
            SourceIngestStaging.canRetry(entry.state)
        else { return nil }
        return await run([entry.url], into: model, chatID: entry.chatID)
    }

    func clearFinished() {
        entries.removeAll {
            switch $0.state {
            case .queued, .reading: return false
            default: return true
            }
        }
    }
}

/// The staging list: one row per file of the recent imports, failures first.
@MainActor
struct SourceIngestStagingView: View {
    @ObservedObject var queue: SourceIngestQueue
    /// Retry goes through the importer's own guard; this mirrors it so the
    /// button is disabled with a reason instead of failing on press.
    let canImport: Bool
    let onRetry: (UUID) -> Void

    var body: some View {
        let summary = queue.summary
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Recent imports", bundle: .module)
                        .themedFont(.small, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                    Text(
                        "\(summary.added) added, \(summary.skipped) skipped, \(summary.failed) failed",
                        bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                Spacer(minLength: 4)
                if !summary.isInFlight {
                    Button { queue.clearFinished() } label: { Text("Clear", bundle: .module) }
                        .appAction(.quiet, size: .small)
                }
            }
            ForEach(SourceIngestStaging.ordered(queue.entries)) { entry in
                SourceIngestRow(entry: entry, canImport: canImport) { onRetry(entry.id) }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }
}

@MainActor
private struct SourceIngestRow: View {
    let entry: SourceIngestStaging.Entry
    let canImport: Bool
    let onRetry: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            glyph
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text(verbatim: entry.fileName)
                    .themedFont(.small)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(Text(verbatim: entry.url.path))
                stateText
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .accessibilityElement(children: .combine)
            Spacer(minLength: 4)
            if SourceIngestStaging.canRetry(entry.state) {
                Button(action: onRetry) { Text("Retry", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .disabled(!canImport)
                    .accessibilityLabel(Text("Retry \(entry.fileName)", bundle: .module))
                    .help(canImport ? Text("Retry", bundle: .module)
                        : Text("Cannot attach files while generating.", bundle: .module))
            }
        }
    }

    /// Glyph plus word, never color alone (StatusLabel guide).
    @ViewBuilder
    private var glyph: some View {
        switch entry.state {
        case .queued:
            Image(systemName: "clock").foregroundStyle(.appSecondary)
        case .reading:
            ProgressView().controlSize(.mini)
        case .added:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.appSuccess)
        case .skipped:
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.appWarning)
        case .failed:
            Image(systemName: "xmark.octagon.fill").foregroundStyle(.appDanger)
        }
    }

    private var stateText: Text {
        switch entry.state {
        case .queued: return Text("Queued", bundle: .module)
        case .reading: return Text("Reading", bundle: .module)
        case .added: return Text("Added", bundle: .module)
        case .skipped(let reason):
            return Text("Skipped", bundle: .module) + Text(verbatim: ": ") + skipReason(reason)
        case .failed(let message):
            // The importer's message, shown verbatim as the old error row did.
            if message.isEmpty { return Text("Failed", bundle: .module) }
            return Text("Failed", bundle: .module) + Text(verbatim: ": \(message)")
        }
    }

    private func skipReason(_ reason: SourceIngestStaging.SkipReason) -> Text {
        switch reason {
        case .tooLarge: return Text("Too large", bundle: .module)
        case .unsupportedType: return Text("Unsupported file type", bundle: .module)
        case .unreadable: return Text("Could not be read", bundle: .module)
        case .noText: return Text("No selectable text. Scanned PDFs need OCR.", bundle: .module)
        }
    }
}
