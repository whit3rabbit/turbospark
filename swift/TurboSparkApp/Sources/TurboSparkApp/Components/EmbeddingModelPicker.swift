import SwiftUI
import TurboSpark

/// Chooses an embedding model: the pinned catalog profiles first, each with
/// its tested status, then a field for any other model directory or alias.
///
/// The same component serves the API Embeddings tab and Memory settings, so
/// both describe models the same way. What counts as tested and which
/// families run is decided in Rust (`EmbeddingProfile.status`,
/// `EmbeddingCatalog.probe`); this view only shows the verdict. A profile
/// this Mac cannot run is listed with the reason rather than hidden.
@MainActor
struct EmbeddingModelPicker: View {
    @ObservedObject var model: AppModel
    @Binding var selection: String
    var disabled: Bool = false
    /// Called when Use fills the field or the field is submitted.
    var onCommit: () -> Void = {}

    private var trimmed: String { selection.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(model.embeddingProfiles) { profile in
                profileRow(profile)
            }
            TextField(text: $selection, prompt: Text("Embedding model path or alias", bundle: .module)) {
                Text("Embedding model", bundle: .module)
            }
            .textFieldStyle(.roundedBorder)
            .disabled(disabled)
            .onSubmit { onCommit() }
            verdictLine
        }
        .onAppear { model.refreshEmbeddingProfiles() }
        // Debounced: the probe reads a few files, and every keystroke
        // would otherwise run one.
        .task(id: trimmed) {
            guard !trimmed.isEmpty else { return }
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            await model.refreshEmbeddingProbe(trimmed)
        }
    }

    private func profileRow(_ profile: EmbeddingProfile) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(verbatim: profile.displayName)
                        .themedFont(.small, weight: .medium)
                    EmbeddingStatusLabel(status: profile.status)
                }
                Text(verbatim: ByteCountFormatter.string(
                    fromByteCount: Int64(clamping: profile.downloadBytes), countStyle: .file))
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                if !profile.canRun, let reason = profile.unavailableReason {
                    Text(verbatim: reason)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                } else if profile.installState == "stale" || profile.installState == "foreign",
                          let problem = profile.installProblem {
                    Text(verbatim: problem)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .accessibilityElement(children: .combine)
            Spacer()
            profileAction(profile)
        }
    }

    @ViewBuilder
    private func profileAction(_ profile: EmbeddingProfile) -> some View {
        if !profile.canRun {
            Label {
                Text("Runtime unavailable", bundle: .module)
            } icon: {
                Image(systemName: "wrench.and.screwdriver")
            }
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
        } else if profile.installedPath == nil {
            Button {
                model.enqueueEmbeddingDownload(profile)
            } label: { Text("Download", bundle: .module) }
            .accessibilityLabel(Text("Download", bundle: .module) + Text(verbatim: " \(profile.displayName)"))
            .appAction(.secondary, size: .small)
            .disabled(disabled || !model.canQueueModelDownload(AppModel.embeddingDownloadRequest(profile)))
            .help(disabled || !model.canQueueModelDownload(AppModel.embeddingDownloadRequest(profile))
                ? Text("Download cannot be queued at this time.", bundle: .module)
                : Text("Download", bundle: .module))
            .appPointerCursor()
        } else if trimmed == profile.alias {
            Label {
                Text("Selected", bundle: .module)
            } icon: {
                Image(systemName: "checkmark.circle")
            }
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
            .accessibilityAddTraits(.isSelected)
        } else {
            Button {
                selection = profile.alias
                onCommit()
            } label: { Text("Use", bundle: .module) }
            .accessibilityLabel(Text("Use", bundle: .module) + Text(verbatim: " \(profile.displayName)"))
            .appAction(.secondary, size: .small)
            .disabled(disabled)
            .appPointerCursor()
        }
    }

    /// The verdict for whatever the field holds, from the Rust probe.
    @ViewBuilder
    private var verdictLine: some View {
        if !trimmed.isEmpty, let probe = model.embeddingProbes[trimmed] {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                if !probe.runnable {
                    Label {
                        Text("Runtime unavailable", bundle: .module)
                    } icon: {
                        Image(systemName: "exclamationmark.octagon")
                    }
                    .foregroundStyle(AppAttention.color)
                    if let reason = probe.refusedBecause {
                        Text(verbatim: reason)
                            .foregroundStyle(.appSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                } else {
                    EmbeddingStatusLabel(status: probe.verified ? "verified" : "runs")
                    if let family = probe.family {
                        Text(verbatim: family)
                            .themedCode(.small)
                            .foregroundStyle(.appSecondary)
                    }
                }
            }
            .themedFont(.small)
            .accessibilityElement(children: .combine)
        }
    }
}

/// A profile's or a probe's tested status, as a glyph and a word so it
/// survives Differentiate Without Color. `verified` is the only status that
/// claims a test; everything else reads as not verified.
struct EmbeddingStatusLabel: View {
    let status: String

    var body: some View {
        switch status {
        case "verified":
            Label {
                Text("Verified", bundle: .module)
            } icon: {
                Image(systemName: "checkmark.seal.fill")
            }
            .themedFont(.tiny, weight: .medium)
            .foregroundStyle(.appAccent)
            .help(Text("Verified: a recorded test ran this exact download.", bundle: .module))
        case "caveat":
            Label {
                Text("Caveat", bundle: .module)
            } icon: {
                Image(systemName: "exclamationmark.triangle.fill")
            }
            .themedFont(.tiny, weight: .medium)
            .foregroundStyle(AppAttention.color)
        default:
            Label {
                Text("Not verified", bundle: .module)
            } icon: {
                Image(systemName: "questionmark.circle")
            }
            .themedFont(.tiny)
            .foregroundStyle(.appSecondary)
            .help(Text("This model family runs here, but no recorded test covers these files.", bundle: .module))
        }
    }
}
