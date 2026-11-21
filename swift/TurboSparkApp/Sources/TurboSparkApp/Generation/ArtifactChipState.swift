import Foundation
import SwiftUI

/// What the transcript's artifact chips can say beyond "here is a file".
///
/// Both answers come only from state the app already holds. There is no
/// progress fraction for a write (the executor returns once, at the end), so
/// "writing" is a status, never a percentage, and the revision is
/// `AppArtifact.revision` as `upsert` bumps it, never a count re-derived from
/// the transcript (a rewrite moves `toolCallID`, so the transcript alone would
/// undercount).
enum ArtifactChipState {
    /// A call that is executing right now and will, if it succeeds, produce
    /// an artifact row anchored on `id`.
    struct Writing: Equatable, Identifiable {
        var id: UUID
        var fileName: String
    }

    /// Running write or edit calls on a markdown or HTML path that have no
    /// result and no artifact anchored on them yet.
    ///
    /// Mirrors `ArtifactRegistrar.artifactCandidates` for `.fileWrite`: a
    /// chip that says "writing" for `src/lib.rs` would never become an
    /// artifact. `.pendingApproval` is excluded on purpose: nothing is being
    /// written while the approval card waits. A path the resolver cannot
    /// settle (missing or conflicting keys) yields no row rather than a guess.
    static func writing(
        calls: [AppToolCall], results: [AppToolResult], artifacts: [AppArtifact]
    ) -> [Writing] {
        let finished = Set(results.map(\.callID))
        let anchored = Set(artifacts.compactMap(\.toolCallID))
        return calls.compactMap { call in
            guard call.status == .running,
                  !finished.contains(call.id),
                  !anchored.contains(call.id)
            else { return nil }
            let name = call.name.lowercased()
            guard AppToolRegistry.writeFileAliases.contains(name)
                    || AppToolRegistry.editFileAliases.contains(name)
            else { return nil }
            let resolved = ToolArgumentResolver.fileArguments(call.arguments)
            guard resolved.conflict == nil, let path = resolved.path, !path.isEmpty else { return nil }
            let fileName = (path as NSString).lastPathComponent
            let ext = (fileName as NSString).pathExtension.lowercased()
            guard AppArtifactRenderKind.markdownExtensions.contains(ext)
                    || AppArtifactRenderKind.htmlExtensions.contains(ext)
            else { return nil }
            return Writing(id: call.id, fileName: fileName)
        }
    }

    /// The revision to badge, or nil for a file written once. Same rule as
    /// `AppArtifact.detailText`, so the chip and the panel header agree.
    static func revisionBadge(revision: Int) -> Int? {
        revision > 1 ? revision : nil
    }
}

/// The in-flight twin of the artifact chip in `MessageRowView`: same shape,
/// not a button, because there is nothing to open until the write returns.
/// Its own view so only a row that is writing observes the motion setting.
struct ArtifactWritingChip: View {
    let fileName: String
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    /// App preference over system value; see `GenerationPhaseIndicator`.
    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        HStack(spacing: 6) {
            // The spinner is the only motion here; with Reduce Motion a
            // static glyph says the same thing, and the text says it in words.
            if reduceMotion {
                Image(systemName: "pencil")
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appAccent)
            } else {
                ProgressView().controlSize(.mini).scaleEffect(0.7)
            }
            Text(fileName)
                .themedFont(.tiny, weight: .medium)
                .lineLimit(1)
                .truncationMode(.middle)
            Text("Writing...", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
            Spacer(minLength: 4)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 1)
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Writing artifact \(fileName)", bundle: .module))
    }
}
