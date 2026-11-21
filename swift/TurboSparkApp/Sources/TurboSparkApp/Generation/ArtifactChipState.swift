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

// MARK: - Hover peek

extension ArtifactChipState {
    /// Lines, characters per line and bytes the peek reads at most. A
    /// tooltip is a glance, and a hover must never read a whole file.
    static let peekLineCount = 4
    static let peekLineLength = 120
    static let peekByteLimit = 4096

    /// The file a peek may read: a markdown, HTML or text artifact at a
    /// plain path. Nil for other kinds (no lines to show), for the in-memory
    /// plan, and for an encrypted-profile asset, whose `url` would decrypt
    /// it to disk just because the pointer passed over the chip.
    static func peekURL(for artifact: AppArtifact) -> URL? {
        guard let path = artifact.path, ManagedAssetStore.assetID(from: path) == nil,
              [.markdown, .html, .text].contains(artifact.renderKind)
        else { return nil }
        return URL(fileURLWithPath: path)
    }

    /// Up to `peekByteLimit` bytes of `url` as text, or nil when it is not a
    /// regular file or not UTF-8. `O_NONBLOCK` plus `fstat` on the open
    /// descriptor, as `SourceLineViewer.load` does, so a FIFO swapped in at
    /// the path never blocks the read.
    static func readHead(_ url: URL) -> String? {
        let fd = open(url.path, O_RDONLY | O_NONBLOCK)
        guard fd >= 0 else { return nil }
        let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG,
              let data = try? handle.read(upToCount: peekByteLimit)
        else { return nil }
        return decodeHead(data, truncated: info.st_size > Int64(data.count))
    }

    /// UTF-8 text of a file head. When the read stopped early the last
    /// character may be cut, so up to three trailing bytes are dropped
    /// before giving up. A NUL byte means binary: no peek rather than noise.
    static func decodeHead(_ data: Data, truncated: Bool) -> String? {
        guard !data.contains(0) else { return nil }
        if let text = String(data: data, encoding: .utf8) { return text }
        guard truncated else { return nil }
        for cut in 1...3 where data.count > cut {
            if let text = String(data: data.dropLast(cut), encoding: .utf8) { return text }
        }
        return nil
    }

    /// The first `peekLineCount` lines after any leading blank ones, each cut
    /// to `peekLineLength` characters. Empty when nothing is worth showing.
    static func peekLines(_ text: String) -> [String] {
        // "\r\n" is one Character, so it must be named as a separator.
        let all = text.split(
            omittingEmptySubsequences: false, whereSeparator: { $0 == "\n" || $0 == "\r\n" }
        ).map { line -> String in
            let trimmed = line.hasSuffix("\r") ? line.dropLast() : line[...]
            return String(trimmed)
        }
        guard let first = all.firstIndex(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty })
        else { return [] }
        return all[first...].prefix(peekLineCount).map { line in
            line.count > peekLineLength ? String(line.prefix(peekLineLength)) + "..." : line
        }
    }

    /// The tooltip: title, kind and version, the first lines when there are
    /// any, then what a click does (the tooltip it replaces said only that).
    /// The kind label is the same unlocalized format word the chip shows.
    static func peekText(
        title: String, formatLabel: String, revision: Int, fileName: String, lines: [String]
    ) -> String {
        var rows = [title, formatLabel + " \u{00B7} " + AppStrings.localized("Version \(revision)")]
        if !lines.isEmpty {
            rows.append("")
            rows += lines
        }
        rows.append("")
        rows.append(AppStrings.localized("Open \(fileName) in the artifact panel"))
        return rows.joined(separator: "\n")
    }
}

/// Press feedback for the artifact chip: the fill deepens and the border
/// takes the accent, so the press reads without motion; the slight scale is
/// added only when motion is allowed.
struct ArtifactChipButtonStyle: ButtonStyle {
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    func makeBody(configuration: Configuration) -> some View {
        let reduce = appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
        let shape = RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        return configuration.label
            .background(configuration.isPressed ? AnyShapeStyle(.appElevated) : AnyShapeStyle(.appSurface))
            .clipShape(shape)
            .overlay(shape.stroke(
                configuration.isPressed ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appBorder), lineWidth: 1))
            .scaleEffect(configuration.isPressed && !reduce ? 0.98 : 1)
            .animation(reduce ? nil : TSMotion.press, value: configuration.isPressed)
    }
}

/// The transcript chip that reopens an artifact in the panel. Its own view
/// so each chip keeps its peek lines; they load on the first hover, off the
/// main actor, and are dropped when the file is rewritten.
struct ArtifactOpenChip: View {
    let artifact: AppArtifact
    let open: () -> Void
    @State private var peekLines: [String]?

    var body: some View {
        Button(action: open) {
            HStack(spacing: 6) {
                Image(systemName: artifact.symbolName)
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text(artifact.fileName)
                    .themedFont(.tiny, weight: .medium)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(artifact.formatLabel)
                    .themedCode(.tiny)
                    .foregroundStyle(.appSecondary)
                if let revision = ArtifactChipState.revisionBadge(revision: artifact.revision) {
                    Text("v\(revision)", bundle: .module)
                        .themedCode(.tiny)
                        .monospacedDigit()
                        .foregroundStyle(.appSecondary)
                        .accessibilityHidden(true)
                }
                Spacer(minLength: 4)
                Image(systemName: "sidebar.right")
                    .themedFont(.micro)
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .contentShape(Rectangle())
        }
        .buttonStyle(ArtifactChipButtonStyle())
        .help(Text(verbatim: ArtifactChipState.peekText(
            title: artifact.title, formatLabel: artifact.formatLabel, revision: artifact.revision,
            fileName: artifact.fileName, lines: peekLines ?? [])))
        .onHover { hovering in
            if hovering { loadPeek() }
        }
        .onChange(of: artifact.contentKey) { _, _ in peekLines = nil }
        .accessibilityLabel(Text("Open artifact \(artifact.fileName)", bundle: .module))
        .accessibilityValue(
            ArtifactChipState.revisionBadge(revision: artifact.revision)
                .map { Text("Version \($0)", bundle: .module) } ?? Text(verbatim: ""))
    }

    /// Once per revision. A file that cannot be read leaves an empty list,
    /// so the tooltip still names the artifact and is not retried per hover.
    private func loadPeek() {
        guard peekLines == nil else { return }
        guard let url = ArtifactChipState.peekURL(for: artifact) else {
            peekLines = []
            return
        }
        peekLines = []
        let key = artifact.contentKey
        Task {
            let lines = await Task.detached(priority: .utility) {
                ArtifactChipState.readHead(url).map(ArtifactChipState.peekLines) ?? []
            }.value
            guard artifact.contentKey == key else { return }
            peekLines = lines
        }
    }
}
