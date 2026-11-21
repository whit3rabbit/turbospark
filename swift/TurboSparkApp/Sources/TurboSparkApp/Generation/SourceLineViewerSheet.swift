import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Read-only view of a file a Read tool call read, with the lines it read
/// highlighted (component audit, source-viewer). It never edits, attaches,
/// or sends anything to the model; the only side effect is Copy Range.
@MainActor
struct SourceLineViewerSheet: View {
    @Environment(\.appTheme) private var theme
    @Environment(\.dismiss) private var dismiss

    /// The path as the call spelled it, shown in the header.
    let displayPath: String
    let resolution: SourceLineViewer.Resolution
    let requested: ClosedRange<Int>
    /// When the tool finished reading, to tell whether the file moved on.
    let readAt: Date

    private enum Phase: Equatable {
        case loading
        case loaded(SourceLineViewer.Loaded, range: ClosedRange<Int>, window: ClosedRange<Int>)
        case failed(Failure)
    }

    private enum Failure: Equatable {
        case noProject, outsideProject, sensitive, notFound, notAFile, tooLarge, notText
        case endsBefore(Int)
    }

    @State private var phase: Phase = .loading

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            content
                .frame(minWidth: 560, idealWidth: 720, minHeight: 360, idealHeight: 520)
            Divider()
            footer
        }
        .background(.appSurface)
        .task { await load() }
        // Esc closes even though Done holds the default action.
        .onExitCommand { dismiss() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label {
                Text(verbatim: displayPath)
                    .font(theme.code(.small, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.middle)
            } icon: {
                Image(systemName: "doc.text")
                    .foregroundStyle(.appAccentText)
                    .accessibilityHidden(true)
            }
            .foregroundStyle(.appText)
            .accessibilityAddTraits(.isHeader)
            .help(Text(verbatim: displayPath))

            HStack(spacing: 8) {
                Text("Read-only", bundle: .module)
                if case .loaded(let loaded, let range, _) = phase {
                    Text("Lines \(range.lowerBound)-\(range.upperBound) of \(loaded.lines.count)", bundle: .module)
                }
            }
            .themedFont(.small)
            .foregroundStyle(.appSecondary)

            if case .loaded(let loaded, _, _) = phase,
                SourceLineViewer.mayHaveChanged(modifiedAt: loaded.modifiedAt, readAt: readAt)
            {
                // Glyph plus words: the warning never rests on color alone.
                Label {
                    Text("File may have changed since it was read", bundle: .module)
                        .foregroundStyle(.appText)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.appWarning)
                        .accessibilityHidden(true)
                }
                .themedFont(.small)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
    }

    @ViewBuilder
    private var content: some View {
        switch phase {
        case .loading:
            ProgressView()
                .controlSize(.small)
                .accessibilityLabel(Text("Loading...", bundle: .module))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .failed(let failure):
            failureText(failure)
                .themedFont(.base)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(24)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .loaded(let loaded, let range, let window):
            lines(loaded.lines, range: range, window: window)
        }
    }

    private func lines(_ all: [String], range: ClosedRange<Int>, window: ClosedRange<Int>) -> some View {
        // Width of the largest number in the window, so the gutter aligns.
        let gutterDigits = String(window.upperBound).count
        return ScrollViewReader { proxy in
            ScrollView([.vertical, .horizontal]) {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(window), id: \.self) { number in
                        let isMarked = range.contains(number)
                        HStack(spacing: 6) {
                            Text(verbatim: String(number).leftPadded(to: gutterDigits))
                                .font(theme.code(.small))
                                .foregroundStyle(.appSecondary)
                            // The bar marks the line by shape as well as tint.
                            Rectangle()
                                .fill(isMarked ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.clear))
                                .frame(width: 2)
                                .accessibilityHidden(true)
                            let text = SourceLineViewer.displayLine(all[number - 1])
                            // An empty Text has no height; a space keeps blank lines.
                            Text(verbatim: text.isEmpty ? " " : text)
                                .font(theme.code(.small))
                                .foregroundStyle(.appText)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(isMarked ? AnyShapeStyle(.appAccent.opacity(0.12)) : AnyShapeStyle(.clear))
                        .accessibilityElement(children: .combine)
                        .accessibilityValue(isMarked ? Text("Highlighted line", bundle: .module) : Text(verbatim: ""))
                        .id(number)
                    }
                }
                // Equal-width rows, so every band spans the widest line.
                .fixedSize(horizontal: true, vertical: false)
                .textSelection(.enabled)
                .padding(12)
            }
            // Jump, never animate: honors Reduce Motion with nothing to gate.
            .onAppear { proxy.scrollTo(range.lowerBound, anchor: .top) }
        }
    }

    private func failureText(_ failure: Failure) -> Text {
        switch failure {
        case .noProject:
            return Text("Select the project this file belongs to, then open it again.", bundle: .module)
        case .outsideProject:
            return Text("This file is outside the project, so it is not shown.", bundle: .module)
        case .sensitive:
            return Text("This file may hold credentials, so it is not shown.", bundle: .module)
        case .notFound:
            return Text("The file no longer exists.", bundle: .module)
        case .notAFile:
            return Text("This path is not a regular file.", bundle: .module)
        case .tooLarge:
            return Text("The file is larger than 2 MB, so it is not shown.", bundle: .module)
        case .notText:
            return Text("The file is not UTF-8 text.", bundle: .module)
        case .endsBefore(let line):
            return Text("The file now ends before line \(line).", bundle: .module)
        }
    }

    private var footer: some View {
        HStack {
            Spacer()
            if case .loaded(let loaded, let range, _) = phase {
                Button { copy(loaded.lines, range: range) } label: {
                    Label { Text("Copy Range", bundle: .module) } icon: { Image(systemName: "doc.on.doc") }
                }
                .appAction(.secondary)
            }
            Button { dismiss() } label: { Text("Done", bundle: .module) }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
        }
        .padding(16)
    }

    private func copy(_ lines: [String], range: ClosedRange<Int>) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(SourceLineViewer.copyText(lines: lines, range: range), forType: .string)
        _ = AccessibilityNotification.Announcement.post(
            .init(AppStrings.localized("Copied lines \(range.lowerBound)-\(range.upperBound)")))
    }

    private func load() async {
        let url: URL
        switch resolution {
        case .file(let resolved): url = resolved
        case .noProject: phase = .failed(.noProject); return
        case .outsideProject: phase = .failed(.outsideProject); return
        case .sensitive: phase = .failed(.sensitive); return
        }
        // Off the main actor: up to 2 MB of disk read and UTF-8 decoding.
        let result = await Task.detached(priority: .userInitiated) { SourceLineViewer.load(url) }.value
        switch result {
        case .failure(let failure):
            switch failure {
            case .notFound: phase = .failed(.notFound)
            case .notAFile: phase = .failed(.notAFile)
            case .tooLarge: phase = .failed(.tooLarge)
            case .notText: phase = .failed(.notText)
            }
        case .success(let loaded):
            guard let range = SourceLineViewer.clamp(requested, lineCount: loaded.lines.count) else {
                phase = .failed(.endsBefore(requested.lowerBound))
                return
            }
            phase = .loaded(
                loaded, range: range,
                window: SourceLineViewer.window(for: range, lineCount: loaded.lines.count))
        }
    }
}

extension String {
    /// Right-aligns a line number in a fixed-width gutter.
    fileprivate func leftPadded(to width: Int) -> String {
        count >= width ? self : String(repeating: " ", count: width - count) + self
    }
}
