import AppKit
import SwiftUI

/// A parsed line of a unified Git diff.
public struct ParsedDiffLine: Identifiable, Hashable, Sendable {
    public enum Kind: Sendable {
        case context
        case addition
        case deletion
        case hunkHeader
    }

    public let id: Int
    public let kind: Kind
    public let oldLineNumber: Int?
    public let newLineNumber: Int?
    public let text: String
}

/// A block of diff lines or a collapsible folded group of unmodified lines.
public enum DiffBlock: Identifiable, Sendable {
    case foldedContext(id: Int, count: Int, lines: [ParsedDiffLine])
    case lines(id: Int, lines: [ParsedDiffLine])

    public var id: Int {
        switch self {
        case .foldedContext(let id, _, _): return id
        case .lines(let id, _): return id
        }
    }
}

/// Rich Git diff viewer displaying collapsible context lines, hunk headers, and line numbers.
@MainActor
public struct WorktreeDiffView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var worktree: WorktreeModel
    public let file: WorktreeFileChange
    public let diff: String
    public let onClose: (() -> Void)?

    @State private var expandedBlocks: Set<Int> = []
    @State private var copied: Bool = false

    public init(
        worktree: WorktreeModel,
        file: WorktreeFileChange,
        diff: String,
        onClose: (() -> Void)? = nil
    ) {
        self.worktree = worktree
        self.file = file
        self.diff = diff
        self.onClose = onClose
    }

    public var body: some View {
        VStack(spacing: 0) {
            fileHeader
            Divider()
            diffContent
        }
        .background(.appSurface.opacity(0.3))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }

    private var fileHeader: some View {
        HStack(spacing: 8) {
            Image(systemName: file.status.systemImage)
                .themedFont(.small)
                .foregroundStyle(file.status.color)

            Text(file.fileName)
                .font(theme.code(.base, weight: .semibold))
                .foregroundStyle(.appText)

            if !file.directoryPath.isEmpty {
                Text(file.directoryPath)
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            Spacer()

            if file.additions > 0 || file.deletions > 0 {
                HStack(spacing: 4) {
                    if file.additions > 0 {
                        Text("+\(file.additions)", bundle: .module)
                            .themedFont(.tiny, weight: .semibold).monospacedDigit()
                            .foregroundStyle(.green)
                    }
                    if file.deletions > 0 {
                        Text("-\(file.deletions)", bundle: .module)
                            .themedFont(.tiny, weight: .semibold).monospacedDigit()
                            .foregroundStyle(.red)
                    }
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.primary.opacity(0.04), in: Capsule())
            }

            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(diff, forType: .string)
                copied = true
                Task {
                    try? await Task.sleep(nanoseconds: 1_500_000_000)
                    copied = false
                }
            } label: {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    .themedFont(.tiny)
                    .foregroundStyle(copied ? .green : .secondary)
            }
            .buttonStyle(.plain)
            .help("Copy raw diff")

            if let onClose = onClose {
                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)
                .help("Close diff view")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.appPage.opacity(0.8))
    }

    private var diffContent: some View {
        let blocks = parseDiff(diff)
        return ScrollView([.vertical, .horizontal], showsIndicators: true) {
            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(blocks) { block in
                    switch block {
                    case .foldedContext(let id, let count, let lines):
                        let isExpanded = expandedBlocks.contains(id)
                        if isExpanded {
                            VStack(spacing: 0) {
                                foldHeader(count: count, isExpanded: true) {
                                    expandedBlocks.remove(id)
                                }
                                ForEach(lines) { line in
                                    lineRow(line)
                                }
                            }
                        } else {
                            foldHeader(count: count, isExpanded: false) {
                                expandedBlocks.insert(id)
                            }
                        }
                    case .lines(_, let lines):
                        ForEach(lines) { line in
                            lineRow(line)
                        }
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    private func foldHeader(count: Int, isExpanded: Bool, toggle: @escaping () -> Void) -> some View {
        Button(action: toggle) {
            HStack(spacing: 6) {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
                Text("\(count) unmodified lines", bundle: .module)
                    .font(theme.code(.small, weight: .medium))
                    .foregroundStyle(.appSecondary)
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(Color.primary.opacity(0.03))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func lineRow(_ line: ParsedDiffLine) -> some View {
        let (fgColor, bgColor): (Color, Color) = {
            switch line.kind {
            case .addition:
                return (.green, Color.green.opacity(0.09))
            case .deletion:
                return (.red, Color.red.opacity(0.09))
            case .hunkHeader:
                return (TurboSparkTheme.accentColor, TurboSparkTheme.accentColor.opacity(0.06))
            case .context:
                return (.secondary, Color.clear)
            }
        }()

        let lineGutter = formatLineNumber(old: line.oldLineNumber, new: line.newLineNumber, kind: line.kind)

        return HStack(spacing: 8) {
            Text(lineGutter)
                .font(theme.code(.small))
                .foregroundStyle(.tertiary)
                .frame(width: 54, alignment: .trailing)

            Text(line.text)
                .font(theme.code(.small))
                .foregroundStyle(fgColor)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 6)
        .padding(.vertical, 1)
        .background(bgColor)
    }

    private func formatLineNumber(old: Int?, new: Int?, kind: ParsedDiffLine.Kind) -> String {
        switch kind {
        case .addition:
            return "\(new.map(String.init) ?? "") +"
        case .deletion:
            return "\(old.map(String.init) ?? "") -"
        case .hunkHeader:
            return "@@"
        case .context:
            return "\(new.map(String.init) ?? "")"
        }
    }

    /// Parses unified diff string into blocks with folded context groups for clean display.
    private func parseDiff(_ raw: String) -> [DiffBlock] {
        var rawLines = raw.components(separatedBy: .newlines)
        if let firstHunkIndex = rawLines.firstIndex(where: { $0.hasPrefix("@@") }) {
            rawLines = Array(rawLines[firstHunkIndex...])
        }

        var parsedLines: [ParsedDiffLine] = []
        var currentOldLine = 0
        var currentNewLine = 0
        var lineIndex = 0

        for text in rawLines {
            lineIndex += 1
            if text.hasPrefix("@@") {
                let (oldStart, newStart) = parseHunkHeader(text)
                currentOldLine = oldStart
                currentNewLine = newStart
                parsedLines.append(
                    ParsedDiffLine(
                        id: lineIndex,
                        kind: .hunkHeader,
                        oldLineNumber: nil,
                        newLineNumber: nil,
                        text: text
                    )
                )
            } else if text.hasPrefix("+") && !text.hasPrefix("+++") {
                parsedLines.append(
                    ParsedDiffLine(
                        id: lineIndex,
                        kind: .addition,
                        oldLineNumber: nil,
                        newLineNumber: currentNewLine,
                        text: text
                    )
                )
                currentNewLine += 1
            } else if text.hasPrefix("-") && !text.hasPrefix("---") {
                parsedLines.append(
                    ParsedDiffLine(
                        id: lineIndex,
                        kind: .deletion,
                        oldLineNumber: currentOldLine,
                        newLineNumber: nil,
                        text: text
                    )
                )
                currentOldLine += 1
            } else if text.isEmpty || text.hasPrefix("\\") {
                // Metadata, not content: the empty trailing component
                // `components(separatedBy:)` leaves behind, and git's
                // "\\ No newline at end of file" marker. Rendering either
                // as a context row would show a phantom line AND advance
                // both counters, shifting every later line number.
                continue
            } else {
                let cleanText = text.hasPrefix(" ") ? String(text.dropFirst()) : text
                parsedLines.append(
                    ParsedDiffLine(
                        id: lineIndex,
                        kind: .context,
                        oldLineNumber: currentOldLine > 0 ? currentOldLine : nil,
                        newLineNumber: currentNewLine > 0 ? currentNewLine : nil,
                        text: cleanText
                    )
                )
                if currentOldLine > 0 { currentOldLine += 1 }
                if currentNewLine > 0 { currentNewLine += 1 }
            }
        }

        var blocks: [DiffBlock] = []
        var contextBuffer: [ParsedDiffLine] = []
        var blockId = 0

        func flushContext() {
            guard !contextBuffer.isEmpty else { return }
            blockId += 1
            if contextBuffer.count >= 8 {
                blocks.append(.foldedContext(id: blockId, count: contextBuffer.count, lines: contextBuffer))
            } else {
                blocks.append(.lines(id: blockId, lines: contextBuffer))
            }
            contextBuffer.removeAll()
        }

        var nonContextBuffer: [ParsedDiffLine] = []
        func flushNonContext() {
            guard !nonContextBuffer.isEmpty else { return }
            blockId += 1
            blocks.append(.lines(id: blockId, lines: nonContextBuffer))
            nonContextBuffer.removeAll()
        }

        for line in parsedLines {
            if line.kind == .context {
                flushNonContext()
                contextBuffer.append(line)
            } else {
                flushContext()
                nonContextBuffer.append(line)
            }
        }
        flushContext()
        flushNonContext()

        return blocks
    }

    private func parseHunkHeader(_ text: String) -> (Int, Int) {
        // Only the span between the two @@ markers carries counts. Scanning
        // the whole line reads git's trailing function-context hint too, and
        // a hunk like "@@ -10,7 +10,8 @@ x = -5" would overwrite oldStart
        // with 5 off the "-5" in the hint.
        // Only the span between the two @@ markers carries counts. Scanning
        // the whole line reads git's trailing function-context hint too, and
        // a hunk like "@@ -10,7 +10,8 @@ x = -5" would overwrite oldStart
        // with 5 off the "-5" in the hint.
        guard let openRange = text.range(of: "@@"),
            let closeRange = text.range(of: "@@", range: openRange.upperBound..<text.endIndex)
        else { return (1, 1) }
        let counts = text[openRange.upperBound..<closeRange.lowerBound]
        var oldStart = 1
        var newStart = 1
        for part in counts.split(separator: " ") {
            if part.hasPrefix("-") {
                let sub = part.dropFirst().split(separator: ",")
                if let num = Int(sub.first ?? "") { oldStart = num }
            } else if part.hasPrefix("+") {
                let sub = part.dropFirst().split(separator: ",")
                if let num = Int(sub.first ?? "") { newStart = num }
            }
        }
        return (oldStart, newStart)
    }
}
