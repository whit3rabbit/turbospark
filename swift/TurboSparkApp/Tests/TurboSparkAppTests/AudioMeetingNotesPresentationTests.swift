import AppKit
import SwiftUI
import XCTest

@testable import TurboSparkApp

/// The Notes & Summary view of an audio item. Generated notes once showed as
/// raw markdown (`### Key Decisions`, `- [ ]`), and the "Detail View" label of
/// the transcript/notes switcher wrapped one or two letters per line in a
/// 1625 pt window, because a labeled segmented picker was squeezed to 280 pt.
@MainActor
final class AudioMeetingNotesPresentationTests: XCTestCase {
    private func host<V: View>(_ view: V, width: CGFloat) -> NSHostingView<AnyView> {
        let host = NSHostingView(rootView: AnyView(view.frame(width: width, alignment: .leading)))
        host.frame = NSRect(x: 0, y: 0, width: width, height: 10)
        let window = NSWindow(contentRect: host.frame, styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        host.frame = NSRect(origin: .zero, size: host.fittingSize)
        host.layoutSubtreeIfNeeded()
        return host
    }

    /// `swift/TurboSparkApp/Sources/TurboSparkApp`.
    private func source(_ relative: String) throws -> String {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
        return try String(contentsOf: sources.appendingPathComponent(relative), encoding: .utf8)
    }

    /// A source check, because a hosted SwiftUI view exposes no accessibility
    /// tree to an XCTest process to read the rendered text from. The reader
    /// must go through the shared markdown view, never a verbatim `Text`.
    func testNotesRenderThroughTheMarkdownView() throws {
        let file = try source("Audio/AudioMeetingNotesView.swift")
        let start = try XCTUnwrap(file.range(of: "struct AudioMeetingNotesReader"))
        let reader = String(file[start.lowerBound...])
        XCTAssertTrue(reader.contains("ChatMessageMarkdownView(text)"), reader)
        XCTAssertFalse(reader.contains("Text(verbatim"), reader)
        // It still renders: a heading and a task list draw something.
        let hosted = host(AudioMeetingNotesReader(text: "### Key Decisions\n- [ ] Draft the notes"), width: 600)
        XCTAssertGreaterThan(hosted.fittingSize.height, 20)
    }

    func testTheModeSwitcherStaysOnOneLineInAWideWindow() {
        for hasSummary in [false, true] {
            let view = AudioDetailModeSwitcher(selection: .constant(.transcript), hasSummary: hasSummary)
            let hosted = host(view, width: 1625)
            // One line of the pill font, plus at most 16 pt of padding. A
            // wrapped label is several lines tall.
            let line = host(Text(verbatim: "Detail View").themedFont(.small, weight: .medium), width: 1625)
                .fittingSize.height
            XCTAssertLessThanOrEqual(
                hosted.fittingSize.height, line + 16,
                "hasSummary \(hasSummary): \(hosted.fittingSize.height) against one line of \(line)")
        }
    }
}
