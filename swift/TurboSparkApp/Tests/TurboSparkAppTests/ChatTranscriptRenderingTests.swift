import AppKit
import SwiftUI
import XCTest

@testable import TurboSparkApp

@MainActor
final class ChatTranscriptRenderingTests: XCTestCase {
    func testCompletedAssistantRowFitsFullReplyPastBothPreviewLimits() async throws {
        let model = AppModel()
        defer { model.stopCronScheduler() }

        let replyEnd = "The complete answer ends here."
        let tallReply = String(repeating: "This is a separate paragraph.\n\n", count: 36) + replyEnd
        let longReply = String(repeating: "This longer paragraph checks that the saved reply reaches its final line.\n\n", count: 100) + replyEnd
        XCTAssertLessThan(tallReply.count, MessageContentPreview.collapseThreshold)
        XCTAssertGreaterThan(longReply.count, MessageContentPreview.collapseThreshold)

        for reply in [tallReply, longReply] {
            let markdown = NSHostingView(rootView: ChatMessageMarkdownView(reply).frame(width: 600))
            let row = NSHostingView(rootView: MessageRowView(
                model: model, message: AppChatMessage(role: .assistant, content: reply))
                .frame(width: 600))
            let frame = NSRect(x: 0, y: 0, width: 600, height: 400)
            let window = NSWindow(contentRect: frame, styleMask: [.borderless], backing: .buffered, defer: false)
            window.contentView = row
            try await settle(window)
            let fullHeight = markdown.fittingSize.height
            XCTAssertGreaterThan(fullHeight, 380)
            XCTAssertGreaterThanOrEqual(row.fittingSize.height, fullHeight,
                "The completed row should fit the full \(reply.count)-character reply.")
        }
    }

    func testActionBarVisibilityKeepsTranscriptViewportAndDocumentSize() async throws {
        let messageID = UUID()
        let date = Date()
        func transcript(showActions: Bool) -> AnyView {
            AnyView(ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    Text(verbatim: "A reply just short of filling the viewport.")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .frame(height: 176)
                    MessageActionBarView(text: "Reply", messageID: messageID, date: date, isVisible: showActions)
                }
            })
        }

        for style: NSScroller.Style in [.legacy, .overlay] {
            let frame = NSRect(x: 0, y: 0, width: 600, height: 200)
            let view = NSHostingView(rootView: transcript(showActions: false))
            let window = NSWindow(contentRect: frame, styleMask: [.borderless], backing: .buffered, defer: false)
            window.contentView = view
            try await settle(window)
            let scroll = try XCTUnwrap(firstScrollView(in: view))
            scroll.scrollerStyle = style
            scroll.tile()
            try await settle(window)
            let viewport = scroll.contentView.frame.size
            let documentHeight = try XCTUnwrap(scroll.documentView).frame.height
            XCTAssertGreaterThan(documentHeight, viewport.height,
                "The action bar's reserved space must put the fixture just over the overflow threshold.")

            for showActions in [true, false] {
                view.rootView = transcript(showActions: showActions)
                try await settle(window)
                XCTAssertIdentical(firstScrollView(in: view), scroll)
                XCTAssertEqual(scroll.scrollerStyle, style)
                XCTAssertEqual(scroll.contentView.frame.size, viewport)
                XCTAssertEqual(scroll.documentView?.frame.height, documentHeight)
            }
        }
    }

    private func settle(_ window: NSWindow) async throws {
        window.contentView?.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(100))
        window.contentView?.layoutSubtreeIfNeeded()
    }

    private func firstScrollView(in view: NSView) -> NSScrollView? {
        if let scroll = view as? NSScrollView { return scroll }
        for child in view.subviews {
            if let scroll = firstScrollView(in: child) { return scroll }
        }
        return nil
    }
}
