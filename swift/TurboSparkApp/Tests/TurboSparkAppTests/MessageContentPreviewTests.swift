import Foundation
import XCTest

@testable import TurboSparkApp

/// Tests for the string-level message preview gate.
final class MessageContentPreviewTests: XCTestCase {
    private func repeated(_ text: String, _ count: Int) -> String {
        String(repeating: text, count: count)
    }

    func testShortTextGetsNoPreview() {
        XCTAssertNil(MessageContentPreview.make(repeated("a", 1_000)))
        XCTAssertNil(MessageContentPreview.make(repeated("a", 5_999)))
    }

    func testAtThresholdGetsNoPreview() {
        XCTAssertNil(MessageContentPreview.make(repeated("a", MessageContentPreview.collapseThreshold)))
    }

    func testUserPreviewKeepsHeadAndTailWithHiddenCount() {
        let text = repeated("a", 3_000) + repeated("m", 4_000) + repeated("z", 3_000)
        let preview = MessageContentPreview.make(text)
        // Built through the same interpolation path as the production
        // marker: String(localized:) formats Int with locale grouping.
        let marker = String(localized: "[... \(5_500) characters hidden ...]", bundle: .module)
        XCTAssertNotNil(preview)
        XCTAssertEqual(
            preview?.visible,
            String(text.prefix(4_000)) + "\n\n" + marker + "\n\n" + String(text.suffix(500)))
        XCTAssertEqual(preview?.hiddenCharacterCount, 5_500)
    }

    func testHeadOnlyPreviewOmitsTheTail() {
        let text = repeated("a", 3_000) + repeated("m", 4_000) + repeated("z", 3_000)
        let preview = MessageContentPreview.make(text, includesTail: false)
        let marker = String(localized: "[... \(6_000) characters hidden ...]", bundle: .module)
        XCTAssertNotNil(preview)
        XCTAssertEqual(preview?.visible, String(text.prefix(4_000)) + "\n\n" + marker)
        XCTAssertEqual(preview?.hiddenCharacterCount, 6_000)
        XCTAssertFalse(preview?.visible.contains(String(text.suffix(500))) ?? true)
    }

    func testMultibyteTextKeepsCharacterCountsExact() {
        let text = repeated("🚀", 7_000)
        let preview = MessageContentPreview.make(text)
        let marker = String(localized: "[... \(2_500) characters hidden ...]", bundle: .module)
        XCTAssertNotNil(preview)
        // head + "\n\n" + marker + "\n\n" + tail
        XCTAssertEqual(preview?.visible.count, 4_000 + 500 + marker.count + 4)
        XCTAssertEqual(preview?.hiddenCharacterCount, 2_500)
        XCTAssertTrue(preview?.visible.hasSuffix(String(text.suffix(500))) ?? false)
    }
}
