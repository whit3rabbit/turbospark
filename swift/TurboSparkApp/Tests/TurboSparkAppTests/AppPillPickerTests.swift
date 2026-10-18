import SwiftUI
import XCTest
@testable import TurboSparkApp

final class AppPillPickerTests: XCTestCase {
    private let values = ["notes", "meeting", "anyAudio"]

    private func neighbor(_ value: String, _ step: Int, _ direction: LayoutDirection = .leftToRight) -> String {
        AppPillPicker<String>.neighbor(of: value, in: values, step: step, layoutDirection: direction)
    }

    func testArrowKeysMoveOneStepAndStopAtTheEnds() {
        XCTAssertEqual(neighbor("notes", 1), "meeting")
        XCTAssertEqual(neighbor("meeting", -1), "notes")
        XCTAssertEqual(neighbor("anyAudio", 1), "anyAudio", "the last pill stays put")
        XCTAssertEqual(neighbor("notes", -1), "notes", "the first pill stays put")
    }

    func testRightToLeftMirrorsTheArrows() {
        XCTAssertEqual(neighbor("notes", -1, .rightToLeft), "meeting", "left moves toward the end when the row is mirrored")
        XCTAssertEqual(neighbor("meeting", 1, .rightToLeft), "notes")
    }

    func testAnUnknownSelectionGoesToTheFirstPill() {
        XCTAssertEqual(neighbor("gone", 1), "notes")
    }
}
