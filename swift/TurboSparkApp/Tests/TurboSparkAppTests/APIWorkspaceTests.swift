import XCTest

@testable import TurboSparkApp

final class APIWorkspaceTests: XCTestCase {
    func testSelectorOrderAndBoundaryNavigation() {
        XCTAssertEqual(APIWorkspaceTab.allCases, [.text, .image, .typeSafe])
        XCTAssertEqual(APIWorkspaceTab.text.moved(-1), .text)
        XCTAssertEqual(APIWorkspaceTab.text.moved(1), .image)
        XCTAssertEqual(APIWorkspaceTab.image.moved(1), .typeSafe)
        XCTAssertEqual(APIWorkspaceTab.typeSafe.moved(1), .typeSafe)
        XCTAssertEqual(APIWorkspaceTab.typeSafe.moved(-1), .image)
        XCTAssertEqual(AppModel.AppNavigationSection.server.title, "API")
    }

    func testSelectorHasLocalizedTitles() {
        XCTAssertEqual(APIWorkspaceTab.text.title, "Text")
        XCTAssertEqual(APIWorkspaceTab.image.title, "Image")
        XCTAssertEqual(APIWorkspaceTab.typeSafe.title, "TypeSafe")
    }
}
