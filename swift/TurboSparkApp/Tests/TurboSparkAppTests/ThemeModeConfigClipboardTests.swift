import XCTest
@testable import TurboSparkApp

final class ThemeModeConfigClipboardTests: XCTestCase {
    func testAConfigRoundTripsThroughTheClipboardText() throws {
        var config = ThemeModeConfig.defaultDark
        config.accentHex = "#123456"
        let json = String(decoding: try JSONEncoder().encode(config), as: UTF8.self)
        XCTAssertEqual(ThemeModeConfig.fromClipboardJSON(json), config)
    }

    func testTextThatIsNotAThemeDecodesToNothing() {
        // The view used to apply a PRESET on this path, so the failure was
        // indistinguishable from a successful import of something else.
        XCTAssertNil(ThemeModeConfig.fromClipboardJSON("hello"))
        XCTAssertNil(ThemeModeConfig.fromClipboardJSON("{\"accentHex\": 1}"))
        XCTAssertNil(ThemeModeConfig.fromClipboardJSON(""))
    }
}
