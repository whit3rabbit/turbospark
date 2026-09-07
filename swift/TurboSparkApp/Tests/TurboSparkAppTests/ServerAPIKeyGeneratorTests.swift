import XCTest
@testable import TurboSparkApp

final class ServerAPIKeyGeneratorTests: XCTestCase {
    func testAKeyIsSkPlusALowercasedUUID() {
        let key = ServerAPIKeyGenerator.generate()
        XCTAssertTrue(key.hasPrefix("sk-"))
        // Parsing as a UUID pins the shape: only 32 hex digits and hyphens
        // after the prefix survive this, so a dropped segment or a stray
        // character fails here. The round-trip through lowercased() is
        // separate, because UUID parsing accepts uppercase.
        XCTAssertNotNil(UUID(uuidString: String(key.dropFirst(3))))
        XCTAssertEqual(key, key.lowercased())
    }

    func testTwoKeysDiffer() {
        XCTAssertNotEqual(ServerAPIKeyGenerator.generate(), ServerAPIKeyGenerator.generate())
    }
}
