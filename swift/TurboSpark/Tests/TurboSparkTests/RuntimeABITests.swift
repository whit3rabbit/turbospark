import XCTest

@testable import TurboSpark

final class RuntimeABITests: XCTestCase {
    func testStagedHeaderAndArchiveAgree() throws {
        XCTAssertEqual(TurboSparkRuntime.headerABIVersion, TurboSparkRuntime.libraryABIVersion)
        XCTAssertNoThrow(try TurboSparkRuntime.verifyABI())
        XCTAssertNoThrow(try TurboSparkRuntime.verifyABIOnce())
    }

    func testSkewIsRefusedWithTheFix() {
        XCTAssertThrowsError(try TurboSparkRuntime.verify(header: 2, library: 1)) { error in
            let message = (error as? TurboSparkError)?.message ?? ""
            XCTAssertTrue(message.contains("ABI 2"), message)
            XCTAssertTrue(message.contains("ABI 1"), message)
            XCTAssertTrue(message.contains("make swift-lib"), message)
        }
        XCTAssertNoThrow(try TurboSparkRuntime.verify(header: 3, library: 3))
    }

    func testBuildInfoDecodesAndMatchesTheLibrary() throws {
        let info = try TurboSparkRuntime.buildInfo()
        XCTAssertEqual(info.abiVersion, TurboSparkRuntime.libraryABIVersion)
        XCTAssertFalse(info.version.isEmpty)
    }
}
