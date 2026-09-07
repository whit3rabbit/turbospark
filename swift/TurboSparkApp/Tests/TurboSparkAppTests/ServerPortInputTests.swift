import XCTest
@testable import TurboSparkApp

final class ServerPortInputTests: XCTestCase {
    func testEmptyMeansAutomatic() {
        XCTAssertEqual(try ServerPortInput.parse("").get(), 0)
        XCTAssertEqual(try ServerPortInput.parse("  ").get(), 0)
    }

    func testAValidPortParses() {
        XCTAssertEqual(try ServerPortInput.parse("8080").get(), 8080)
        XCTAssertEqual(try ServerPortInput.parse("65535").get(), 65535)
    }

    func testOutOfRangeIsAnErrorNotAutomatic() {
        // `UInt16("70000") ?? 0` was the old parser: this read as "automatic".
        guard case .failure(let error) = ServerPortInput.parse("70000") else {
            return XCTFail("70000 must not parse")
        }
        XCTAssertTrue(error.message.contains("65535"))
        guard case .failure = ServerPortInput.parse("0") else {
            return XCTFail("an explicit 0 is not a port; empty is the spelling for automatic")
        }
    }

    func testATypoIsAnErrorNotAutomatic() {
        guard case .failure(let error) = ServerPortInput.parse("8O80") else {
            return XCTFail("a letter O must not parse")
        }
        XCTAssertEqual(error.message, "Digits only")
    }
}
