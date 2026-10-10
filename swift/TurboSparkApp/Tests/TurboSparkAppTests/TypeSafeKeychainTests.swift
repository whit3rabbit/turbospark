import Foundation
import XCTest
@testable import TurboSparkApp

@MainActor
final class TypeSafeKeychainTests: XCTestCase {
    func testTypeSafeKeyGenerationProducesValidAndUniqueKeys() {
        let key1 = TypeSafeKeychain.generateKey()
        let key2 = TypeSafeKeychain.generateKey()

        XCTAssertFalse(key1.isEmpty)
        XCTAssertFalse(key2.isEmpty)
        XCTAssertNotEqual(key1, key2)

        // Must be visible ASCII without whitespace
        for byte in key1.utf8 {
            XCTAssertTrue((0x21...0x7e).contains(byte), "Key characters must be visible ASCII without whitespace")
        }

        // Expected format starts with ok_ (from OpenKindAPIKey) or sk- (from ServerAPIKeyGenerator)
        XCTAssertTrue(key1.hasPrefix("ok_") || key1.hasPrefix("sk-"))
    }

    func testStartTypeSafeServerFailsClosedWhenKeyIsEmpty() async {
        let model = AppModel()
        model.typeSafePort = "18080"
        model.typeSafeAPIKeyInput = ""

        await model.startTypeSafeServer()

        XCTAssertNil(model.typeSafeServer, "Server must not start when API key is empty")
        XCTAssertEqual(model.typeSafeError, "TypeSafe API key is required to start the daemon.")
    }

    func testStartTypeSafeServerFailsClosedWhenKeyIsWhitespace() async {
        let model = AppModel()
        model.typeSafePort = "18080"
        model.typeSafeAPIKeyInput = "   \n\t  "

        await model.startTypeSafeServer()

        XCTAssertNil(model.typeSafeServer, "Server must not start when API key is only whitespace")
        XCTAssertEqual(model.typeSafeError, "TypeSafe API key is required to start the daemon.")
    }

    func testEmptyKeyDisablesStartButtonCondition() {
        let model = AppModel()
        model.typeSafePort = "18080"
        model.typeSafeAPIKeyInput = ""

        let isDisabledWhenEmpty = model.typeSafeBusy
            || (model.typeSafeServer == nil
                && (model.typeSafeBaseURL == nil
                    || model.typeSafeAPIKeyInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))

        XCTAssertTrue(isDisabledWhenEmpty, "Start button must be disabled when API key is empty")

        model.typeSafeAPIKeyInput = "ok_validcredential"
        let isDisabledWhenValid = model.typeSafeBusy
            || (model.typeSafeServer == nil
                && (model.typeSafeBaseURL == nil
                    || model.typeSafeAPIKeyInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))

        XCTAssertFalse(isDisabledWhenValid, "Start button must be enabled when port and API key are valid")
    }
}
