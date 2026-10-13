import Foundation
import Security
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

    // MARK: - Keychain storage is opt-in

    func testKeychainStorageIsOffUntilTheUserTurnsItOn() {
        let suite = "TypeSafeKeychainTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }

        XCTAssertFalse(TypeSafeKeychain.isKeychainEnabled(defaults),
                       "A fresh install must not store the TypeSafe key in the Keychain")
        TypeSafeKeychain.setKeychainEnabled(true, defaults: defaults)
        XCTAssertTrue(TypeSafeKeychain.isKeychainEnabled(defaults))
        TypeSafeKeychain.setKeychainEnabled(false, defaults: defaults)
        XCTAssertFalse(TypeSafeKeychain.isKeychainEnabled(defaults))
    }

    func testInitialKeyWithoutOptInIsAFreshSessionKey() {
        // Does not touch the Keychain: a stored key would be identical on
        // both calls, and a failed read would come back empty.
        let first = TypeSafeKeychain.initialKey(useKeychain: false)
        let second = TypeSafeKeychain.initialKey(useKeychain: false)

        XCTAssertFalse(first.key.isEmpty, "The daemon must still start authenticated")
        XCTAssertFalse(first.loadFailed, "Nothing was read, so nothing can have failed to load")
        XCTAssertNotEqual(first.key, second.key, "Without the opt-in every launch gets its own key")
    }

    func testKeychainFailureNamesTheSystemReasonAndCode() {
        let text = TypeSafeKeychain.describe(status: errSecNoDefaultKeychain)
        XCTAssertTrue(text.contains("(\(errSecNoDefaultKeychain))"), text)
        XCTAssertGreaterThan(text.count, "(\(errSecNoDefaultKeychain))".count + 8,
                             "The system's own wording should come before the code")

        let message = AppModel.keychainFailureMessage(
            "Could not save the TypeSafe API key to Keychain.", status: errSecNoDefaultKeychain)
        XCTAssertTrue(message.hasPrefix("Could not save the TypeSafe API key to Keychain."), message)
        XCTAssertTrue(message.hasSuffix(text), message)
    }
}
