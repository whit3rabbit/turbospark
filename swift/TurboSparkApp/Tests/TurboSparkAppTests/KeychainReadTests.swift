import Foundation
import Security
import XCTest

@testable import TurboSparkApp

/// A Keychain read error must never be mistaken for "nothing stored", or a
/// later save compares "" with the real item and deletes it.
final class KeychainReadTests: XCTestCase {
    func testClassifyDistinguishesFoundNotFoundAndError() {
        XCTAssertEqual(
            KeychainReadResult.classify(status: errSecSuccess, data: Data("sk-1".utf8)), .found("sk-1"))
        XCTAssertEqual(KeychainReadResult.classify(status: errSecItemNotFound, data: nil), .notFound)
        XCTAssertEqual(
            KeychainReadResult.classify(status: errSecInteractionNotAllowed, data: nil),
            .error(errSecInteractionNotAllowed))
        XCTAssertEqual(
            KeychainReadResult.classify(status: errSecAuthFailed, data: nil), .error(errSecAuthFailed))
        // Success without decodable data is an error, not an empty key.
        XCTAssertTrue(KeychainReadResult.classify(status: errSecSuccess, data: nil).isError)
    }

    func testFailedLoadThenSuccessfulReadNeverDeletesTheStoredKey() {
        // Launch read was denied, field started empty, later read succeeds.
        let action = KeychainSecretSync.action(
            current: .found("sk-secret"), input: "", loadFailed: true)
        XCTAssertEqual(action, .adopt("sk-secret"))
        XCTAssertNotEqual(action, .delete)
    }

    func testReadErrorWithEmptyFieldNeverDeletes() {
        XCTAssertEqual(
            KeychainSecretSync.action(current: .error(errSecAuthFailed), input: "", loadFailed: true),
            .none)
        XCTAssertEqual(
            KeychainSecretSync.action(current: .error(errSecAuthFailed), input: "", loadFailed: false),
            .none)
    }

    func testExplicitEditsStillWork() {
        XCTAssertEqual(
            KeychainSecretSync.action(current: .found("old"), input: "new", loadFailed: false),
            .save("new"))
        // The user clearing a key that was successfully loaded is a real delete.
        XCTAssertEqual(
            KeychainSecretSync.action(current: .found("old"), input: "", loadFailed: false), .delete)
        XCTAssertEqual(
            KeychainSecretSync.action(current: .notFound, input: "fresh", loadFailed: false),
            .save("fresh"))
        XCTAssertEqual(
            KeychainSecretSync.action(current: .error(errSecAuthFailed), input: "typed", loadFailed: true),
            .save("typed"))
        XCTAssertEqual(
            KeychainSecretSync.action(current: .found("same"), input: "same", loadFailed: false), .none)
    }
}
