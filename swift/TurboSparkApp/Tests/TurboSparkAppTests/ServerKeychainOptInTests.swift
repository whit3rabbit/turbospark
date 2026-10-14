import Foundation
import XCTest
@testable import TurboSparkApp

/// The server key's Keychain opt-in. None of these touch the real Keychain:
/// the existence probe is injected, and the preference lives in a throwaway
/// defaults suite.
final class ServerKeychainOptInTests: XCTestCase {
    private func freshDefaults() -> (UserDefaults, () -> Void) {
        let suite = "ServerKeychainOptInTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        return (defaults, { defaults.removePersistentDomain(forName: suite) })
    }

    func testNewInstallWithNoStoredKeyStaysOffAndRecordsNothing() {
        let (defaults, cleanup) = freshDefaults()
        defer { cleanup() }

        XCTAssertFalse(ServerKeychain.resolveOptIn(defaults: defaults, itemExists: { false }),
                       "A new install must not use the Keychain")
        XCTAssertNil(defaults.object(forKey: ServerKeychain.optInDefaultsKey),
                     "Nothing is recorded, so a later launch can still notice a stored key")
    }

    func testAnExistingStoredKeyIsKeptAndTheChoiceRecorded() {
        let (defaults, cleanup) = freshDefaults()
        defer { cleanup() }

        // An earlier version stored a key the person typed. Dropping it would
        // start their server with no authentication.
        XCTAssertTrue(ServerKeychain.resolveOptIn(defaults: defaults, itemExists: { true }))
        XCTAssertTrue(ServerKeychain.isKeychainEnabled(defaults), "The consent is recorded once")
    }

    func testAnExplicitOffIsNotOverriddenByAnItemThatStillExists() {
        let (defaults, cleanup) = freshDefaults()
        defer { cleanup() }
        ServerKeychain.setKeychainEnabled(false, defaults: defaults)

        var probed = false
        let enabled = ServerKeychain.resolveOptIn(defaults: defaults, itemExists: { probed = true; return true })

        XCTAssertFalse(enabled, "Someone who turned it off stays off")
        XCTAssertFalse(probed, "A recorded choice never reaches the Keychain at all")
    }

    func testAnExplicitOnSkipsTheProbe() {
        let (defaults, cleanup) = freshDefaults()
        defer { cleanup() }
        ServerKeychain.setKeychainEnabled(true, defaults: defaults)

        var probed = false
        XCTAssertTrue(ServerKeychain.resolveOptIn(defaults: defaults, itemExists: { probed = true; return false }))
        XCTAssertFalse(probed)
    }
}
