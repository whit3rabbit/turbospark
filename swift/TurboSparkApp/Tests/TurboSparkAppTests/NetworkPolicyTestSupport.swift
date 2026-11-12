import XCTest
@testable import TurboSparkApp

extension AppNetworkPolicy {
    /// The behavior before Settings > Network existed: every public domain
    /// runs and every non-public host is refused. Suites that test the
    /// private-host refusal path pin this so the shipped defaults (Ask,
    /// localhost allowed) do not change what they measure.
    static let strictLegacy = AppNetworkPolicy(mode: .allowAll, allowLocalhost: false)
}

/// Sets `AppNetworkPolicyStore.current` for one test and restores it after.
/// The store is process-wide, and `swift test` runs suites serially here.
class NetworkPolicyPinnedTestCase: XCTestCase {
    var pinnedNetworkPolicy: AppNetworkPolicy { .strictLegacy }
    private var savedNetworkPolicy = AppNetworkPolicy()

    override func setUp() {
        super.setUp()
        savedNetworkPolicy = AppNetworkPolicyStore.current
        AppNetworkPolicyStore.current = pinnedNetworkPolicy
    }

    override func tearDown() {
        AppNetworkPolicyStore.current = savedNetworkPolicy
        super.tearDown()
    }
}
