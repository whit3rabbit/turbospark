import WebKit
import XCTest
@testable import TurboSparkApp

/// A scheduler the test fires by hand, so no test waits on a real clock.
@MainActor
private final class ManualScheduler {
    private(set) var scheduledDelays: [TimeInterval] = []
    private var fires: [Int: @MainActor () -> Void] = [:]
    private var next = 0

    var scheduler: SitePermissionScheduler {
        { [self] delay, fire in
            scheduledDelays.append(delay)
            let id = next
            next += 1
            fires[id] = fire
            return { [self] in fires[id] = nil }
        }
    }

    var activeTimerCount: Int { fires.count }

    func fireAll() {
        let pending = fires
        fires.removeAll()
        for (_, fire) in pending.sorted(by: { $0.key < $1.key }) { fire() }
    }
}

private final class DecisionLog: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Bool] = []
    var all: [Bool] { lock.lock(); defer { lock.unlock() }; return values }
    func append(_ value: Bool) { lock.lock(); values.append(value); lock.unlock() }
    func decision() -> SingleShotPermissionDecision {
        SingleShotPermissionDecision { [self] in append($0) }
    }
}

@MainActor
final class SitePermissionTests: XCTestCase {
    private let site = BrowserOrigin(origin: "https://example.com")!
    private let other = BrowserOrigin(origin: "https://other.example")!

    private struct Harness {
        let policy: SitePermissionPolicy
        let store: SitePermissionStore
        let timers: ManualScheduler
    }

    private final class Env {
        var owner: BrowserTabOwner? = .user
        var active: BrowserTabID?
    }

    private func makeHarness(
        tabID: BrowserTabID,
        env: Env = Env(),
        store storeOverride: SitePermissionStore? = nil
    ) -> (Harness, Env) {
        let store = storeOverride ?? SitePermissionStore(fileURL: nil)
        env.active = env.active ?? tabID
        let timers = ManualScheduler()
        let policy = SitePermissionPolicy(
            store: store,
            ownerProvider: { _ in env.owner },
            activeTabProvider: { env.active },
            scheduler: timers.scheduler
        )
        return (Harness(policy: policy, store: store, timers: timers), env)
    }

    private func request(
        _ tabID: BrowserTabID,
        origin: BrowserOrigin?,
        top: BrowserOrigin? = nil,
        classes: [SitePermissionClass] = [.camera]
    ) -> SitePermissionRequest {
        SitePermissionRequest(
            tabID: tabID, requestingOrigin: origin, topLevelOrigin: top ?? origin, classes: classes)
    }

    // MARK: - Classification

    func testClassificationTableCoversOnlyGovernedClasses() {
        XCTAssertEqual(SitePermissionClassifier.classes(forCapture: .camera), [.camera])
        XCTAssertEqual(SitePermissionClassifier.classes(forCapture: .microphone), [.microphone])
        XCTAssertEqual(
            SitePermissionClassifier.classes(forCapture: .cameraAndMicrophone), [.camera, .microphone])
        XCTAssertEqual(SitePermissionClassifier.geolocation, [.geolocation])
    }

    func testUnknownCaptureTypeIsNotClassifiedAndPolicyDeniesIt() throws {
        let unknown = try XCTUnwrap(WKMediaCaptureType(rawValue: 99))
        let classes = SitePermissionClassifier.classes(forCapture: unknown)
        XCTAssertEqual(classes, [])

        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site, classes: classes), decision: log.decision())
        XCTAssertEqual(log.all, [false])
        XCTAssertTrue(h.policy.pendingPrompts.isEmpty)
        XCTAssertEqual(h.store.auditLog.first?.reason, .unclassified)
    }

    func testOriginMappingIsExactSchemeHostPort() {
        let apex = SitePermissionClassifier.origin(scheme: "https", host: "example.com", port: 0)
        let www = SitePermissionClassifier.origin(scheme: "https", host: "www.example.com", port: 0)
        XCTAssertNotEqual(apex, www)
        XCTAssertEqual(apex?.canonicalString, "https://example.com")
        // Default port and trailing dot are the same site; a custom port is not.
        XCTAssertEqual(
            SitePermissionClassifier.origin(scheme: "https", host: "example.com", port: 443), apex)
        XCTAssertEqual(
            SitePermissionClassifier.origin(scheme: "https", host: "example.com.", port: 0), apex)
        XCTAssertNotEqual(
            SitePermissionClassifier.origin(scheme: "https", host: "example.com", port: 8443), apex)
        XCTAssertNil(SitePermissionClassifier.origin(scheme: "file", host: "", port: 0))
        XCTAssertNil(SitePermissionClassifier.origin(scheme: "about", host: "blank", port: 0))
        XCTAssertEqual(
            SitePermissionClassifier.origin(scheme: "http", host: "::1", port: 3000)?.canonicalString,
            "http://[::1]:3000")
    }

    // MARK: - Agent tabs

    func testAgentTabIsDeniedEvenWithAStoredAllowRecord() {
        let tab = BrowserTabID()
        let (h, env) = makeHarness(tabID: tab)
        h.store.setVerdict(.allow, origin: site, permissionClass: .camera)
        env.owner = .agent
        let log = DecisionLog()

        h.policy.decide(request(tab, origin: site), decision: log.decision())

        XCTAssertEqual(log.all, [false])
        XCTAssertTrue(h.policy.pendingPrompts.isEmpty, "an agent tab never gets a bubble")
        XCTAssertEqual(h.store.auditLog.first?.decision, .autoDeny)
        XCTAssertEqual(h.store.auditLog.first?.reason, .agentTab)
        XCTAssertNotEqual(h.store.auditLog.first?.decision, .allowRecord, "no record lookup")
    }

    func testClosedTabIsDenied() {
        let tab = BrowserTabID()
        let (h, env) = makeHarness(tabID: tab)
        env.owner = nil
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: log.decision())
        XCTAssertEqual(log.all, [false])
        XCTAssertEqual(h.store.auditLog.first?.reason, .tabClosed)
    }

    func testBackgroundTabIsDeniedWithoutBubble() {
        let tab = BrowserTabID()
        let env = Env()
        env.active = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab, env: env)
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: log.decision())
        XCTAssertEqual(log.all, [false])
        XCTAssertTrue(h.policy.pendingPrompts.isEmpty)
        XCTAssertEqual(h.store.auditLog.first?.reason, .background)
    }

    func testOpaqueOriginIsDenied() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: nil), decision: log.decision())
        XCTAssertEqual(log.all, [false])
        XCTAssertEqual(h.store.auditLog.first?.canonicalOrigin, BrowserPermissionDiagnosticEvent.opaqueOrigin)
        XCTAssertEqual(h.store.auditLog.first?.reason, .unknownOrigin)
    }

    // MARK: - Records

    func testStoredAllowAnswersWithoutAsking() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        h.store.setVerdict(.allow, origin: site, permissionClass: .camera)
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: log.decision())
        XCTAssertEqual(log.all, [true])
        XCTAssertTrue(h.policy.pendingPrompts.isEmpty)
        XCTAssertEqual(h.store.auditLog.first?.decision, .allowRecord)
    }

    func testStoredDenyAnswersWithoutAskingAndWinsOverAllowForSibling() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        h.store.setVerdict(.allow, origin: site, permissionClass: .camera)
        h.store.setVerdict(.deny, origin: site, permissionClass: .microphone)
        let log = DecisionLog()
        h.policy.decide(
            request(tab, origin: site, classes: [.camera, .microphone]), decision: log.decision())
        XCTAssertEqual(log.all, [false])
        XCTAssertTrue(h.policy.pendingPrompts.isEmpty)
    }

    func testRecordsAreKeyedPerOriginAndClass() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        h.store.setVerdict(.allow, origin: site, permissionClass: .camera)
        let log = DecisionLog()
        // Same class, sibling subdomain: a miss, so it asks.
        let www = BrowserOrigin(origin: "https://www.example.com")!
        h.policy.decide(request(tab, origin: www), decision: log.decision())
        XCTAssertTrue(log.all.isEmpty)
        XCTAssertEqual(h.policy.pendingPrompts.count, 1)
    }

    func testPartialRecordPromptsOnlyForTheMissingClass() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        h.store.setVerdict(.allow, origin: site, permissionClass: .camera)
        let log = DecisionLog()
        h.policy.decide(
            request(tab, origin: site, classes: [.camera, .microphone]), decision: log.decision())
        XCTAssertEqual(h.policy.pendingPrompts.first?.classes, [.microphone])
        let promptID = h.policy.pendingPrompts[0].id
        XCTAssertTrue(h.policy.answer(promptID: promptID, .allowOnce))
        XCTAssertEqual(log.all, [true])
    }

    // MARK: - Prompt, answers, timeout

    func testMissAsksAndTimeoutDeniesWithAuditAndNoRecord() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: log.decision())

        XCTAssertEqual(h.policy.pendingPrompts.count, 1)
        XCTAssertEqual(h.timers.scheduledDelays, [sitePermissionPromptTimeout])
        XCTAssertEqual(sitePermissionPromptTimeout, 30)
        XCTAssertTrue(log.all.isEmpty)

        h.timers.fireAll()

        XCTAssertEqual(log.all, [false])
        XCTAssertTrue(h.policy.pendingPrompts.isEmpty)
        XCTAssertTrue(h.store.records.isEmpty)
        XCTAssertEqual(h.store.auditLog.first?.decision, .autoDeny)
        XCTAssertEqual(h.store.auditLog.first?.reason, .timeout)
    }

    func testAnswersWriteRecordsOnlyForSiteChoices() {
        let cases: [(SitePermissionAnswer, Bool, SitePermissionVerdict?, SitePermissionAuditDecision)] = [
            (.allowOnce, true, nil, .allowOnce),
            (.allowSite, true, .allow, .allowSite),
            (.denyOnce, false, nil, .denyOnce),
            (.denySite, false, .deny, .denySite),
        ]
        for (answer, expectedAllow, expectedRecord, expectedAudit) in cases {
            let tab = BrowserTabID()
            let (h, _) = makeHarness(tabID: tab)
            let log = DecisionLog()
            h.policy.decide(request(tab, origin: site), decision: log.decision())
            XCTAssertTrue(h.policy.answer(promptID: h.policy.pendingPrompts[0].id, answer))
            XCTAssertEqual(log.all, [expectedAllow], "\(answer)")
            XCTAssertEqual(h.store.verdict(for: site, permissionClass: .camera), expectedRecord, "\(answer)")
            XCTAssertEqual(h.store.auditLog.first?.decision, expectedAudit, "\(answer)")
            XCTAssertEqual(h.timers.activeTimerCount, 0, "the countdown is cancelled")
        }
    }

    func testSavedAllowIsUsedOnTheNextRequestAndDeleteAsksAgain() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        let first = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: first.decision())
        h.policy.answer(promptID: h.policy.pendingPrompts[0].id, .allowSite)

        let second = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: second.decision())
        XCTAssertEqual(second.all, [true])

        let record = h.store.records[0]
        XCTAssertTrue(h.store.removeRecord(id: record.id))
        XCTAssertEqual(h.store.auditLog.first?.decision, .recordRemoved)

        let third = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: third.decision())
        XCTAssertTrue(third.all.isEmpty, "asks again after the record is deleted")
        XCTAssertEqual(h.policy.pendingPrompts.count, 1)
    }

    func testAnswerAfterResolutionIsIgnoredAndHandlerRunsOnce() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: log.decision())
        let id = h.policy.pendingPrompts[0].id
        h.timers.fireAll()
        XCTAssertFalse(h.policy.answer(promptID: id, .allowSite), "a late Allow must not grant")
        XCTAssertEqual(log.all, [false])
        XCTAssertTrue(h.store.records.isEmpty)
    }

    func testSingleShotDecisionCallsHandlerOnce() {
        let log = DecisionLog()
        let decision = log.decision()
        XCTAssertTrue(decision.resolve(true))
        XCTAssertFalse(decision.resolve(false))
        XCTAssertFalse(decision.resolve(true))
        XCTAssertEqual(log.all, [true])
    }

    func testLeakedSingleShotDecisionDeniesOnDeinit() {
        let log = DecisionLog()
        autoreleasepool { _ = log.decision() }
        XCTAssertEqual(log.all, [false])

        // A resolved one does not call again when it is released.
        let log2 = DecisionLog()
        autoreleasepool { _ = log2.decision().resolve(true) }
        XCTAssertEqual(log2.all, [true])
    }

    // MARK: - Coalescing and flood

    func testIdenticalRequestsCoalesceOntoOneBubble() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        let a = DecisionLog(), b = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: a.decision())
        h.policy.decide(request(tab, origin: site), decision: b.decision())
        XCTAssertEqual(h.policy.pendingPrompts.count, 1)
        XCTAssertEqual(h.timers.scheduledDelays.count, 1)
        h.policy.answer(promptID: h.policy.pendingPrompts[0].id, .allowOnce)
        XCTAssertEqual(a.all, [true])
        XCTAssertEqual(b.all, [true])
    }

    func testDifferentRequestWhilePendingIsDeniedAsAFlood() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        let a = DecisionLog(), b = DecisionLog(), c = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: a.decision())
        h.policy.decide(request(tab, origin: other), decision: b.decision())
        h.policy.decide(request(tab, origin: site, classes: [.microphone]), decision: c.decision())
        XCTAssertEqual(h.policy.pendingPrompts.count, 1)
        XCTAssertEqual(b.all, [false])
        XCTAssertEqual(c.all, [false])
        XCTAssertTrue(a.all.isEmpty, "the first question is still open")
        XCTAssertEqual(h.store.auditLog.first?.reason, .tooManyRequests)
    }

    // MARK: - Cancellation paths

    func testCancelPromptsDeniesWithReasonForEveryEdgeCase() {
        let reasons: [SitePermissionAutoDenyReason] = [
            .navigation, .tabClosed, .processTerminated, .ownershipChanged, .background,
        ]
        for reason in reasons {
            let tab = BrowserTabID()
            let (h, _) = makeHarness(tabID: tab)
            let log = DecisionLog()
            h.policy.decide(request(tab, origin: site), decision: log.decision())
            h.policy.cancelPrompts(for: tab, reason: reason)
            XCTAssertEqual(log.all, [false], "\(reason)")
            XCTAssertTrue(h.policy.pendingPrompts.isEmpty)
            XCTAssertEqual(h.store.auditLog.first?.reason, reason)
            XCTAssertEqual(h.timers.activeTimerCount, 0)
        }
    }

    func testSwitchingTabsDeniesThePendingBubble() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: log.decision())
        h.policy.cancelPromptsOutside(activeTabID: tab)
        XCTAssertEqual(h.policy.pendingPrompts.count, 1, "still the active tab")
        h.policy.cancelPromptsOutside(activeTabID: BrowserTabID())
        XCTAssertEqual(log.all, [false])
    }

    func testOwnershipFlipWhilePendingDeniesEvenAnAllowAnswer() {
        let tab = BrowserTabID()
        let (h, env) = makeHarness(tabID: tab)
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: log.decision())
        env.owner = .agent
        XCTAssertTrue(h.policy.answer(promptID: h.policy.pendingPrompts[0].id, .allowSite))
        XCTAssertEqual(log.all, [false])
        XCTAssertTrue(h.store.records.isEmpty, "no record from a flipped tab")
        XCTAssertEqual(h.store.auditLog.first?.reason, .ownershipChanged)
    }

    func testAnswerAfterTabLeavesForegroundDenies() {
        let tab = BrowserTabID()
        let (h, env) = makeHarness(tabID: tab)
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: log.decision())
        env.active = BrowserTabID()
        h.policy.answer(promptID: h.policy.pendingPrompts[0].id, .allowOnce)
        XCTAssertEqual(log.all, [false])
    }

    // MARK: - Prompt content

    func testPromptNamesBothOriginsOnlyWhenTheyDiffer() {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        h.policy.decide(request(tab, origin: site, top: other), decision: DecisionLog().decision())
        let prompt = h.policy.pendingPrompts[0]
        XCTAssertEqual(prompt.requestingOrigin, site, "keyed by the frame that asked")
        XCTAssertEqual(prompt.topLevelOrigin, other)
        XCTAssertTrue(prompt.showsPageOrigin)

        let same = SitePermissionPrompt(
            id: UUID(), tabID: tab, requestingOrigin: site, topLevelOrigin: site,
            classes: [.camera], expiresAt: Date(), duration: 30)
        XCTAssertFalse(same.showsPageOrigin)
    }

    func testFullStoreDowngradesSiteAnswerToOnce() throws {
        let tab = BrowserTabID()
        let (h, _) = makeHarness(tabID: tab)
        for index in 0..<SitePermissionStore.maximumRecords {
            let origin = try XCTUnwrap(BrowserOrigin(origin: "https://h\(index).example"))
            XCTAssertTrue(h.store.setVerdict(.deny, origin: origin, permissionClass: .camera))
        }
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: log.decision())
        h.policy.answer(promptID: h.policy.pendingPrompts[0].id, .allowSite)
        XCTAssertEqual(log.all, [true])
        XCTAssertNil(h.store.verdict(for: site, permissionClass: .camera))
        XCTAssertEqual(h.store.auditLog.first?.decision, .allowOnce)
        XCTAssertEqual(h.store.records.count, SitePermissionStore.maximumRecords)
    }

    // MARK: - Store

    private func tempFileURL() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("site-permissions-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir.appendingPathComponent("browser_site_permissions.json")
    }

    func testStoreRoundTripPersistsRecordsAndAudit() throws {
        let url = try tempFileURL()
        let store = SitePermissionStore(fileURL: url)
        store.setVerdict(.allow, origin: site, permissionClass: .camera)
        store.recordAudit(permissionClass: .camera, decision: .allowSite, origin: site)

        let reloaded = SitePermissionStore(fileURL: url)
        XCTAssertEqual(reloaded.verdict(for: site, permissionClass: .camera), .allow)
        XCTAssertNil(reloaded.verdict(for: site, permissionClass: .microphone))
        XCTAssertEqual(reloaded.auditLog.first?.decision, .allowSite)
        XCTAssertEqual(reloaded.auditLog.first?.canonicalOrigin, "https://example.com")
    }

    func testCorruptStoreFileIsQuarantinedAndEveryRequestMisses() throws {
        let url = try tempFileURL()
        try Data("{ not json".utf8).write(to: url)
        let store = SitePermissionStore(fileURL: url)
        XCTAssertNil(store.verdict(for: site, permissionClass: .camera))
        XCTAssertTrue(store.records.isEmpty)
        let siblings = try FileManager.default.contentsOfDirectory(atPath: url.deletingLastPathComponent().path)
        XCTAssertTrue(siblings.contains { $0.hasPrefix("corrupt-") || $0.contains("corrupt") },
                      "the unreadable file is kept, not overwritten: \(siblings)")
        AppJSONStore.clearLastReadError()

        // A miss in an agent tab denies, in a user tab asks.
        let tab = BrowserTabID()
        let (h, env) = makeHarness(tabID: tab, store: store)
        let log = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: log.decision())
        XCTAssertEqual(h.policy.pendingPrompts.count, 1)
        env.owner = .agent
        let agent = DecisionLog()
        h.policy.decide(request(tab, origin: site), decision: agent.decision())
        XCTAssertEqual(agent.all, [false])
    }

    func testRecordsAreCappedAndCanonicalKeysCollapseEquivalentHosts() throws {
        let store = SitePermissionStore(fileURL: nil)
        let apex = try XCTUnwrap(BrowserOrigin(origin: "https://example.com"))
        let apexDefaultPort = try XCTUnwrap(BrowserOrigin(origin: "https://example.com:443"))
        let dotted = try XCTUnwrap(BrowserOrigin(origin: "https://example.com."))
        let www = try XCTUnwrap(BrowserOrigin(origin: "https://www.example.com"))
        store.setVerdict(.allow, origin: apex, permissionClass: .camera)
        store.setVerdict(.deny, origin: apexDefaultPort, permissionClass: .camera)
        store.setVerdict(.deny, origin: dotted, permissionClass: .camera)
        XCTAssertEqual(store.records.count, 1)
        XCTAssertEqual(store.verdict(for: apex, permissionClass: .camera), .deny)
        store.setVerdict(.allow, origin: www, permissionClass: .camera)
        XCTAssertEqual(store.records.count, 2, "www and apex stay distinct")
    }

    func testAuditLogIsBoundedAndNewestFirst() {
        let store = SitePermissionStore(fileURL: nil)
        for _ in 0..<(SitePermissionStore.maximumAuditEntries + 25) {
            store.recordAudit(permissionClass: .camera, decision: .denyOnce, origin: site)
        }
        store.recordAudit(permissionClass: .microphone, decision: .allowOnce, origin: site)
        XCTAssertEqual(store.auditLog.count, SitePermissionStore.maximumAuditEntries)
        XCTAssertEqual(store.auditLog.first?.permissionClass, .microphone)
    }

    func testAuditEventCarriesNoPathOrQuery() throws {
        let origin = try XCTUnwrap(BrowserOrigin(url: URL(string: "https://example.com/secret/path?token=abc")!))
        let event = BrowserPermissionDiagnosticEvent(
            permissionClass: .camera, decision: .denyOnce, origin: origin, timestamp: Date())
        let json = try String(data: JSONEncoder().encode(event), encoding: .utf8) ?? ""
        XCTAssertFalse(json.contains("secret"))
        XCTAssertFalse(json.contains("token"))
        // JSONEncoder escapes slashes, so match on the host.
        XCTAssertTrue(json.contains("example.com"), json)
    }

    func testSearchFiltersByOriginAndClass() throws {
        let store = SitePermissionStore(fileURL: nil)
        store.setVerdict(.allow, origin: site, permissionClass: .camera)
        store.setVerdict(.deny, origin: other, permissionClass: .microphone)
        XCTAssertEqual(SitePermissionStore.filter(store.records, query: "EXAMPLE.com").map(\.origin), ["https://example.com"])
        XCTAssertEqual(SitePermissionStore.filter(store.records, query: "micro").count, 1)
        XCTAssertEqual(SitePermissionStore.filter(store.records, query: "  ").count, 2)
        XCTAssertEqual(SitePermissionStore.filter(store.records, query: "nomatch").count, 0)
    }

    // MARK: - Engine seam

    func testEngineDeniesAgentTabAndAsksInUserTabAndTabCloseDenies() throws {
        let timers = ManualScheduler()
        let tabStore = BrowserTabStore()
        let permissionStore = SitePermissionStore(fileURL: nil)
        let engine = WebKitBrowserEngine(
            tabStore: tabStore,
            authorizeNavigation: { _ in .allow },
            sitePermissionStore: permissionStore,
            sitePermissionScheduler: timers.scheduler
        )
        let userTab = engine.createTab(owner: .user)
        let agentTab = engine.createTab(owner: .agent, select: false)

        let agentLog = DecisionLog()
        engine.handleSitePermissionRequest(
            request(agentTab, origin: site), decision: agentLog.decision())
        XCTAssertEqual(agentLog.all, [false])
        XCTAssertTrue(engine.pendingSitePermissions.isEmpty)

        let userLog = DecisionLog()
        engine.handleSitePermissionRequest(
            request(userTab, origin: site), decision: userLog.decision())
        XCTAssertEqual(engine.pendingSitePermissions.count, 1, "the pane reads this to show the bubble")
        XCTAssertTrue(userLog.all.isEmpty)

        try engine.closeTab(userTab)
        XCTAssertEqual(userLog.all, [false], "closing the tab denies the pending request")
        XCTAssertTrue(engine.pendingSitePermissions.isEmpty)
        XCTAssertEqual(permissionStore.auditLog.first?.reason, .tabClosed)
    }

    func testEngineUserTakeoverDeniesAgentTabAfterwardsEvenWithRecord() throws {
        let tabStore = BrowserTabStore()
        let permissionStore = SitePermissionStore(fileURL: nil)
        permissionStore.setVerdict(.allow, origin: site, permissionClass: .camera)
        let engine = WebKitBrowserEngine(
            tabStore: tabStore,
            authorizeNavigation: { _ in .allow },
            sitePermissionStore: permissionStore
        )
        let tab = engine.createTab(owner: .agent)
        let before = DecisionLog()
        engine.handleSitePermissionRequest(request(tab, origin: site), decision: before.decision())
        XCTAssertEqual(before.all, [false], "stored allow ignored while the agent owns the tab")

        engine.prepareForDirectUserInput(in: tab)
        let after = DecisionLog()
        engine.handleSitePermissionRequest(request(tab, origin: site), decision: after.decision())
        XCTAssertEqual(after.all, [true], "after takeover the user's record applies")
    }

    // MARK: - ArtifactWebView is untouched

    func testArtifactWebViewDoesNotGainPermissionHooks() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp/Generation/ArtifactWebView.swift")
        let text = try String(contentsOf: sources, encoding: .utf8)
        XCTAssertFalse(text.contains("SitePermission"))
        XCTAssertFalse(text.contains("requestMediaCapturePermission"))
    }
}
