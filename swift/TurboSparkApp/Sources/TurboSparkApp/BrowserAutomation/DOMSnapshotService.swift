import Foundation
import CoreFoundation
import WebKit
import WebKit

/// The isolated page bridge that creates bounded snapshots and keeps refs tied to live nodes.
@MainActor
public final class DOMSnapshotService: NSObject, WKScriptMessageHandlerWithReply {
    private static let worldName = "com.turbospark.browser.dom-snapshot"
    private static let handlerName = "turboSparkDOMSnapshot"
    private static let contentWorld = WKContentWorld.world(name: worldName)

    private weak var webView: WKWebView?
    private let maximumNodes: Int
    private let maximumBytes: Int
    private var expectedOrigin: String?
    private var currentGeneration = UUID()
    private var hasCommittedDocument = false
    private var isInstalled = true

    public init(
        webView: WKWebView,
        expectedOrigin: String,
        maximumNodes: Int = 500,
        maximumBytes: Int = 64_000
    ) {
        self.webView = webView
        self.expectedOrigin = Self.canonicalOrigin(expectedOrigin)
        self.maximumNodes = max(1, maximumNodes)
        self.maximumBytes = max(512, maximumBytes)
        super.init()

        let controller = webView.configuration.userContentController
        controller.addScriptMessageHandler(self, contentWorld: Self.contentWorld, name: Self.handlerName)
        controller.addUserScript(WKUserScript(
            source: Self.walkerScript,
            injectionTime: .atDocumentEnd,
            forMainFrameOnly: true,
            in: Self.contentWorld
        ))
    }

    /// Invalidates the previous document immediately when a new main-frame load starts.
    @discardableResult
    public func navigationStarted(expectedOrigin: String? = nil) -> UUID {
        if let expectedOrigin {
            self.expectedOrigin = Self.canonicalOrigin(expectedOrigin)
        }
        hasCommittedDocument = false
        currentGeneration = UUID()
        return currentGeneration
    }

    /// Makes the generation active after the engine confirms the new main-frame commit.
    public func navigationCommitted(generation: UUID) {
        guard generation == currentGeneration else { return }
        hasCommittedDocument = true
    }

    /// Revalidates the service lifecycle after asynchronous WebKit calls.
    func isActive(generation: UUID) -> Bool {
        isInstalled && hasCommittedDocument && generation == currentGeneration
    }

    var canPickElement: Bool {
        isInstalled && hasCommittedDocument && expectedOrigin != nil && webView != nil
    }

    /// Releases the registered script handler when its tab is closed or the bridge is replaced.
    public func invalidate() {
        guard isInstalled else { return }
        webView?.configuration.userContentController.removeScriptMessageHandler(
            forName: Self.handlerName,
            contentWorld: Self.contentWorld
        )
        hasCommittedDocument = false
        isInstalled = false
    }

    /// The active document generation, without taking a snapshot. Element
    /// actions only need the generation to address references minted by an
    /// earlier snapshot; the full DOM walk they used to run first cost O(n^2)
    /// in the page and could fail (byte cap) for actions that never needed it.
    /// Throws the same errors `snapshot()` throws before it touches the page.
    public func committedGeneration() throws -> UUID {
        guard isInstalled else { throw DOMSnapshotServiceError.unavailable }
        guard hasCommittedDocument, expectedOrigin != nil, webView != nil else {
            throw DOMSnapshotServiceError.inactiveDocument
        }
        return currentGeneration
    }

    public func snapshot() async throws -> DOMSnapshotEnvelope {
        guard isInstalled else { throw DOMSnapshotServiceError.unavailable }
        guard hasCommittedDocument, expectedOrigin != nil, let webView else {
            throw DOMSnapshotServiceError.inactiveDocument
        }

        let generation = currentGeneration
        let result: Any?
        do {
            result = try await webView.callAsyncJavaScript(
                "return await globalThis.__tsSnapshotV1.snapshot(request)",
                arguments: [
                    "request": [
                        "generation": generation.uuidString,
                        "maximumNodes": maximumNodes,
                        "maximumBytes": maximumBytes
                    ]
                ],
                in: nil,
                contentWorld: Self.contentWorld
            )
        } catch {
            throw DOMSnapshotServiceError.rejectedMessage
        }

        // State may change while WebKit is delivering the asynchronous reply.
        guard isInstalled else { throw DOMSnapshotServiceError.unavailable }
        guard isActive(generation: generation), expectedOrigin != nil else {
            throw DOMSnapshotServiceError.inactiveDocument
        }

        guard let object = result as? [String: Any],
              JSONSerialization.isValidJSONObject(object),
              // Same encoding the JS budget measured: escaped slashes would
              // otherwise inflate a near-budget snapshot past the cap.
              let data = try? JSONSerialization.data(
                  withJSONObject: object, options: [.withoutEscapingSlashes]),
              data.count <= maximumBytes,
              let envelope = try? JSONDecoder().decode(DOMSnapshotEnvelope.self, from: data),
              envelope.version == "ts_snapshot_v1",
              envelope.generation == generation else {
            throw DOMSnapshotServiceError.invalidSnapshot
        }
        return envelope
    }

    public func pickCandidate(at point: CGPoint) async throws -> DOMSnapshotPickCandidate? {
        guard point.x.isFinite, point.y.isFinite, point.x >= 0, point.y >= 0,
              point.x <= 32_768, point.y <= 32_768 else {
            throw DOMSnapshotServiceError.invalidLocator
        }
        guard isInstalled else { throw DOMSnapshotServiceError.unavailable }
        guard canPickElement, let webView else { throw DOMSnapshotServiceError.inactiveDocument }

        let generation = currentGeneration
        let result: Any?
        do {
            result = try await webView.callAsyncJavaScript(
                "return await globalThis.__tsSnapshotV1.pickAt(request)",
                arguments: ["request": [
                    "generation": generation.uuidString,
                    "x": point.x,
                    "y": point.y
                ]],
                in: nil,
                contentWorld: Self.contentWorld
            )
        } catch {
            guard isActive(generation: generation) else {
                throw DOMSnapshotServiceError.inactiveDocument
            }
            throw DOMSnapshotServiceError.rejectedMessage
        }

        guard isActive(generation: generation) else {
            throw DOMSnapshotServiceError.inactiveDocument
        }
        guard let object = result as? [String: Any] else {
            if result == nil || result is NSNull { return nil }
            throw DOMSnapshotServiceError.invalidSnapshot
        }
        if object["error"] as? String == "rejected" {
            throw DOMSnapshotServiceError.rejectedMessage
        }
        guard Self.hasExactKeys(object, ["version", "generation", "reference", "role", "name", "bounds"]),
              JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object),
              data.count <= 4_096,
              let candidate = try? JSONDecoder().decode(DOMSnapshotPickCandidate.self, from: data),
              candidate.version == "ts_element_pick_v1",
              candidate.generation == generation,
              Self.isValidReference(candidate.reference),
              Self.isValidRole(candidate.role),
              candidate.name.utf8.count <= 2_048,
              candidate.bounds.x.isFinite,
              candidate.bounds.y.isFinite,
              candidate.bounds.width.isFinite,
              candidate.bounds.height.isFinite,
              candidate.bounds.width > 0,
              candidate.bounds.height > 0 else {
            throw DOMSnapshotServiceError.invalidSnapshot
        }
        return candidate
    }

    public func resolve(
        locator: DOMSnapshotLocator,
        generation: UUID
    ) async throws -> DOMSnapshotResolvedTarget {
        guard isInstalled else { throw DOMSnapshotServiceError.unavailable }
        guard isActive(generation: generation), expectedOrigin != nil, let webView else {
            if case .reference(let reference) = locator {
                throw BrowserControlError.staleReference(reference: reference)
            }
            throw DOMSnapshotServiceError.inactiveDocument
        }

        let locatorPayload = try Self.locatorPayload(locator)
        let result: Any?
        do {
            result = try await webView.callAsyncJavaScript(
                "return await globalThis.__tsSnapshotV1.resolveLocator(request)",
                arguments: ["request": ["generation": generation.uuidString, "locator": locatorPayload]],
                in: nil,
                contentWorld: Self.contentWorld
            )
        } catch {
            guard isActive(generation: generation) else {
                if case .reference(let reference) = locator {
                    throw BrowserControlError.staleReference(reference: reference)
                }
                throw DOMSnapshotServiceError.inactiveDocument
            }
            throw DOMSnapshotServiceError.rejectedMessage
        }

        guard isActive(generation: generation), expectedOrigin != nil else {
            if case .reference(let reference) = locator {
                throw BrowserControlError.staleReference(reference: reference)
            }
            throw DOMSnapshotServiceError.inactiveDocument
        }
        return try Self.resolvedTarget(from: result, locator: locator)
    }

    public func perform(
        _ action: DOMSnapshotAction,
        on locator: DOMSnapshotLocator,
        generation: UUID
    ) async throws -> DOMSnapshotActionOutcome {
        guard isInstalled else { throw DOMSnapshotServiceError.unavailable }
        guard isActive(generation: generation), expectedOrigin != nil, let webView else {
            if case .reference(let reference) = locator {
                throw BrowserControlError.staleReference(reference: reference)
            }
            throw DOMSnapshotServiceError.inactiveDocument
        }

        let locatorPayload = try Self.locatorPayload(locator)
        let actionPayload = try Self.actionPayload(action)
        let result: Any?
        do {
            result = try await webView.callAsyncJavaScript(
                "return await globalThis.__tsSnapshotV1.perform(request)",
                arguments: ["request": [
                    "generation": generation.uuidString,
                    "locator": locatorPayload,
                    "action": actionPayload
                ]],
                in: nil,
                contentWorld: Self.contentWorld
            )
        } catch {
            guard isActive(generation: generation) else {
                if case .reference(let reference) = locator {
                    throw BrowserControlError.staleReference(reference: reference)
                }
                throw DOMSnapshotServiceError.inactiveDocument
            }
            throw DOMSnapshotServiceError.rejectedMessage
        }

        guard isActive(generation: generation), expectedOrigin != nil else {
            if case .reference(let reference) = locator {
                throw BrowserControlError.staleReference(reference: reference)
            }
            throw DOMSnapshotServiceError.inactiveDocument
        }
        return try Self.actionOutcome(from: result, locator: locator)
    }

    public func resolve(reference: String, generation: UUID) async throws {
        guard isInstalled else {
            throw BrowserControlError.staleReference(reference: reference)
        }
        guard hasCommittedDocument, expectedOrigin != nil, let webView else {
            throw BrowserControlError.staleReference(reference: reference)
        }
        guard generation == currentGeneration else {
            throw BrowserControlError.staleReference(reference: reference)
        }

        let result: Any?
        do {
            result = try await webView.callAsyncJavaScript(
                "return globalThis.__tsSnapshotV1.resolve(request)",
                arguments: ["request": ["reference": reference, "generation": generation.uuidString]],
                in: nil,
                contentWorld: Self.contentWorld
            )
        } catch {
            throw BrowserControlError.staleReference(reference: reference)
        }
        // Invalidation or navigation can race WebKit's reply continuation.
        guard isActive(generation: generation),
              expectedOrigin != nil,
              result as? Bool == true else {
            throw BrowserControlError.staleReference(reference: reference)
        }
    }

    /// Samples an element condition through the isolated bridge. A detached or replaced reference
    /// satisfies `hidden`, but remains stale for the other conditions.
    public func isWaitConditionSatisfied(
        reference: String,
        condition: BrowserElementWaitCondition
    ) async throws -> Bool {
        guard Self.isValidReference(reference) else {
            throw BrowserControlError.staleReference(reference: reference)
        }
        guard isInstalled else { throw DOMSnapshotServiceError.unavailable }
        guard hasCommittedDocument, expectedOrigin != nil, let webView else {
            return condition == .hidden
        }

        let generation = currentGeneration
        let result: Any?
        do {
            result = try await webView.callAsyncJavaScript(
                "return globalThis.__tsSnapshotV1.waitCondition(request)",
                arguments: ["request": [
                    "generation": generation.uuidString,
                    "reference": reference,
                    "condition": condition.rawValue
                ]],
                in: nil,
                contentWorld: Self.contentWorld
            )
        } catch {
            guard isActive(generation: generation) else {
                return condition == .hidden
            }
            throw DOMSnapshotServiceError.rejectedMessage
        }

        guard isInstalled else { throw DOMSnapshotServiceError.unavailable }
        guard isActive(generation: generation) else {
            return condition == .hidden
        }
        guard let satisfied = result as? Bool else {
            throw DOMSnapshotServiceError.invalidSnapshot
        }
        return satisfied
    }

    public func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage,
        replyHandler: @escaping (Any?, String?) -> Void
    ) {
        guard isInstalled,
              message.name == Self.handlerName,
              message.world.name == Self.worldName,
              message.frameInfo.isMainFrame,
              hasCommittedDocument,
              let expectedOrigin,
              Self.canonicalOrigin(message.frameInfo.securityOrigin) == expectedOrigin,
              let webView,
              message.webView === webView,
              let pageURL = webView.url,
              Self.canonicalOrigin(pageURL) == expectedOrigin,
              let body = message.body as? [String: Any],
              body["generation"] as? String == currentGeneration.uuidString else {
            replyHandler(nil, "Snapshot message rejected")
            return
        }

        switch body["type"] as? String {
        case "snapshot":
            guard Self.hasExactKeys(body, ["type", "generation", "snapshot"]),
                  let snapshot = body["snapshot"] as? [String: Any],
                  Self.hasExactKeys(snapshot, ["version", "generation", "nodes", "truncated"]),
                  snapshot["version"] as? String == "ts_snapshot_v1",
                  snapshot["generation"] as? String == currentGeneration.uuidString,
                  JSONSerialization.isValidJSONObject(snapshot),
                  let data = try? JSONSerialization.data(
                      withJSONObject: snapshot, options: [.withoutEscapingSlashes]),
                  data.count <= maximumBytes else {
                replyHandler(nil, "Snapshot message rejected")
                return
            }
        case "resolve":
            guard Self.hasExactKeys(body, ["type", "generation", "locator"]),
                  Self.isValidBridgeLocator(body["locator"]) else {
                replyHandler(nil, "Locator message rejected")
                return
            }
        case "action":
            guard Self.hasExactKeys(body, ["type", "generation", "locator", "action"]),
                  Self.isValidBridgeLocator(body["locator"]),
                  Self.isValidBridgeAction(body["action"]) else {
                replyHandler(nil, "Action message rejected")
                return
            }
        case "pick":
            guard Self.hasExactKeys(body, ["type", "generation", "candidate"]),
                  Self.isValidBridgePickCandidate(body["candidate"], generation: currentGeneration.uuidString) else {
                replyHandler(nil, "Pick message rejected")
                return
            }
        default:
            replyHandler(nil, "Bridge message type rejected")
            return
        }
        replyHandler(["accepted": true], nil)
    }

    private static func locatorPayload(_ locator: DOMSnapshotLocator) throws -> [String: Any] {
        switch locator {
        case .reference(let reference):
            guard isValidReference(reference) else { throw DOMSnapshotServiceError.invalidLocator }
            return ["kind": "reference", "reference": reference]
        case .role(let role, let name):
            guard isValidRole(role), isBoundedNonemptyText(name, maximumBytes: 500) else {
                throw DOMSnapshotServiceError.invalidLocator
            }
            return ["kind": "role", "role": role.lowercased(), "name": name]
        case .visibleText(let text):
            guard isBoundedNonemptyText(text, maximumBytes: 500) else {
                throw DOMSnapshotServiceError.invalidLocator
            }
            return ["kind": "text", "text": text]
        case .cssSelector(let selector):
            guard isBoundedNonemptyText(selector, maximumBytes: 512) else {
                throw DOMSnapshotServiceError.invalidLocator
            }
            return ["kind": "css", "selector": selector]
        }
    }

    private static func actionPayload(_ action: DOMSnapshotAction) throws -> [String: Any] {
        switch action {
        case .click:
            return ["kind": "click"]
        case .type(let text, let submit):
            guard text.utf8.count <= 4096 else { throw DOMSnapshotServiceError.invalidAction }
            return ["kind": "type", "text": text, "submit": submit]
        case .pressKey(let key):
            guard isValidKey(key) else { throw DOMSnapshotServiceError.invalidAction }
            return ["kind": "pressKey", "key": key]
        case .scroll(let direction, let amount):
            guard (1...5000).contains(amount) else { throw DOMSnapshotServiceError.invalidAction }
            return ["kind": "scroll", "direction": direction.rawValue, "amount": amount]
        }
    }

    private static func resolvedTarget(
        from result: Any?,
        locator: DOMSnapshotLocator
    ) throws -> DOMSnapshotResolvedTarget {
        guard let object = result as? [String: Any] else { throw DOMSnapshotServiceError.invalidSnapshot }
        if let error = object["error"] as? String {
            try throwBridgeError(error, locator: locator)
        }
        guard hasExactKeys(object, ["ok", "reference", "ambiguous"]),
              strictBoolean(object["ok"]) == true,
              let reference = object["reference"] as? String,
              isValidReference(reference),
              let ambiguous = strictBoolean(object["ambiguous"]) else {
            throw DOMSnapshotServiceError.invalidSnapshot
        }
        return DOMSnapshotResolvedTarget(reference: reference, isAmbiguous: ambiguous)
    }

    private static func actionOutcome(
        from result: Any?,
        locator: DOMSnapshotLocator
    ) throws -> DOMSnapshotActionOutcome {
        guard let object = result as? [String: Any] else { throw DOMSnapshotServiceError.invalidSnapshot }
        if let error = object["error"] as? String {
            try throwBridgeError(error, locator: locator)
        }
        guard hasExactKeys(object, ["ok", "reference", "ambiguous", "didSubmit"]),
              strictBoolean(object["ok"]) == true,
              let reference = object["reference"] as? String,
              isValidReference(reference),
              let ambiguous = strictBoolean(object["ambiguous"]),
              let didSubmit = strictBoolean(object["didSubmit"]) else {
            throw DOMSnapshotServiceError.invalidSnapshot
        }
        return DOMSnapshotActionOutcome(reference: reference, isAmbiguous: ambiguous, didSubmit: didSubmit)
    }

    private static func throwBridgeError(_ code: String, locator: DOMSnapshotLocator) throws -> Never {
        switch code {
        case "stale_reference":
            if case .reference(let reference) = locator {
                throw BrowserControlError.staleReference(reference: reference)
            }
            throw DOMSnapshotServiceError.targetNotFound
        case "not_found": throw DOMSnapshotServiceError.targetNotFound
        case "locator_limit": throw DOMSnapshotServiceError.locatorLimitExceeded
        case "invalid_locator": throw DOMSnapshotServiceError.invalidLocator
        case "invalid_action": throw DOMSnapshotServiceError.invalidAction
        case "action_rejected": throw DOMSnapshotServiceError.actionRejected
        case "rejected": throw DOMSnapshotServiceError.rejectedMessage
        default: throw DOMSnapshotServiceError.invalidSnapshot
        }
    }

    private static func hasExactKeys(_ object: [String: Any], _ expected: [String]) -> Bool {
        Set(object.keys) == Set(expected)
    }

    private static func isValidReference(_ reference: String) -> Bool {
        reference.utf8.count == 32 && reference.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }

    private static func isBoundedNonemptyText(_ value: String, maximumBytes: Int) -> Bool {
        value.utf8.count <= maximumBytes && !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private static func isValidRole(_ role: String) -> Bool {
        let scalars = Array(role.lowercased().unicodeScalars)
        guard let first = scalars.first, (97...122).contains(first.value), scalars.count <= 64 else { return false }
        return scalars.dropFirst().allSatisfy {
            (97...122).contains($0.value) || (48...57).contains($0.value) || $0.value == 45
        }
    }

    private static func isValidKey(_ key: String) -> Bool {
        let namedKeys: Set<String> = [
            "Enter", "Escape", "Tab", "Backspace", "Delete", "ArrowUp", "ArrowDown",
            "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown", " "
        ]
        if namedKeys.contains(key) { return true }
        return key.count == 1 && key.unicodeScalars.allSatisfy {
            $0.properties.generalCategory != .control
        }
    }

    private static func strictBoolean(_ value: Any?) -> Bool? {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
        return number.boolValue
    }

    private static func strictInteger(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite,
              number.doubleValue.rounded(.towardZero) == number.doubleValue,
              number.doubleValue >= Double(Int.min),
              number.doubleValue <= Double(Int.max) else { return nil }
        return number.intValue
    }

    private static func strictFiniteNumber(_ value: Any?) -> Double? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite else { return nil }
        return number.doubleValue
    }

    private static func isValidBridgePickCandidate(_ value: Any?, generation: String) -> Bool {
        guard let candidate = value as? [String: Any],
              hasExactKeys(candidate, ["version", "generation", "reference", "role", "name", "bounds"]),
              candidate["version"] as? String == "ts_element_pick_v1",
              candidate["generation"] as? String == generation,
              let reference = candidate["reference"] as? String,
              isValidReference(reference),
              let role = candidate["role"] as? String,
              isValidRole(role),
              let name = candidate["name"] as? String,
              name.utf8.count <= 2_048,
              let bounds = candidate["bounds"] as? [String: Any],
              hasExactKeys(bounds, ["x", "y", "width", "height"]),
              let x = strictFiniteNumber(bounds["x"]),
              let y = strictFiniteNumber(bounds["y"]),
              let width = strictFiniteNumber(bounds["width"]),
              let height = strictFiniteNumber(bounds["height"]) else { return false }
        return abs(x) <= 1_000_000 && abs(y) <= 1_000_000 && width > 0 && height > 0
    }

    private static func isValidBridgeLocator(_ value: Any?) -> Bool {
        guard let locator = value as? [String: Any], let kind = locator["kind"] as? String else { return false }
        switch kind {
        case "reference":
            guard hasExactKeys(locator, ["kind", "reference"]),
                  let reference = locator["reference"] as? String else { return false }
            return isValidReference(reference)
        case "role":
            guard hasExactKeys(locator, ["kind", "role", "name"]),
                  let role = locator["role"] as? String,
                  let name = locator["name"] as? String else { return false }
            return isValidRole(role) && isBoundedNonemptyText(name, maximumBytes: 500)
        case "text":
            guard hasExactKeys(locator, ["kind", "text"]),
                  let text = locator["text"] as? String else { return false }
            return isBoundedNonemptyText(text, maximumBytes: 500)
        case "css":
            guard hasExactKeys(locator, ["kind", "selector"]),
                  let selector = locator["selector"] as? String else { return false }
            return isBoundedNonemptyText(selector, maximumBytes: 512)
        default:
            return false
        }
    }

    private static func isValidBridgeAction(_ value: Any?) -> Bool {
        guard let action = value as? [String: Any], let kind = action["kind"] as? String else { return false }
        switch kind {
        case "click":
            return hasExactKeys(action, ["kind"])
        case "type":
            guard hasExactKeys(action, ["kind", "text", "submit"]),
                  let text = action["text"] as? String else { return false }
            return text.utf8.count <= 4096 && strictBoolean(action["submit"]) != nil
        case "pressKey":
            guard hasExactKeys(action, ["kind", "key"]),
                  let key = action["key"] as? String else { return false }
            return isValidKey(key)
        case "scroll":
            guard hasExactKeys(action, ["kind", "direction", "amount"]),
                  let direction = action["direction"] as? String,
                  ["up", "down", "left", "right"].contains(direction),
                  let amount = strictInteger(action["amount"]) else { return false }
            return (1...5000).contains(amount)
        default:
            return false
        }
    }

    private static func canonicalOrigin(_ origin: String) -> String? {
        guard let url = URL(string: origin) else { return nil }
        return canonicalOrigin(url)
    }

    private static func canonicalOrigin(_ url: URL?) -> String? {
        guard let url, let schemeValue = url.scheme else { return nil }
        let scheme = schemeValue.lowercased()
        if scheme == "file" { return "file://" }
        guard let hostValue = url.host, !hostValue.isEmpty else { return nil }
        let host = hostValue.lowercased()
        let port = url.port
        if (scheme == "http" && port == 80) || (scheme == "https" && port == 443) {
            return "\(scheme)://\(host)"
        }
        if let port {
            return "\(scheme)://\(host):\(port)"
        }
        return "\(scheme)://\(host)"
    }

    private static func canonicalOrigin(_ origin: WKSecurityOrigin) -> String? {
        let scheme = origin.`protocol`.lowercased()
        if scheme == "file" { return "file://" }
        let host = origin.host.lowercased()
        guard !host.isEmpty else { return nil }
        if (scheme == "http" && origin.port == 80) || (scheme == "https" && origin.port == 443) {
            return "\(scheme)://\(host)"
        }
        if origin.port > 0 {
            return "\(scheme)://\(host):\(origin.port)"
        }
        return "\(scheme)://\(host)"
    }
}
