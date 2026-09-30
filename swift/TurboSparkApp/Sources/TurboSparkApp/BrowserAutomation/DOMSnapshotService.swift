import Foundation
import CoreFoundation
import WebKit

public struct DOMSnapshotBounds: Codable, Equatable, Sendable {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

public struct DOMSnapshotNode: Codable, Equatable, Sendable {
    public let role: String
    public let name: String
    public let parentIndex: Int?
    public let bounds: DOMSnapshotBounds?
    public let reference: String?

    public init(
        role: String,
        name: String,
        parentIndex: Int?,
        bounds: DOMSnapshotBounds?,
        reference: String?
    ) {
        self.role = role
        self.name = name
        self.parentIndex = parentIndex
        self.bounds = bounds
        self.reference = reference
    }
}

public struct DOMSnapshotEnvelope: Codable, Equatable, Sendable {
    public let version: String
    public let generation: UUID
    public let nodes: [DOMSnapshotNode]
    public let truncated: Bool

    public init(version: String, generation: UUID, nodes: [DOMSnapshotNode], truncated: Bool) {
        self.version = version
        self.generation = generation
        self.nodes = nodes
        self.truncated = truncated
    }
}

public enum DOMSnapshotServiceError: Error, Equatable {
    case inactiveDocument
    case rejectedMessage
    case invalidSnapshot
    case invalidLocator
    case targetNotFound
    case locatorLimitExceeded
    case invalidAction
    case actionRejected
    case unavailable
}

public enum DOMSnapshotLocator: Sendable, Equatable {
    case reference(String)
    case role(role: String, name: String)
    case visibleText(String)
    case cssSelector(String)
}

public enum DOMSnapshotScrollDirection: String, Sendable, Equatable {
    case up
    case down
    case left
    case right
}

public enum DOMSnapshotAction: Sendable, Equatable {
    case click
    case type(text: String, submit: Bool)
    case pressKey(key: String)
    case scroll(direction: DOMSnapshotScrollDirection, amount: Int)
}

public struct DOMSnapshotResolvedTarget: Sendable, Equatable {
    public let reference: String
    public let isAmbiguous: Bool

    public init(reference: String, isAmbiguous: Bool) {
        self.reference = reference
        self.isAmbiguous = isAmbiguous
    }
}

public struct DOMSnapshotActionOutcome: Sendable, Equatable {
    public let reference: String
    public let isAmbiguous: Bool
    public let didSubmit: Bool

    public init(reference: String, isAmbiguous: Bool, didSubmit: Bool) {
        self.reference = reference
        self.isAmbiguous = isAmbiguous
        self.didSubmit = didSubmit
    }
}

/// The isolated page bridge that creates bounded snapshots and keeps refs tied to live nodes.
@MainActor
public final class DOMSnapshotService: NSObject, WKScriptMessageHandlerWithReply {
    private static let worldName = "com.turbospark.browser.dom-snapshot"
    private static let handlerName = "turboSparkDOMSnapshot"
    private static let contentWorld = WKContentWorld.world(name: worldName)
    private static let walkerScript = #"""
    (() => {
      const referencesByNode = new WeakMap();
      const nodesByReference = new Map();
      const maximumLocatorNodes = 5000;
      const actionableSelector = [
        "button", "a[href]", "input:not([type='hidden'])", "textarea", "select",
        "[role='button']", "[role='link']", "[role='checkbox']", "[role='radio']",
        "[role='switch']", "[role='tab']", "[tabindex]:not([tabindex='-1'])"
      ].join(",");
      const excludedTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE"]);

      const normalize = value => (value || "").replace(/\s+/g, " ").trim().slice(0, 500);
      const roleFor = element => {
        const explicit = element.getAttribute("role");
        if (explicit) return explicit.trim().split(/\s+/)[0].toLowerCase();
        const tag = element.tagName.toLowerCase();
        if (tag === "a" && element.hasAttribute("href")) return "link";
        if (tag === "button" || tag === "summary") return "button";
        if (tag === "textarea") return "textbox";
        if (tag === "select") return element.multiple ? "listbox" : "combobox";
        if (tag === "input") {
          const type = (element.getAttribute("type") || "text").toLowerCase();
          if (type === "checkbox") return "checkbox";
          if (type === "radio") return "radio";
          if (type === "range") return "slider";
          if (["button", "submit", "reset", "image"].includes(type)) return "button";
          return "textbox";
        }
        if (/^h[1-6]$/.test(tag)) return "heading";
        if (tag === "main") return "main";
        if (tag === "nav") return "navigation";
        if (tag === "form") return "form";
        if (tag === "dialog") return "dialog";
        if (tag === "p") return "paragraph";
        return "generic";
      };
      const accessibleName = element => {
        const type = element.tagName === "INPUT" ? (element.getAttribute("type") || "text").toLowerCase() : "";
        const labelledBy = normalize((element.getAttribute("aria-labelledby") || "")
          .split(/\s+/).map(id => {
            const label = document.getElementById(id);
            if (!label || ["INPUT", "TEXTAREA", "SELECT"].includes(label.tagName)) return "";
            return label.textContent || "";
          }).join(" "));
        if (labelledBy) return labelledBy;
        const ariaLabel = normalize(element.getAttribute("aria-label"));
        if (ariaLabel) return ariaLabel;
        if (type === "password") return "Password field";
        if (element.labels && element.labels.length) {
          const label = normalize(Array.from(element.labels).map(node => node.textContent || "").join(" "));
          if (label) return label;
        }
        if (["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName)) return "";
        const alt = normalize(element.getAttribute("alt"));
        if (alt) return alt;
        const title = normalize(element.getAttribute("title"));
        if (title) return title;
        const tag = element.tagName;
        if (["BUTTON", "A", "SUMMARY", "OPTION", "DIALOG"].includes(tag)) {
          return normalize(element.textContent);
        }
        return normalize(Array.from(element.childNodes)
          .filter(node => node.nodeType === Node.TEXT_NODE)
          .map(node => node.nodeValue || "").join(" "));
      };
      const visible = element => {
        const style = getComputedStyle(element);
        if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) return false;
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      };
      const referenceFor = (element, generation) => {
        let reference = referencesByNode.get(element);
        if (!reference) {
          const bytes = crypto.getRandomValues(new Uint8Array(16));
          reference = Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
          referencesByNode.set(element, reference);
        }
        nodesByReference.set(reference, { node: new WeakRef(element), generation });
        return reference;
      };
      const boundsFor = element => {
        const rect = element.getBoundingClientRect();
        return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
      };
      const isActionable = element => element.matches(actionableSelector);
      const hasExactKeys = (value, expected) => {
        if (!value || typeof value !== "object" || Array.isArray(value)) return false;
        const actual = Object.keys(value).sort();
        return actual.length === expected.length && actual.every((key, index) => key === expected.slice().sort()[index]);
      };
      const validReference = value => typeof value === "string" && /^[0-9a-f]{32}$/.test(value);
      const validBoundedText = (value, maximum) =>
        typeof value === "string" && value.length > 0 && new TextEncoder().encode(value).length <= maximum;
      const validKey = value => {
        const namedKeys = new Set([
          "Enter", "Escape", "Tab", "Backspace", "Delete", "ArrowUp", "ArrowDown",
          "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown", " "
        ]);
        return typeof value === "string" && new TextEncoder().encode(value).length <= 32
          && (namedKeys.has(value) || [...value].length === 1);
      };
      const validLocator = locator => {
        if (!locator || typeof locator !== "object" || Array.isArray(locator)) return false;
        if (locator.kind === "reference") {
          return hasExactKeys(locator, ["kind", "reference"]) && validReference(locator.reference);
        }
        if (locator.kind === "role") {
          return hasExactKeys(locator, ["kind", "role", "name"])
            && typeof locator.role === "string" && /^[a-z][a-z0-9-]{0,63}$/.test(locator.role)
            && validBoundedText(locator.name, 500);
        }
        if (locator.kind === "text") {
          return hasExactKeys(locator, ["kind", "text"]) && validBoundedText(locator.text, 500);
        }
        if (locator.kind === "css") {
          return hasExactKeys(locator, ["kind", "selector"]) && validBoundedText(locator.selector, 512);
        }
        return false;
      };
      const validAction = action => {
        if (!action || typeof action !== "object" || Array.isArray(action)) return false;
        if (action.kind === "click") return hasExactKeys(action, ["kind"]);
        if (action.kind === "type") {
          return hasExactKeys(action, ["kind", "text", "submit"])
            && typeof action.text === "string" && new TextEncoder().encode(action.text).length <= 4096
            && typeof action.submit === "boolean";
        }
        if (action.kind === "pressKey") {
          return hasExactKeys(action, ["kind", "key"]) && validKey(action.key);
        }
        if (action.kind === "scroll") {
          return hasExactKeys(action, ["kind", "direction", "amount"])
            && ["up", "down", "left", "right"].includes(action.direction)
            && Number.isInteger(action.amount) && action.amount >= 1 && action.amount <= 5000;
        }
        return false;
      };
      const findTarget = request => {
        if (!request || typeof request.generation !== "string" || !validLocator(request.locator)) {
          return { error: "invalid_locator" };
        }
        const locator = request.locator;
        if (locator.kind === "reference") {
          const record = nodesByReference.get(locator.reference);
          const node = record && record.node.deref();
          if (!node || record.generation !== request.generation || !node.isConnected
              || referencesByNode.get(node) !== locator.reference) {
            return { error: "stale_reference" };
          }
          return { element: node, ambiguous: false };
        }

        let matches;
        if (locator.kind === "role") {
          const role = locator.role.toLowerCase();
          const name = normalize(locator.name);
          matches = element => roleFor(element) === role && accessibleName(element) === name;
        } else if (locator.kind === "text") {
          const text = normalize(locator.text);
          matches = element => normalize(element.innerText || "") === text;
        } else {
          try {
            document.documentElement.matches(locator.selector);
          } catch (_) {
            return { error: "invalid_locator" };
          }
          matches = element => element.matches(locator.selector);
        }

        const walker = document.createTreeWalker(document, NodeFilter.SHOW_ELEMENT);
        let first = null;
        let ambiguous = false;
        let visited = 0;
        let element = walker.nextNode();
        while (element) {
          if (visited >= maximumLocatorNodes) return { error: "locator_limit" };
          visited += 1;
          if (visible(element) && matches(element)) {
            if (first) {
              ambiguous = true;
              break;
            }
            first = element;
          }
          element = walker.nextNode();
        }
        return first ? { element: first, ambiguous } : { error: "not_found" };
      };
      const targetResult = (request, target) => ({
        ok: true,
        reference: referenceFor(target.element, request.generation),
        ambiguous: target.ambiguous === true
      });
      const dispatchKey = (element, key) => {
        const init = { key, bubbles: true, cancelable: true, view: window };
        element.dispatchEvent(new KeyboardEvent("keydown", init));
        if (key.length === 1) element.dispatchEvent(new KeyboardEvent("keypress", init));
        element.dispatchEvent(new KeyboardEvent("keyup", init));
      };
      const performAction = (element, action) => {
        if (!validAction(action)) return { error: "invalid_action" };
        if (action.kind === "click") {
          if (element instanceof HTMLElement) element.click();
          else element.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
          return { didSubmit: false };
        }
        if (action.kind === "type") {
          if (!(element instanceof HTMLElement)) return { error: "action_rejected" };
          const form = action.submit ? (element.form || element.closest("form")) : null;
          if (action.submit && !form) return { error: "action_rejected" };
          const tag = element.tagName;
          if (tag === "INPUT") {
            const inputType = (element.getAttribute("type") || "text").toLowerCase();
            if (!["text", "search", "email", "url", "tel", "password", "number"].includes(inputType)) {
              return { error: "action_rejected" };
            }
            element.focus();
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
            setter.call(element, action.text);
          } else if (tag === "TEXTAREA") {
            element.focus();
            const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
            setter.call(element, action.text);
          } else if (element.isContentEditable) {
            element.focus();
            element.textContent = action.text;
          } else {
            return { error: "action_rejected" };
          }
          element.dispatchEvent(new InputEvent("input", {
            bubbles: true,
            cancelable: false,
            inputType: "insertText",
            data: action.text
          }));
          element.dispatchEvent(new Event("change", { bubbles: true }));
          if (!action.submit) return { didSubmit: false };
          dispatchKey(element, "Enter");
          form.requestSubmit();
          return { didSubmit: true };
        }
        if (action.kind === "pressKey") {
          if (element instanceof HTMLElement) element.focus();
          dispatchKey(element, action.key);
          return { didSubmit: false };
        }
        const delta = action.amount * (action.direction === "up" || action.direction === "left" ? -1 : 1);
        const deltaX = action.direction === "left" || action.direction === "right" ? delta : 0;
        const deltaY = action.direction === "up" || action.direction === "down" ? delta : 0;
        element.dispatchEvent(new WheelEvent("wheel", {
          deltaX, deltaY, bubbles: true, cancelable: true, view: window
        }));
        if (element === document.scrollingElement || element === document.documentElement || element === document.body) {
          window.scrollBy(deltaX, deltaY);
        } else if (element instanceof HTMLElement) {
          element.scrollBy(deltaX, deltaY);
        }
        return { didSubmit: false };
      };
      const version = "ts_snapshot_v1";

      globalThis.__tsSnapshotV1 = {
        async snapshot(request) {
          const nodes = [];
          let truncated = false;
          let visitedElements = 0;
          const envelopeFor = (candidateNodes, isTruncated) => ({
            version,
            generation: request.generation,
            nodes: candidateNodes,
            truncated: isTruncated
          });
          const fits = candidateNodes => new TextEncoder().encode(
            JSON.stringify(envelopeFor(candidateNodes, true))
          ).length <= request.maximumBytes;
          const visit = (element, parentIndex, depth) => {
            if (truncated) return;
            visitedElements += 1;
            if (visitedElements > request.maximumNodes * 4) {
              truncated = true;
              return;
            }
            if (nodes.length >= request.maximumNodes || depth > 32) {
              truncated = true;
              return;
            }
            if (excludedTags.has(element.tagName) || !visible(element)) return;

            const actionable = isActionable(element);
            const candidate = {
              role: roleFor(element),
              name: accessibleName(element),
              parentIndex,
              bounds: boundsFor(element),
              reference: actionable ? referenceFor(element, request.generation) : null
            };
            if (!fits(nodes.concat(candidate))) {
              truncated = true;
              return;
            }
            const index = nodes.length;
            nodes.push(candidate);
            for (const child of element.children) visit(child, index, depth + 1);
          };

          if (document.documentElement) visit(document.documentElement, null, 0);
          const result = envelopeFor(nodes, truncated);
          const acknowledgement = await webkit.messageHandlers.turboSparkDOMSnapshot.postMessage({
            type: "snapshot",
            generation: request.generation,
            snapshot: result
          });
          if (!acknowledgement || acknowledgement.accepted !== true) {
            throw new Error("Snapshot bridge rejected the page message");
          }
          return result;
        },
        resolve(request) {
          const record = nodesByReference.get(request.reference);
          const node = record && record.node.deref();
          return Boolean(node && record.generation === request.generation && node.isConnected
            && referencesByNode.get(node) === request.reference);
        },
        async resolveLocator(request) {
          const acknowledgement = await webkit.messageHandlers.turboSparkDOMSnapshot.postMessage({
            type: "resolve",
            generation: request.generation,
            locator: request.locator
          });
          if (!acknowledgement || acknowledgement.accepted !== true) return { error: "rejected" };
          const target = findTarget(request);
          return target.error ? { error: target.error } : targetResult(request, target);
        },
        async perform(request) {
          const acknowledgement = await webkit.messageHandlers.turboSparkDOMSnapshot.postMessage({
            type: "action",
            generation: request.generation,
            locator: request.locator,
            action: request.action
          });
          if (!acknowledgement || acknowledgement.accepted !== true) return { error: "rejected" };
          const target = findTarget(request);
          if (target.error) return { error: target.error };
          const reference = referenceFor(target.element, request.generation);
          const actionResult = performAction(target.element, request.action);
          if (actionResult.error) return { error: actionResult.error };
          return {
            ok: true,
            reference,
            ambiguous: target.ambiguous === true,
            didSubmit: actionResult.didSubmit === true
          };
        }
      };
    })();
    """#

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
              let data = try? JSONSerialization.data(withJSONObject: object),
              data.count <= maximumBytes,
              let envelope = try? JSONDecoder().decode(DOMSnapshotEnvelope.self, from: data),
              envelope.version == "ts_snapshot_v1",
              envelope.generation == generation else {
            throw DOMSnapshotServiceError.invalidSnapshot
        }
        return envelope
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
                  let data = try? JSONSerialization.data(withJSONObject: snapshot),
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
