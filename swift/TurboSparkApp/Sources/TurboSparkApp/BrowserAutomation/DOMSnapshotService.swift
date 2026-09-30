import Foundation
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
    case unavailable
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
        if (type === "password") return "Password field";
        const ariaLabel = normalize(element.getAttribute("aria-label"));
        if (ariaLabel) return ariaLabel;
        if (element.labels && element.labels.length) {
          const label = normalize(Array.from(element.labels).map(node => node.textContent || "").join(" "));
          if (label) return label;
        }
        if (["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName)) return "";
        const labelledBy = normalize((element.getAttribute("aria-labelledby") || "")
          .split(/\s+/).map(id => document.getElementById(id)?.textContent || "").join(" "));
        if (labelledBy) return labelledBy;
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
      const referenceFor = element => {
        let reference = referencesByNode.get(element);
        if (!reference) {
          const bytes = crypto.getRandomValues(new Uint8Array(16));
          reference = Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
          referencesByNode.set(element, reference);
          nodesByReference.set(reference, new WeakRef(element));
        }
        return reference;
      };
      const boundsFor = element => {
        const rect = element.getBoundingClientRect();
        return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
      };
      const isActionable = element => element.matches(actionableSelector);
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
              reference: actionable ? referenceFor(element) : null
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
          const weakNode = nodesByReference.get(request.reference);
          const node = weakNode && weakNode.deref();
          return Boolean(node && node.isConnected && referencesByNode.get(node) === request.reference);
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
                arguments: ["request": ["reference": reference]],
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
              body["type"] as? String == "snapshot",
              body["generation"] as? String == currentGeneration.uuidString,
              let snapshot = body["snapshot"] as? [String: Any],
              snapshot["version"] as? String == "ts_snapshot_v1",
              snapshot["generation"] as? String == currentGeneration.uuidString,
              JSONSerialization.isValidJSONObject(snapshot),
              let data = try? JSONSerialization.data(withJSONObject: snapshot),
              data.count <= maximumBytes else {
            replyHandler(nil, "Snapshot message rejected")
            return
        }
        replyHandler(["accepted": true], nil)
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
