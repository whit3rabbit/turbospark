import Foundation
import WebKit

// `window.turbospark`, the only app-provided script surface of a plugin page
// (ZP-06). The native side below is the authority: the JavaScript shim is a
// convenience, and a hostile page can post to the message handler directly,
// so every rule is enforced here and none relies on the shim.

/// The named refusals a page can see. The `code` is the stable string the
/// page's `catch` receives as `error.code`.
enum PageBridgeErrorCode: String, Equatable {
    case undeclaredMethod = "undeclared_method"
    case permissionDenied = "permission_denied"
    case payloadTooLarge = "payload_too_large"
    case rateLimited = "rate_limited"
    case invalidRequest = "invalid_request"
    case badFrame = "bad_frame"
    case toolNotDeclared = "tool_not_declared"
    case gestureRequired = "gesture_required"
    case tooManyPending = "too_many_pending"
    case toolFailed = "tool_failed"
}

struct PageBridgeError: Error, Equatable {
    let code: PageBridgeErrorCode
    let message: String

    init(_ code: PageBridgeErrorCode, _ message: String) {
        self.code = code
        self.message = message
    }

    /// The string the message handler's reply carries; the shim splits it
    /// back into `code` and `message`.
    var wire: String { "\(code.rawValue): \(message)" }
}

/// A page's tool call was refused by the permission engine or a hook, as
/// opposed to the tool itself failing.
enum PageToolError: Error, Equatable {
    case denied(String)
    case failed(String)
}

/// What a page is and what it declared. Identity is plugin id + surface id +
/// session id (the chat the page was opened from).
struct PluginPageIdentity: Hashable, Sendable {
    let pluginID: String
    let surfaceID: String
    let sessionID: String

    var key: String { "\(pluginID)#\(surfaceID)#\(sessionID)" }
}

/// Session-scoped widget state, keyed by page identity so no page can read
/// another's. Cleared with the page on teardown. In memory only.
@MainActor
final class PluginWidgetStateStore {
    private var states: [String: String] = [:]

    func get(_ identity: PluginPageIdentity) -> String? { states[identity.key] }

    func set(_ identity: PluginPageIdentity, json: String) { states[identity.key] = json }

    func clear(_ identity: PluginPageIdentity) { states[identity.key] = nil }

    var count: Int { states.count }
}

/// The per-page bridge. One instance per live page.
@MainActor
final class PageBridge: NSObject, WKScriptMessageHandlerWithReply {
    static let handlerName = "turbospark"

    typealias ToolInvoker = @Sendable (_ fullToolName: String, _ arguments: [String: Any]) async throws -> String

    let identity: PluginPageIdentity
    let pluginName: String
    let pageTitle: String
    let token: String
    private let capabilities: Set<PluginUiCapability>
    /// Fully qualified `mcp__plugin:<plugin>:<server>__<tool>` names, keyed
    /// by the manifest's `"server/tool"` spelling.
    private let declaredTools: [String: String]
    private let widgetStates: PluginWidgetStateStore
    private let invoker: ToolInvoker
    private let draftSink: @MainActor (String) -> Void
    private let now: () -> Date

    /// Set by the hosting web view on a genuine mouse or key event.
    var lastUserGestureAt: Date?

    private var messageTimes: [Date] = []
    private var toolCallTimes: [Date] = []
    private var followUpTimes: [Date] = []
    private(set) var pendingToolCalls = 0

    init(
        identity: PluginPageIdentity,
        pluginName: String,
        pageTitle: String,
        token: String,
        capabilities: Set<PluginUiCapability>,
        declaredTools: [String: String],
        widgetStates: PluginWidgetStateStore,
        invoker: @escaping ToolInvoker,
        draftSink: @escaping @MainActor (String) -> Void,
        now: @escaping () -> Date = Date.init
    ) {
        self.identity = identity
        self.pluginName = pluginName
        self.pageTitle = pageTitle
        self.token = token
        self.capabilities = capabilities
        self.declaredTools = declaredTools
        self.widgetStates = widgetStates
        self.invoker = invoker
        self.draftSink = draftSink
        self.now = now
    }

    // MARK: - Pure policy

    /// Only the page's own main frame may speak: a subframe (which CSP and
    /// the navigation policy already deny) and a foreign origin are refused
    /// here again so the handler does not depend on either.
    nonisolated static func isTrustedSender(isMainFrame: Bool, scheme: String?, host: String?, token: String) -> Bool {
        isMainFrame && scheme == PluginPageScheme.name && host?.lowercased() == token.lowercased()
    }

    /// Method name to the capability it needs. Unknown names are not in the
    /// table, which is what makes them `undeclared_method`.
    nonisolated static func requiredCapability(for method: String) -> PluginUiCapability? {
        switch method {
        case "invokeTool": return .invokeTool
        case "getWidgetState", "setWidgetState": return .widgetState
        case "sendFollowUp": return .sendFollowUp
        default: return nil
        }
    }

    /// Rolling-window limiter over an injected clock.
    nonisolated static func admit(_ times: inout [Date], limit: Int, now: Date, window: TimeInterval = 60) -> Bool {
        times.removeAll { now.timeIntervalSince($0) >= window }
        guard times.count < limit else { return false }
        times.append(now)
        return true
    }

    nonisolated static func sanitizedLabel(_ text: String) -> String {
        let scalars = text.unicodeScalars.filter { $0.value >= 0x20 && $0.value != 0x7f }
        return String(String.UnicodeScalarView(scalars)).trimmingCharacters(in: .whitespaces).prefix(80).description
    }

    /// Strips control characters except newline and tab, so a page cannot
    /// smuggle escape sequences into the composer.
    nonisolated static func sanitizedDraft(_ text: String) -> String {
        let scalars = text.unicodeScalars.filter { $0.value == 0x0a || $0.value == 0x09 || ($0.value >= 0x20 && $0.value != 0x7f) }
        return String(String.UnicodeScalarView(scalars))
    }

    nonisolated static func truncatedUTF8(_ text: String, maxBytes: Int) -> (text: String, truncated: Bool) {
        guard text.utf8.count > maxBytes else { return (text, false) }
        var bytes = Array(text.utf8.prefix(maxBytes))
        // Back off to a character boundary.
        while !bytes.isEmpty, String(bytes: bytes, encoding: .utf8) == nil { bytes.removeLast() }
        return (String(decoding: bytes, as: UTF8.self), true)
    }

    // MARK: - Dispatch

    /// Validates and runs one message. Order is the security order: sender,
    /// size, shape, method and capability, flood limit, then the method's own
    /// rules.
    func handle(body: Any, isMainFrame: Bool, scheme: String?, host: String?) async -> Result<Any, PageBridgeError> {
        guard Self.isTrustedSender(isMainFrame: isMainFrame, scheme: scheme, host: host, token: token) else {
            return .failure(PageBridgeError(.badFrame, "messages are accepted from the page's main frame only"))
        }
        guard JSONSerialization.isValidJSONObject(body),
            let data = try? JSONSerialization.data(withJSONObject: body)
        else {
            return .failure(PageBridgeError(.invalidRequest, "the message is not a JSON object"))
        }
        guard data.count <= PluginPageLimits.maxMessageBytes else {
            return .failure(PageBridgeError(.payloadTooLarge, "a bridge message is at most \(PluginPageLimits.maxMessageBytes) bytes"))
        }
        guard let message = body as? [String: Any], let method = message["method"] as? String else {
            return .failure(PageBridgeError(.invalidRequest, "the message has no method"))
        }
        guard let needed = Self.requiredCapability(for: method), capabilities.contains(needed) else {
            return .failure(PageBridgeError(.undeclaredMethod, "'\(method.prefix(40))' is not declared by this page's manifest"))
        }
        let params = (message["params"] as? [String: Any]) ?? [:]
        let current = now()
        guard Self.admit(&messageTimes, limit: PluginPageLimits.maxMessagesPerMinute, now: current) else {
            return .failure(PageBridgeError(.rateLimited, "too many bridge messages"))
        }

        switch method {
        case "getWidgetState":
            guard let json = widgetStates.get(identity), let value = try? JSONSerialization.jsonObject(with: Data(json.utf8), options: [.fragmentsAllowed]) else {
                return .success(NSNull())
            }
            return .success(value)
        case "setWidgetState":
            guard let state = params["state"] else {
                return .failure(PageBridgeError(.invalidRequest, "setWidgetState needs a state"))
            }
            guard JSONSerialization.isValidJSONObject(["s": state]),
                let encoded = try? JSONSerialization.data(withJSONObject: state, options: [.fragmentsAllowed]),
                let text = String(data: encoded, encoding: .utf8)
            else {
                return .failure(PageBridgeError(.invalidRequest, "the state is not JSON"))
            }
            guard encoded.count <= PluginPageLimits.maxWidgetStateBytes else {
                return .failure(PageBridgeError(.payloadTooLarge, "widget state is at most \(PluginPageLimits.maxWidgetStateBytes) bytes"))
            }
            widgetStates.set(identity, json: text)
            return .success(true)
        case "sendFollowUp":
            return sendFollowUp(params: params, now: current)
        default:
            return await invokeTool(params: params, now: current)
        }
    }

    private func hasRecentGesture(at current: Date) -> Bool {
        guard let last = lastUserGestureAt else { return false }
        let age = current.timeIntervalSince(last)
        return age >= 0 && age <= PluginPageLimits.userGestureWindowSeconds
    }

    private func sendFollowUp(params: [String: Any], now current: Date) -> Result<Any, PageBridgeError> {
        guard let raw = params["text"] as? String else {
            return .failure(PageBridgeError(.invalidRequest, "sendFollowUp needs text"))
        }
        guard raw.count <= PluginPageLimits.maxFollowUpCharacters else {
            return .failure(PageBridgeError(.payloadTooLarge, "a follow-up is at most \(PluginPageLimits.maxFollowUpCharacters) characters"))
        }
        let text = Self.sanitizedDraft(raw).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            return .failure(PageBridgeError(.invalidRequest, "the follow-up is empty"))
        }
        guard hasRecentGesture(at: current) else {
            return .failure(PageBridgeError(.gestureRequired, "a follow-up needs a recent click or key press in the page"))
        }
        guard Self.admit(&followUpTimes, limit: 6, now: current) else {
            return .failure(PageBridgeError(.rateLimited, "too many follow-ups"))
        }
        // The label names the page and the plugin so the person can see where
        // the text came from. It only fills the draft; nothing is sent.
        let label = AppStrings.localized(
            "From the page \(Self.sanitizedLabel(pageTitle)) of the plugin \(Self.sanitizedLabel(pluginName)):")
        draftSink(label + "\n\n" + text)
        return .success(true)
    }

    private func invokeTool(params: [String: Any], now current: Date) async -> Result<Any, PageBridgeError> {
        guard let requested = params["name"] as? String else {
            return .failure(PageBridgeError(.invalidRequest, "invokeTool needs a tool name"))
        }
        guard let fullName = declaredTools[requested] else {
            return .failure(PageBridgeError(.toolNotDeclared, "'\(requested.prefix(80))' is not a tool this page declares"))
        }
        var arguments: [String: Any] = [:]
        if let rawArguments = params["args"] {
            guard let dict = rawArguments as? [String: Any] else {
                return .failure(PageBridgeError(.invalidRequest, "args must be an object"))
            }
            arguments = dict
        }
        guard hasRecentGesture(at: current) else {
            return .failure(PageBridgeError(.gestureRequired, "a tool call needs a recent click or key press in the page"))
        }
        guard pendingToolCalls < PluginPageLimits.maxPendingToolCalls else {
            return .failure(PageBridgeError(.tooManyPending, "at most \(PluginPageLimits.maxPendingToolCalls) tool calls at once"))
        }
        guard Self.admit(&toolCallTimes, limit: PluginPageLimits.maxToolCallsPerMinute, now: current) else {
            return .failure(PageBridgeError(.rateLimited, "at most \(PluginPageLimits.maxToolCallsPerMinute) tool calls per minute"))
        }
        pendingToolCalls += 1
        defer { pendingToolCalls -= 1 }
        do {
            // The invoker is the ordinary permission path: hooks, deny
            // rules, session approvals and the engine's decision. An ask
            // verdict cannot show a card from here, so it arrives as a
            // refusal (fail closed).
            let output = try await invoker(fullName, arguments)
            let clipped = Self.truncatedUTF8(output, maxBytes: PluginPageLimits.maxToolResultBytes)
            return .success(["output": clipped.text, "truncated": clipped.truncated])
        } catch let error as PageToolError {
            switch error {
            case .denied(let reason):
                return .failure(PageBridgeError(.permissionDenied, String(reason.prefix(300))))
            case .failed(let reason):
                return .failure(PageBridgeError(.toolFailed, String(reason.prefix(300))))
            }
        } catch {
            return .failure(PageBridgeError(.toolFailed, String(error.localizedDescription.prefix(300))))
        }
    }

    // MARK: - WKScriptMessageHandlerWithReply

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage,
        replyHandler: @escaping @MainActor @Sendable (Any?, String?) -> Void
    ) {
        let frame = message.frameInfo
        let body = message.body
        let isMain = frame.isMainFrame
        let scheme = frame.securityOrigin.protocol
        let host = frame.securityOrigin.host
        Task { @MainActor [weak self] in
            guard let self else {
                replyHandler(nil, "\(PageBridgeErrorCode.badFrame.rawValue): the page is closed")
                return
            }
            switch await self.handle(body: body, isMainFrame: isMain, scheme: scheme, host: host) {
            case .success(let value): replyHandler(value, nil)
            case .failure(let error): replyHandler(nil, error.wire)
            }
        }
    }
}

/// The script injected at document start, main frame only. It defines a
/// frozen `window.turbospark` whose methods post to the native handler and
/// turn the `"code: message"` rejection into an `Error` with `.code`.
enum PageBridgeScript {
    static let source = """
    (function () {
      'use strict';
      var handler = window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.turbospark;
      function call(method, params) {
        if (!handler) { return Promise.reject(Object.assign(new Error('bridge unavailable'), { code: 'bridge_unavailable' })); }
        return handler.postMessage({ v: 1, method: method, params: params || {} }).catch(function (e) {
          var text = String(e && e.message !== undefined ? e.message : e);
          var i = text.indexOf(': ');
          var err = new Error(i > 0 ? text.slice(i + 2) : text);
          err.code = i > 0 ? text.slice(0, i) : 'bridge_error';
          throw err;
        });
      }
      var api = Object.freeze({
        version: 1,
        invokeTool: function (name, args) { return call('invokeTool', { name: name, args: args || {} }); },
        getWidgetState: function () { return call('getWidgetState'); },
        setWidgetState: function (state) { return call('setWidgetState', { state: state }); },
        sendFollowUp: function (text) { return call('sendFollowUp', { text: text }); }
      });
      Object.defineProperty(window, 'turbospark', { value: api, writable: false, configurable: false, enumerable: true });
    })();
    """
}

/// Builds the fully qualified MCP tool names a surface declared. Only the
/// plugin's OWN servers can be named: the prefix is fixed to
/// `plugin:<plugin>:`, so a page cannot reach a user, project or other
/// plugin's server by spelling.
enum PluginUiToolNames {
    /// `"server/tool"` (manifest spelling) to `mcp__plugin:<plugin>:<server>__<tool>`.
    /// An entry whose server contains `__` is skipped: the name splitter
    /// divides on `__`, so it could not round-trip.
    static func declaredTools(pluginName: String, surface: PluginUiSurface) -> [String: String] {
        guard surface.declares(.invokeTool) else { return [:] }
        var result: [String: String] = [:]
        for entry in surface.tools {
            let parts = entry.split(separator: "/", maxSplits: 1).map(String.init)
            guard parts.count == 2, !parts[0].contains("__") else { continue }
            result[entry] = "mcp__plugin:\(pluginName):\(parts[0])__\(parts[1])"
        }
        return result
    }
}
