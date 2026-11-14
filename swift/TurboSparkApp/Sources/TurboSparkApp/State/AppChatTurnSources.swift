import Foundation

/// How a turn came to know a source (OC-09).
enum TurnSourceKind: String, Equatable, Sendable {
    /// A row of a successful web search result.
    case searchResult
    /// A page the turn fetched in full. Wins over `searchResult` when both
    /// name the same URL: a page the reply read is the stronger citation.
    case fetchedPage
}

/// One web source a turn relied on: title, URL and domain.
struct TurnSource: Identifiable, Equatable, Sendable {
    /// The dedupe key, so SwiftUI identity and dedupe agree.
    var id: String { key }
    let key: String
    var kind: TurnSourceKind
    var title: String
    /// The URL exactly as the tool saw it (trimmed). Always http or https.
    let url: String
    /// Lowercased host without a leading `www.`.
    let domain: String
    /// The URL path without query or fragment; empty for a site root.
    let displayPath: String

    /// The only schemes a source may carry. A model-written search row can
    /// hold `file:`, `javascript:` or `data:` text, and the popover's open
    /// action is a user click that navigates a browser tab, so anything
    /// else is dropped at collection, not at the click.
    static func isOpenableScheme(_ scheme: String?) -> Bool {
        guard let scheme = scheme?.lowercased() else { return false }
        return scheme == "http" || scheme == "https"
    }

    /// Builds a source from raw tool text, or nil for an unusable URL
    /// (wrong scheme, no host, whitespace inside).
    init?(kind: TurnSourceKind, title: String?, rawURL: String) {
        let trimmed = rawURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              !trimmed.contains(where: { $0.isWhitespace }),
              let components = URLComponents(string: trimmed),
              Self.isOpenableScheme(components.scheme),
              let host = components.host?.lowercased(), !host.isEmpty
        else { return nil }

        let domain = host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
        var path = components.path
        if path == "/" { path = "" }

        // The key drops the fragment and a trailing slash and folds scheme
        // and host case. The query stays: two queries are two pages.
        var keyPath = components.path
        while keyPath.count > 1 && keyPath.hasSuffix("/") { keyPath.removeLast() }
        if keyPath == "/" { keyPath = "" }
        var key = "\(components.scheme!.lowercased())://\(host)"
        if let port = components.port { key += ":\(port)" }
        key += keyPath
        if let query = components.query, !query.isEmpty { key += "?\(query)" }

        let cleanTitle = title?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        self.key = key
        self.kind = kind
        self.title = cleanTitle.isEmpty ? domain : cleanTitle
        self.url = trimmed
        self.domain = domain
        self.displayPath = path
    }

    /// Whether `title` came from a search row rather than falling back to the
    /// domain, so a later duplicate can replace a bare fallback.
    var hasRealTitle: Bool { title != domain }
}

/// The deduplicated sources of one transcript turn.
///
/// **Derived at render time from the turn's archived tool calls and
/// results, never stored.** A stored copy would be a second source of truth
/// that has to be written, migrated through the tolerant message decoder and
/// kept in step with edits and variants; deriving gives old chats their
/// chips for free (docs/designs/opencode2-parity/OC-09-sources-panel.md,
/// "Resolved decisions"). Search output is the markdown `formatMarkdown`
/// writes (`N. [title](url)`), and a fetch's URL comes from the call
/// arguments because its output is the page body.
struct TurnSources: Equatable, Sendable {
    /// Hard cap on listed sources per turn; the rest only count.
    static let maximumSources = 50

    private(set) var sources: [TurnSource] = []
    /// Distinct sources beyond `maximumSources`.
    private(set) var overflowCount = 0
    private var seen = Set<String>()

    var isEmpty: Bool { sources.isEmpty }
    /// Every distinct source the turn used, listed or not.
    var totalCount: Int { sources.count + overflowCount }

    /// A domain with its sources, in first-seen order.
    struct Group: Equatable, Identifiable, Sendable {
        var id: String { domain }
        let domain: String
        var sources: [TurnSource]
    }

    var groups: [Group] {
        var order: [String] = []
        var byDomain: [String: [TurnSource]] = [:]
        for source in sources {
            if byDomain[source.domain] == nil { order.append(source.domain) }
            byDomain[source.domain, default: []].append(source)
        }
        return order.map { Group(domain: $0, sources: byDomain[$0] ?? []) }
    }

    // MARK: Collection

    private static let searchTools: Set<String> = ["websearch", "web_search", "search_web"]
    private static let fetchTools: Set<String> = ["webfetch", "web_fetch", "fetch_url", "read_url_content"]
    private static let fetchURLKeys = ["url", "uri", "Url", "URL", "href"]

    /// Cheap pre-check so a turn with no web call never parses a result.
    static func hasWebCall(in messages: [AppChatMessage]) -> Bool {
        messages.contains { message in
            message.toolCalls.contains { webKind(of: $0.name) != nil }
        }
    }

    private static func webKind(of name: String) -> TurnSourceKind? {
        let lowered = name.lowercased()
        if searchTools.contains(lowered) { return .searchResult }
        if fetchTools.contains(lowered) { return .fetchedPage }
        return nil
    }

    /// Collects the sources of `messages` (one turn, transcript order).
    ///
    /// Only calls with a SUCCESSFUL result count: a failed or interrupted
    /// call read nothing. The synthetic Tavily "direct answer" row is
    /// skipped, because its URL is the provider's home page rather than
    /// something the reply drew on.
    static func collect(in messages: [AppChatMessage]) -> TurnSources {
        var collected = TurnSources()
        for message in messages where message.role == .assistant {
            for call in message.toolCalls {
                guard let kind = webKind(of: call.name),
                      let result = message.toolResults.first(where: { $0.callID == call.id }),
                      !result.isError
                else { continue }
                switch kind {
                case .searchResult:
                    for (title, url) in searchRows(in: result.output) {
                        collected.add(TurnSource(kind: .searchResult, title: title, rawURL: url))
                    }
                case .fetchedPage:
                    let raw = fetchURLKeys.lazy.compactMap { call.arguments[$0] }.first
                    if let raw {
                        collected.add(TurnSource(kind: .fetchedPage, title: nil, rawURL: raw))
                    }
                }
            }
        }
        return collected
    }

    private mutating func add(_ candidate: TurnSource?) {
        guard let candidate else { return }
        if let index = sources.firstIndex(where: { $0.key == candidate.key }) {
            // Same page again: keep the first position, upgrade in place.
            if candidate.kind == .fetchedPage { sources[index].kind = .fetchedPage }
            if !sources[index].hasRealTitle && candidate.hasRealTitle {
                sources[index].title = candidate.title
            }
            return
        }
        guard !seen.contains(candidate.key) else { return }
        seen.insert(candidate.key)
        if sources.count < Self.maximumSources {
            sources.append(candidate)
        } else {
            overflowCount += 1
        }
    }

    // MARK: Search output parsing

    /// Rows of `formatMarkdown`: `1. [title](url)`. A title may hold
    /// brackets and a URL may hold parentheses, so the split is on the LAST
    /// `](` of the line, and the URL is what follows up to the final `)`.
    static func searchRows(in output: String) -> [(title: String, url: String)] {
        var rows: [(String, String)] = []
        for rawLine in output.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard let dot = line.firstIndex(of: "."),
                  dot != line.startIndex,
                  line[line.startIndex..<dot].allSatisfy(\.isNumber)
            else { continue }
            let rest = line[line.index(after: dot)...].drop(while: { $0 == " " })
            guard rest.hasPrefix("["), rest.hasSuffix(")"),
                  let split = rest.range(of: "](", options: .backwards)
            else { continue }
            let title = String(rest[rest.index(after: rest.startIndex)..<split.lowerBound])
            let url = String(rest[split.upperBound..<rest.index(before: rest.endIndex)])
            // Tavily's synthetic "direct answer" is not a page.
            if title == "Tavily Direct Answer", url.lowercased().contains("tavily.com") { continue }
            rows.append((title, url))
        }
        return rows
    }
}

extension TurnSources {
    /// The id of the row that carries the chip for `turnMessageIDs`: the
    /// final assistant message with content and no tool calls, the same row a
    /// collapsed turn keeps visible. A turn that ends on a tool call or on
    /// nothing has no reply row and so no chip.
    static func replyRowID(in messages: [AppChatMessage]) -> UUID? {
        let answer = messages.last {
            $0.role == .assistant && (!$0.content.isEmpty || !$0.toolCalls.isEmpty)
        }
        guard let answer, answer.toolCalls.isEmpty else { return nil }
        return answer.id
    }
}
