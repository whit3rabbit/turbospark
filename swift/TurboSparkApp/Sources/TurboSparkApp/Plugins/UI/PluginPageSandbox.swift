import Foundation

// The pure half of the plugin page sandbox (ZP-06). Everything that decides
// WHAT a plugin page may read, load or say lives here as plain functions so
// each guard can be tested, and mutation-checked, without a window. The
// WebKit wiring (`PluginPageWebView`, `PluginPageSchemeHandler`) only applies
// these decisions.

/// Hard numeric limits for plugin pages. One place, so a test can assert the
/// value it claims and the docs can name it.
enum PluginPageLimits {
    /// One served file. A page bundle is html, js, css and small images.
    static let maxResourceBytes = 2 * 1024 * 1024
    /// One bridge message after JSON encoding.
    static let maxMessageBytes = 16 * 1024
    /// Widget state per page identity.
    static let maxWidgetStateBytes = 64 * 1024
    /// A follow-up draft. The composer is never sent for the user.
    static let maxFollowUpCharacters = 4_000
    /// Tool call results handed back to the page.
    static let maxToolResultBytes = 64 * 1024
    /// Tool calls a page may have in flight at once.
    static let maxPendingToolCalls = 2
    /// Tool calls per rolling minute, per page.
    static let maxToolCallsPerMinute = 12
    /// Any bridge message per rolling minute, per page (flood guard).
    static let maxMessagesPerMinute = 120
    /// How long after a real click or key press in the page a tool call is
    /// still treated as user initiated.
    static let userGestureWindowSeconds: TimeInterval = 5
    /// Surfaces one plugin may declare.
    static let maxSurfacesPerPlugin = 8
    /// Declared resources per surface.
    static let maxResourcesPerSurface = 64
}

/// Why a page request was refused. Typed so the failure card and the tests
/// name the cause instead of matching strings.
enum PluginPageError: Error, Equatable {
    /// The request path is malformed or climbs out of the page root.
    case forbiddenPath
    /// The path is well formed but the manifest does not declare it.
    case notDeclared
    /// A symlink sits on the path, or the file resolves outside the root.
    case symlinkRefused
    case notFound
    case tooLarge
    case unsupportedType
    case unreadable
    /// The surface's page could not be loaded by the web view at all.
    case loadFailed(String)
    /// The web content process ended and the one allowed reload also failed.
    case processTerminated
    /// The offline rule list could not be installed, so the page is not loaded.
    case sandboxUnavailable

    /// A stable machine name for logs.
    var code: String {
        switch self {
        case .forbiddenPath: return "forbidden_path"
        case .notDeclared: return "not_declared"
        case .symlinkRefused: return "symlink_refused"
        case .notFound: return "not_found"
        case .tooLarge: return "too_large"
        case .unsupportedType: return "unsupported_type"
        case .unreadable: return "unreadable"
        case .loadFailed: return "load_failed"
        case .processTerminated: return "process_terminated"
        case .sandboxUnavailable: return "sandbox_unavailable"
        }
    }
}

struct PluginPageData: Equatable {
    let data: Data
    let mimeType: String
}

enum PluginPageResolver {
    /// The only file types a plugin page may serve. Anything else (html is
    /// the one document type) is refused even when declared.
    static let mimeTypes: [String: String] = [
        "html": "text/html; charset=utf-8",
        "htm": "text/html; charset=utf-8",
        "js": "text/javascript; charset=utf-8",
        "mjs": "text/javascript; charset=utf-8",
        "css": "text/css; charset=utf-8",
        "json": "application/json; charset=utf-8",
        "txt": "text/plain; charset=utf-8",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml",
        "woff2": "font/woff2",
    ]

    /// Normalizes a bundle-relative path, or nil when it is not one.
    ///
    /// Refused: empty, absolute, any `..` or `.` segment other than a single
    /// leading `./`, empty segments, backslashes, NUL and control characters,
    /// a URL scheme, a percent sign (a path is decoded exactly once, by the
    /// caller, before it reaches here), and hidden (dot) segments.
    static func normalizeRelativePath(_ raw: String) -> String? {
        var path = raw
        if path.hasPrefix("./") { path.removeFirst(2) }
        guard !path.isEmpty, path.utf8.count <= 512 else { return nil }
        guard !path.hasPrefix("/"), !path.contains("\\"), !path.contains("%"),
            !path.contains(":"), !path.contains("?"), !path.contains("#")
        else { return nil }
        for scalar in path.unicodeScalars where scalar.value < 0x20 || scalar.value == 0x7f {
            return nil
        }
        let segments = path.split(separator: "/", omittingEmptySubsequences: false)
        for segment in segments {
            if segment.isEmpty || segment.hasPrefix(".") { return nil }
        }
        return segments.joined(separator: "/")
    }

    /// The lowercased extension, if it is one of the served types.
    static func mimeType(forPath path: String) -> String? {
        let ext = (path as NSString).pathExtension.lowercased()
        return mimeTypes[ext]
    }

    /// Extracts and normalizes the path of a `tsplugin://<token>/<path>`
    /// request URL, decoding percent escapes exactly once. Nil for a host
    /// other than `token`, so a page cannot reach a sibling page's origin.
    static func requestPath(from url: URL, token: String) -> String? {
        guard url.scheme == PluginPageScheme.name,
            url.host?.lowercased() == token.lowercased()
        else { return nil }
        // `path` already percent-decodes; a second decode would turn %252e
        // into a dot, so normalize runs on exactly one decode.
        let decoded = url.path
        guard decoded.hasPrefix("/") else { return nil }
        return normalizeRelativePath(String(decoded.dropFirst()))
    }

    /// Resolves and reads one declared file under `root`.
    ///
    /// Order, each step failing closed: the path must be declared, its type
    /// allow-listed, every component from the root down must be a real
    /// directory or the final regular file (no symlinks), the real path must
    /// stay under the real root, and the size must fit. The file is opened
    /// with `O_NOFOLLOW` and sized with `fstat` on the open descriptor so a
    /// swap between the checks and the read cannot widen what is returned.
    static func load(
        relativePath: String,
        root: URL,
        declared: Set<String>,
        maxBytes: Int = PluginPageLimits.maxResourceBytes
    ) -> Result<PluginPageData, PluginPageError> {
        guard let path = normalizeRelativePath(relativePath) else {
            return .failure(.forbiddenPath)
        }
        guard declared.contains(path) else { return .failure(.notDeclared) }
        guard let mime = mimeType(forPath: path) else { return .failure(.unsupportedType) }

        let realRoot = root.resolvingSymlinksInPath().standardizedFileURL.path
        var current = realRoot
        let segments = path.split(separator: "/").map(String.init)
        for (index, segment) in segments.enumerated() {
            current += "/" + segment
            var info = stat()
            guard lstat(current, &info) == 0 else { return .failure(.notFound) }
            let mode = info.st_mode & S_IFMT
            if mode == S_IFLNK { return .failure(.symlinkRefused) }
            let isLast = index == segments.count - 1
            if isLast, mode != S_IFREG { return .failure(.notFound) }
            if !isLast, mode != S_IFDIR { return .failure(.notFound) }
        }
        // Belt: the real path of the leaf still sits under the real root.
        let leaf = URL(fileURLWithPath: current).resolvingSymlinksInPath().standardizedFileURL.path
        guard leaf == current, leaf.hasPrefix(realRoot + "/") else {
            return .failure(.symlinkRefused)
        }

        let descriptor = open(current, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else {
            return .failure(errno == ELOOP ? .symlinkRefused : .unreadable)
        }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else {
            return .failure(.unreadable)
        }
        guard info.st_size >= 0, Int(info.st_size) <= maxBytes else { return .failure(.tooLarge) }
        var data = Data()
        data.reserveCapacity(Int(info.st_size))
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            let count = read(descriptor, &buffer, buffer.count)
            if count < 0 { return .failure(.unreadable) }
            if count == 0 { break }
            data.append(buffer, count: count)
            // The file grew after fstat: refuse rather than return more
            // than the cap.
            if data.count > maxBytes { return .failure(.tooLarge) }
        }
        return .success(PluginPageData(data: data, mimeType: mime))
    }

    /// The set of paths a surface may serve: its page and its declared
    /// resources, normalized. A malformed entry is skipped (the parser has
    /// already reported it), so a bad declaration widens nothing.
    static func declaredPaths(for surface: PluginUiSurface) -> Set<String> {
        var set = Set<String>()
        for raw in [surface.resourceUri] + surface.resources {
            if let normalized = normalizeRelativePath(raw) { set.insert(normalized) }
        }
        return set
    }
}

/// The custom scheme every plugin page loads through. It is the only origin
/// the navigation policy allows, so there is no file: or network origin for
/// a page to reach. The host is a per-load random token, so two pages never
/// share an origin (and therefore never share storage or cookies in memory).
enum PluginPageScheme {
    static let name = "tsplugin"

    static func url(token: String, path: String) -> URL? {
        var components = URLComponents()
        components.scheme = name
        components.host = token
        components.path = "/" + path
        return components.url
    }

    /// A fresh unguessable origin token, lowercase hex.
    static func makeToken() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }
}

/// The Content-Security-Policy every served response carries, derived from
/// the declaration rather than a constant: a surface that declares no
/// resources gets a policy with no `'self'` script source at all beyond its
/// own page. Minimum privilege: no network (`connect-src 'none'`), no
/// frames, no plugins, no forms, no base rewrite, and no inline script.
enum PluginPageCSP {
    static func header(for surface: PluginUiSurface) -> String {
        let declared = PluginPageResolver.declaredPaths(for: surface)
        func has(_ extensions: Set<String>) -> Bool {
            declared.contains { extensions.contains(($0 as NSString).pathExtension.lowercased()) }
        }
        let script = has(["js", "mjs"]) ? "'self'" : "'none'"
        let style = has(["css"]) ? "'self' 'unsafe-inline'" : "'unsafe-inline'"
        let image = has(["png", "jpg", "jpeg", "gif", "webp", "svg"]) ? "'self' data:" : "data:"
        let font = has(["woff2"]) ? "'self'" : "'none'"
        return [
            "default-src 'none'",
            "script-src \(script)",
            "style-src \(style)",
            "img-src \(image)",
            "font-src \(font)",
            "connect-src 'none'",
            "frame-src 'none'",
            "object-src 'none'",
            "media-src 'none'",
            "worker-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors 'none'",
        ].joined(separator: "; ")
    }
}

/// Which navigations the page may perform. Pure.
enum PluginPageNavigationPolicy {
    /// The initial load of the page's own document is the ONLY navigation
    /// allowed. A link, a form, `location = ...`, a reload by the page, a
    /// subframe, a new window, an `http(s)`, `file`, `data`, `javascript`
    /// or `blob` URL are all refused; `about:blank` is the empty shell
    /// WebKit starts from.
    static func allows(
        url: URL?,
        pageURL: URL,
        isMainFrame: Bool,
        initialLoadPending: Bool
    ) -> Bool {
        guard let url else { return false }
        if url.scheme?.lowercased() == "about", url.absoluteString == "about:blank" {
            return isMainFrame
        }
        guard isMainFrame, initialLoadPending else { return false }
        return url == pageURL
    }
}
