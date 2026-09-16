import Foundation

/// One parsed context reference: what a typed `@...` token MEANS, distinct
/// from the plain `@path` mention it may also be. The schemes mirror the
/// Hermes CLI's context references (`@file:`, `@folder:`, `@diff`,
/// `@staged`, `@git:N`, `@url:`); bare and quoted tokens stay paths, exactly
/// as they resolved before schemes existed.
enum ContextReference: Equatable {
    /// A file or folder resolved against the project root at submit time
    /// (which one is decided by what exists on disk, not by the syntax),
    /// optionally sliced to 1-indexed inclusive lines of the extracted text.
    case path(text: String, lines: ClosedRange<Int>?)
    /// The unstaged working tree diff (`git diff`).
    case diff
    /// The staged diff (`git diff --cached`).
    case staged
    /// The last N commits with patches, N clamped to 1...10.
    case git(count: Int)
    /// A web page fetched and converted to markdown.
    case url(URL)
}

/// A token parsed into a `ContextReference`, with the parse metadata
/// resolution needs.
struct ParsedContextReference: Equatable {
    let reference: ContextReference
    /// The path a line range was sliced from, when one was: resolution falls
    /// back to it when the sliced path does not exist but the raw one does
    /// (so a file literally named `notes:2024` still mentions).
    let unslicedPath: String?
    /// The token named an explicit scheme, so the user ASKED for that
    /// content: a failure is worth a toast. A bare `@word` in prose keeps
    /// the resolver's silence.
    let isExplicit: Bool

    init(reference: ContextReference, unslicedPath: String? = nil, isExplicit: Bool) {
        self.reference = reference
        self.unslicedPath = unslicedPath
        self.isExplicit = isExplicit
    }
}

/// Pure `@token` grammar for the typed reference schemes. No model, no I/O.
///
/// Parse rules, in the order they are tried on an UNQUOTED token body
/// (quoted tokens skip straight to the path rule, because quotes are how a
/// user names a file that collides with a reserved word such as `diff`):
///
/// - exactly `diff` or `staged`: the git reference. Reserved words: a file
///   of that name is reachable as `@"diff"`.
/// - `git:N`: N clamped to 1...10; anything unparsable after the colon is
///   not a reference and stays prose.
/// - `url:...`: the rest must parse as an http/https URL.
/// - `file:p` / `folder:p`: a path under an explicit scheme.
/// - anything else: a bare path, the original mention form.
///
/// Every value carries Hermes' two fuzz rules: trailing punctuation
/// (`,.;!?`) is stripped from the value, and an invalid line range silently
/// means the full file.
enum ContextReferenceParser {
    /// `@git:N` clamps into this range, matching Hermes.
    static let gitCommitLimit = 10

    static func parse(token: String, quoted: Bool) -> ParsedContextReference? {
        if quoted {
            // Quotes mean a literal path: no scheme words, no line ranges.
            let value = stripTrailingPunctuation(token)
            guard !value.isEmpty else { return nil }
            return ParsedContextReference(reference: .path(text: value, lines: nil), isExplicit: false)
        }
        // Punctuation strips BEFORE scheme matching, so the common prose
        // positions ("check @diff, then @staged.") still name the scheme.
        let body = stripTrailingPunctuation(token)
        if body == "diff" {
            return ParsedContextReference(reference: .diff, isExplicit: true)
        }
        if body == "staged" {
            return ParsedContextReference(reference: .staged, isExplicit: true)
        }
        if body.hasPrefix("git:") {
            guard let count = Int(String(body.dropFirst(4))) else { return nil }
            let clamped = min(max(count, 1), gitCommitLimit)
            return ParsedContextReference(reference: .git(count: clamped), isExplicit: true)
        }
        if body.hasPrefix("url:") {
            let value = String(body.dropFirst(4))
            guard let url = URL(string: value),
                let scheme = url.scheme?.lowercased(),
                scheme == "http" || scheme == "https"
            else { return nil }
            return ParsedContextReference(reference: .url(url), isExplicit: true)
        }
        if body.hasPrefix("file:") {
            let value = String(body.dropFirst(5))
            guard !value.isEmpty else { return nil }
            let (path, lines) = splitLineRange(value)
            return ParsedContextReference(
                reference: .path(text: path, lines: lines),
                unslicedPath: lines == nil ? nil : value,
                isExplicit: true)
        }
        if body.hasPrefix("folder:") {
            let value = String(body.dropFirst(7))
            guard !value.isEmpty else { return nil }
            // A folder has no lines; the whole value is the path.
            return ParsedContextReference(reference: .path(text: value, lines: nil), isExplicit: true)
        }
        // Bare token: a path, with an optional line-range suffix.
        guard !body.isEmpty else { return nil }
        let (path, lines) = splitLineRange(body)
        return ParsedContextReference(
            reference: .path(text: path, lines: lines),
            unslicedPath: lines == nil ? nil : body,
            isExplicit: false)
    }

    /// Hermes strips trailing reference punctuation so "check @diff." and
    /// "see @main.py," parse. A run is stripped: `@v1.2..` -> `v1.2`.
    static func stripTrailingPunctuation(_ value: String) -> String {
        var result = value
        while let last = result.last, ",.;!?".contains(last) {
            result.removeLast()
        }
        return result
    }

    /// Splits a trailing `:N` or `:N-M` (1-indexed, inclusive) off a path
    /// value. A suffix that does not even look like a range leaves the whole
    /// value as the path; one that is range-SHAPED but invalid (`:0`,
    /// `:25-10`) drops the range and keeps the path before the colon, which
    /// is Hermes' "invalid ranges are silently ignored (full file is
    /// returned)" rule.
    static func splitLineRange(_ value: String) -> (path: String, lines: ClosedRange<Int>?) {
        guard let colon = value.lastIndex(of: ":") else { return (value, nil) }
        let suffix = value[value.index(after: colon)...]
        guard let first = suffix.first, first.isNumber else { return (value, nil) }
        let path = String(value[..<colon])
        let parts = suffix.split(separator: "-", omittingEmptySubsequences: false)
        guard parts.count == 1 || parts.count == 2 else { return (path, nil) }
        let start = Int(parts[0])
        let end = parts.count == 2 ? Int(parts[1]) : start
        guard let s = start, let e = end, s >= 1, e >= s else {
            return (path, nil)
        }
        return (path, s...e)
    }
}
