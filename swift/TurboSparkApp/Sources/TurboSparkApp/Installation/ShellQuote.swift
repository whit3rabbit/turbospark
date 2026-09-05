import Foundation

/// POSIX single-quote quoting, for text this app puts into a command the user
/// will paste into a shell.
///
/// It exists because the default system prompt is USER-AUTHORED free text and
/// the launch snippet embeds it verbatim. An apostrophe is the common case,
/// not an exotic one -- "don't", "the user's request" -- and unquoted it ends
/// the argument and leaves the rest of the prompt as stray shell words.
enum ShellQuote {
    /// Wraps `value` in single quotes, safe for any byte a shell can carry.
    ///
    /// Single quotes suppress every expansion a shell does, so the only
    /// character needing care is the single quote itself, which cannot be
    /// escaped inside a single-quoted string at all. The standard workaround
    /// is to close the string, emit an escaped quote, and reopen:
    /// `'` becomes `'\''`. Newlines need nothing; they are literal inside
    /// single quotes.
    static func single(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
