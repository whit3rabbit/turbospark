import Foundation

// Typed errors for the model-library surface.
//
// A refusal puts a stable lowercase code first in the engine's last-error
// string, "<code>: <human sentence>" (crates/ffi/src/api/libraries.rs). The
// code is the contract and a host localizes by it; the sentence is for logs.
// This file turns the prefix into a case, and it keys on the PREFIX alone:
//
// **`TurboSparkError.Code.busy` IS NOT THE LIBRARY `busy`.** That status
// already means "the native audio device or session is busy"
// (`AudioWorkspaceController+Installs` keys on it), and a library change
// refused while an install or a move runs uses the same status with the
// prefix `busy: `. Reading the status alone would show one of them as the
// other, so a caller that cares asks `error.libraryFault?.code == .busy`.
//
// A message with no code is not a library fault and `libraryFault` is nil for
// it. Install failures keep their pre-library form with no code (a network
// error, `install cancelled`), and so does `options is not valid install
// options`; inventing a prefix for them would be a guess presented as a
// parse.

/// Why a location cannot be a library (add and import). All of these come
/// back as `TurboSparkError.Code.invalidArgument`.
public enum LibraryRefusal: String, Sendable, Equatable, Hashable, CaseIterable {
    /// The folder is on the same disk as the TurboSpark home, which This Mac
    /// already covers.
    case sameDiskAsHome = "same_disk_as_home"
    /// A network share. Libraries must be on a directly connected drive.
    case network
    /// A file system that cannot hold a file of 4 GiB or more.
    case smallFileLimit = "small_file_limit"
    /// A read-only drive.
    case readOnly = "read_only"
    /// Overlaps a library that is already listed.
    case nested
    /// Overlaps the TurboSpark home folder.
    case insideHome = "inside_home"
    /// The drive cannot be identified: no exact mount entry, an unreadable
    /// mount table, or a mount nested inside another volume.
    case unknownVolume = "unknown_volume"
    /// The path is not an existing folder.
    case notADirectory = "not_a_directory"
}

/// The stable code at the front of a library error, as a case.
///
/// One flat set rather than a type per entry point, because the codes overlap
/// (`not_connected` answers an add, an install, a move and a delete). Which
/// entry point can return which code is listed in the header's "model
/// libraries" block and in `docs/SWIFT_BINDINGS.md`.
public enum LibraryErrorCode: Sendable, Equatable, Hashable {
    /// Add and import: the location breaks a rule.
    case refusal(LibraryRefusal)

    // Libraries.
    /// The id names no listed library.
    case unknownLibrary
    /// This Mac cannot be removed.
    case internalLibrary
    /// The library's drive is not connected.
    case notConnected
    /// A library marker that cannot be read or written.
    case marker
    /// Any other file system failure. The sentence says what.
    case io

    // Installs (also moves and deletes where noted in the docs).
    /// The drive went away while an install ran. Nothing partial is listed.
    case libraryDisconnected
    /// A library a newer TurboSpark wrote: listed, not changed.
    case libraryReadOnly
    /// The model stays on This Mac (embedding and OCR models).
    case homeOnly
    /// The alias and modality are already installed in This Mac or another
    /// library, so the install is a move instead. The sentence names where.
    case aliasElsewhere

    // Deletes and moves.
    /// The id names no recorded install, or no image folder.
    case unknownInstall
    /// `ts_model_delete` of an alias a library holds: use the install id.
    case useInstallId
    /// The destination is the library the install is already in.
    case sameLibrary
    /// The destination already has an install with this alias.
    case destinationTaken
    /// The destination has less free space than the model needs. The figures
    /// exist only in the sentence.
    case insufficientSpace
    /// The copy did not match the original, and was removed.
    case verifyFailed
    /// The caller could not measure the destination's free space.
    case freeSpaceUnknown

    // Statuses of their own.
    /// An install or a model move is running, so a library change, a delete,
    /// an install or a move was refused. Nothing changed. Distinct from the
    /// audio device being busy: see the note at the top of this file.
    case busy
    /// The caller's own cancel ended a move. Not a failure.
    case cancelled

    /// A well-formed code this build does not know. The spelling is kept, so
    /// an engine newer than the app degrades to a generic message rather than
    /// a crash.
    case unrecognized(String)

    /// The spelling on the wire, as the header documents it.
    public var wireCode: String {
        switch self {
        case .refusal(let refusal): return refusal.rawValue
        case .unknownLibrary: return "unknown_library"
        case .internalLibrary: return "internal_library"
        case .notConnected: return "not_connected"
        case .marker: return "marker"
        case .io: return "io"
        case .libraryDisconnected: return "library_disconnected"
        case .libraryReadOnly: return "library_read_only"
        case .homeOnly: return "home_only"
        case .aliasElsewhere: return "alias_elsewhere"
        case .unknownInstall: return "unknown_install"
        case .useInstallId: return "use_install_id"
        case .sameLibrary: return "same_library"
        case .destinationTaken: return "destination_taken"
        case .insufficientSpace: return "insufficient_space"
        case .verifyFailed: return "verify_failed"
        case .freeSpaceUnknown: return "free_space_unknown"
        case .busy: return "busy"
        case .cancelled: return "cancelled"
        case .unrecognized(let code): return code
        }
    }

    /// Maps a spelling to its case. Never fails: an unknown spelling becomes
    /// `.unrecognized`.
    public init(wireCode: String) {
        if let refusal = LibraryRefusal(rawValue: wireCode) {
            self = .refusal(refusal)
            return
        }
        switch wireCode {
        case "unknown_library": self = .unknownLibrary
        case "internal_library": self = .internalLibrary
        case "not_connected": self = .notConnected
        case "marker": self = .marker
        case "io": self = .io
        case "library_disconnected": self = .libraryDisconnected
        case "library_read_only": self = .libraryReadOnly
        case "home_only": self = .homeOnly
        case "alias_elsewhere": self = .aliasElsewhere
        case "unknown_install": self = .unknownInstall
        case "use_install_id": self = .useInstallId
        case "same_library": self = .sameLibrary
        case "destination_taken": self = .destinationTaken
        case "insufficient_space": self = .insufficientSpace
        case "verify_failed": self = .verifyFailed
        case "free_space_unknown": self = .freeSpaceUnknown
        case "busy": self = .busy
        case "cancelled": self = .cancelled
        default: self = .unrecognized(wireCode)
        }
    }

    /// The location refusal, when this is one.
    public var refusal: LibraryRefusal? {
        if case .refusal(let refusal) = self { return refusal }
        return nil
    }
}

/// A library error read from the engine's "<code>: <sentence>" message.
public struct LibraryFault: Sendable, Equatable, CustomStringConvertible {
    public let code: LibraryErrorCode
    /// The status the call returned with the message.
    public let status: TurboSparkError.Code
    /// The engine's sentence, exactly as sent.
    ///
    /// **Untrusted.** It can carry a library or volume name, an alias or a
    /// path from a drive anyone could have written. The engine escapes
    /// control and bidirectional characters and caps it at 300 characters, but
    /// do not show it unprocessed: use `displaySentence`.
    public let sentence: String

    /// `sentence` escaped and cut for display (see `UntrustedText`).
    public var displaySentence: String { UntrustedText.display(sentence) }

    public var description: String { "\(code.wireCode): \(displaySentence)" }

    /// Reads a failed call's status and message. Nil when the message carries
    /// no code (see the note at the top of this file).
    public init?(status: TurboSparkError.Code, message: String) {
        guard let (code, sentence) = LibraryFault.split(message) else { return nil }
        self.code = LibraryErrorCode(wireCode: code)
        self.status = status
        self.sentence = sentence
    }

    public init?(_ error: TurboSparkError) {
        self.init(status: error.code, message: error.message)
    }

    /// The longest code accepted. The longest the engine writes is 20
    /// characters; the margin is for codes added later, and the cap keeps an
    /// ordinary sentence with an early colon from reading as a code.
    private static let longestCode = 40

    /// Splits "<code>: <sentence>". A code is lowercase ASCII letters and
    /// underscores, and the colon is followed by a space or ends the message.
    static func split(_ message: String) -> (code: String, sentence: String)? {
        guard let colon = message.firstIndex(of: ":") else { return nil }
        let code = message[..<colon]
        guard !code.isEmpty, code.count <= longestCode,
            code.utf8.allSatisfy({ ($0 >= 0x61 && $0 <= 0x7A) || $0 == 0x5F })
        else { return nil }
        var rest = message[message.index(after: colon)...]
        if rest.isEmpty { return (String(code), "") }
        guard rest.first == " " else { return nil }
        rest = rest.dropFirst()
        return (String(code), String(rest))
    }
}

extension TurboSparkError {
    /// The library code and sentence, when the engine put one first in the
    /// message. Nil for every other error, including the audio device being
    /// busy (`code == .busy` with no prefix).
    public var libraryFault: LibraryFault? { LibraryFault(self) }
}

/// Text that came from a drive, a marker or an untrusted peer, made safe to
/// show: a library name, an alias, a path, an error sentence.
///
/// Control characters would move the cursor or recolor a terminal, and
/// bidirectional marks can reorder what the user reads (a right-to-left
/// override makes `gnp.exe` read as `exe.png`). Both are shown as escapes, in
/// the spelling the engine uses for its own sentences. The result is cut to
/// `limit` characters (counted after escaping, so an escape cannot be used to
/// widen the text) with `...` as the last three.
public enum UntrustedText {
    /// About one line in a list row.
    public static let defaultLimit = 120

    public static func display(_ text: String, limit: Int = defaultLimit) -> String {
        var escaped = ""
        escaped.reserveCapacity(text.utf8.count)
        for scalar in text.unicodeScalars {
            if needsEscape(scalar) {
                escaped += escape(scalar)
            } else {
                escaped.unicodeScalars.append(scalar)
            }
        }
        guard escaped.count > limit else { return escaped }
        return String(escaped.prefix(max(limit - 3, 0))) + "..."
    }

    private static func needsEscape(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .control, .lineSeparator, .paragraphSeparator:
            return true
        default:
            break
        }
        switch scalar.value {
        // The explicit direction marks, embeddings, overrides and isolates:
        // the same set the engine escapes in its own sentences.
        case 0x061C, 0x200E, 0x200F, 0x202A...0x202E, 0x2066...0x2069:
            return true
        default:
            return false
        }
    }

    private static func escape(_ scalar: Unicode.Scalar) -> String {
        switch scalar {
        case "\n": return "\\n"
        case "\r": return "\\r"
        case "\t": return "\\t"
        default: return "\\u{" + String(scalar.value, radix: 16) + "}"
        }
    }
}
