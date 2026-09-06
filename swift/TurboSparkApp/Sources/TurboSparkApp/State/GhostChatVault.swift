import CryptoKit
import Foundation

/// The conversation contents of a ghost chat, held only inside the vault.
///
/// A ghost chat's row in `chats` keeps these five fields empty on purpose:
/// every mutation and read goes through `GhostChatVault`, so the plaintext
/// transcript exists only as function-local values while a turn is running.
public struct GhostChatPayload: Codable, Equatable {
    /// Committed conversation message history.
    public var messages: [AppChatMessage]
    /// Active task checklist for the session.
    public var todos: [TodoItem]
    /// Optional context summary or metadata.
    public var contextSummary: String?
    /// Leading message rows the summary replaces in the prompt. Same meaning
    /// as `AppChat.compactedMessageCount`; the vault is per-launch, so this
    /// needs no tolerant decode.
    public var compactedMessageCount: Int
    /// Bounded execution state, when the project runs in SKILL.state mode.
    public var skillState: AppSkillState?
    /// Uncommitted draft prompt text.
    public var draft: String

    public init(
        messages: [AppChatMessage] = [],
        todos: [TodoItem] = [],
        contextSummary: String? = nil,
        compactedMessageCount: Int = 0,
        skillState: AppSkillState? = nil,
        draft: String = ""
    ) {
        self.messages = messages
        self.todos = todos
        self.contextSummary = contextSummary
        self.compactedMessageCount = compactedMessageCount
        self.skillState = skillState
        self.draft = draft
    }

    /// Whether the payload carries anything a user could lose.
    public var hasContent: Bool {
        !messages.isEmpty || !todos.isEmpty || !draft.isEmpty
    }
}

/// In-memory AES-GCM vault for Ghost Mode chats.
///
/// **THE KEY IS PER LAUNCH AND NEVER SERIALIZED.** A fresh 256-bit
/// `SymmetricKey` is generated when the model is created and dies with the
/// process, so the ciphertext in memory is unreadable to anything that
/// scrapes it after the fact, and a payload has no meaning outside the run
/// that sealed it. The hard guarantee against persistence is NOT this class
/// -- it is the `isGhost` filter at the two archive-construction points in
/// `AppModel+Persistence.swift`; the vault is defense in depth, and this
/// class is what "encrypted in memory" means here.
///
/// `open` failure falls back to an EMPTY payload rather than throwing: the
/// only way it fails is ciphertext sealed by another launch (impossible --
/// the dictionary dies with the key) or corruption, and an empty transcript
/// beats a wedged UI either way.
@MainActor
public final class GhostChatVault {
    private var key = SymmetricKey(size: .bits256)
    private var ciphertext: [UUID: Data] = [:]
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    public init() {}

    /// Decrypts the payload sealed for `chatID`, or an empty one.
    public func payload(for chatID: UUID) -> GhostChatPayload {
        guard let data = ciphertext[chatID] else { return GhostChatPayload() }
        do {
            let sealed = try AES.GCM.SealedBox(combined: data)
            let plaintext = try AES.GCM.open(sealed, using: key)
            return try decoder.decode(GhostChatPayload.self, from: plaintext)
        } catch {
            return GhostChatPayload()
        }
    }

    /// Re-seals the payload for `chatID` under a fresh nonce.
    public func store(_ payload: GhostChatPayload, for chatID: UUID) {
        // Every field of the payload is a Codable value type, so encoding
        // cannot fail in practice; dropping the entry is the safe recovery
        // if one ever does, and `open` above already answers empty.
        guard let plaintext = try? encoder.encode(payload) else {
            ciphertext[chatID] = nil
            return
        }
        guard let sealed = try? AES.GCM.seal(plaintext, using: key).combined else {
            ciphertext[chatID] = nil
            return
        }
        ciphertext[chatID] = sealed
    }

    /// Whether any ciphertext is sealed for `chatID`.
    public func hasPayload(for chatID: UUID) -> Bool {
        ciphertext[chatID] != nil
    }

    /// Drops one chat's ciphertext.
    public func wipe(for chatID: UUID) {
        ciphertext[chatID] = nil
    }

    /// Drops everything and replaces the key, so no later seal can be
    /// correlated with an earlier one. Called on quit.
    public func wipeAll() {
        ciphertext.removeAll()
        key = SymmetricKey(size: .bits256)
    }
}
