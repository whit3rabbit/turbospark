import Foundation
import Syntext
import SyntextContext
import TurboSpark

/// Conversation recall: syntext's context engine journals each chat's turns
/// to a local store and, once turns have left the model's window (compacted
/// or dropped by the fit), brings the relevant ones back as cited evidence.
///
/// **THE STORE IS PLAINTEXT.** Syntext has no encryption, and the journal and
/// segments hold turn text as written. Ghost chats are never sent here, and
/// a passphrase-protected profile has no store at all (protecting one purges
/// it), because its promise is that nothing is readable while it is locked.
/// A local-key profile keeps its database key beside the database, so a
/// plaintext store beside it exposes nothing new to anyone who can read the
/// folder. See `docs/PROFILE_VAULT.md`.
///
/// Every call blocks in the FFI (an fsync, a store scan), so all of them run
/// on one private serial queue rather than an actor: blocking work inside
/// Swift's cooperative pool starves unrelated tasks. The serial order is also
/// what makes a reconcile land before the evidence request that follows it.
final class ChatRecallStore: @unchecked Sendable {
    static let shared = ChatRecallStore()

    /// Each chat is its own memory scope. Syntext grants a conversation's
    /// whole SCOPE to a request, not just its session, so a shared scope
    /// would let one chat recall another's turns.
    static func scope(for chatID: UUID) -> String { "turbospark.chat.\(chatID.uuidString)" }

    private let queue = DispatchQueue(label: "com.turbospark.chat-recall", qos: .utility)
    private let directory: URL
    private let isProtected: @Sendable () -> Bool

    // Queue-confined state.
    private var enabled = true
    private var store: SyntextContextStore?
    private var engine: (modelPath: String, engine: SyntextContextEngine)?
    private var conversations: [UUID: SyntextConversation] = [:]
    private var conversationOrder: [UUID] = []
    private var evidenceCache: [String: String] = [:]
    private var evidenceCacheOrder: [String] = []
    private var compactTask: DispatchWorkItem?
    private var needsForcedCompaction = false

    /// Opening a conversation scans the store, so a few stay open.
    static let maxOpenConversations = 4

    init(
        directory: URL = AppStorageRoot.directory.appendingPathComponent("context-recall", isDirectory: true),
        isProtected: @escaping @Sendable () -> Bool = { ProfileVaultStore.shared.isProtected }
    ) {
        self.directory = directory
        self.isProtected = isProtected
    }

    var storeDirectory: URL { directory }

    // MARK: - Gate

    /// Mirrors the user's setting. Turning it off also deletes the store:
    /// it is derived from the transcript and rebuilds lazily when turned on.
    func setEnabled(_ enabled: Bool) {
        queue.async {
            self.enabled = enabled
            if !enabled { self.purgeLocked() }
        }
    }

    /// Checked on the queue before every write and every read.
    private var allowedLocked: Bool { enabled && !isProtected() }

    // MARK: - Recording

    /// Brings the chat's journal in line with its transcript: appends what
    /// is new, or forgets and re-records when the transcript was rewritten
    /// (edit, retry, rewind, variant switch).
    func reconcile(chatID: UUID, turns: [ChatRecallPlanner.Turn]) {
        queue.async { self.reconcileLocked(chatID: chatID, turns: turns) }
    }

    /// What the store holds for a chat, oldest first. Tests use it.
    func recordedTurns(chatID: UUID) async -> [(role: String, text: String)] {
        await withCheckedContinuation { continuation in
            queue.async {
                let conversation = self.conversationLocked(chatID: chatID, create: true, mustExist: true)
                let turns = (try? conversation?.history()) ?? []
                continuation.resume(returning: turns.map { (role: $0.role.rawValue, text: $0.content) })
            }
        }
    }

    /// Waits for every queued reconcile, forget, and purge. Tests use it.
    func drain() async {
        await withCheckedContinuation { continuation in
            queue.async { continuation.resume() }
        }
    }

    private func reconcileLocked(chatID: UUID, turns: [ChatRecallPlanner.Turn]) {
        guard allowedLocked, let conversation = conversationLocked(chatID: chatID, create: true) else {
            return
        }
        do {
            let recorded = try conversation.history().map { (role: $0.role.rawValue, text: $0.content) }
            switch ChatRecallPlanner.reconcilePlan(recorded: recorded, recordable: turns) {
            case .none:
                return
            case .append(let suffix):
                try record(suffix, into: conversation)
            case .rebuild:
                try conversation.forget()
                try record(turns, into: conversation)
                scheduleCompactionLocked(force: false)
            }
            invalidateEvidenceLocked(chatID: chatID)
        } catch {
            // Derived data: the next reconcile tries again from scratch.
            dropConversationLocked(chatID)
        }
    }

    private func record(_ turns: [ChatRecallPlanner.Turn], into conversation: SyntextConversation) throws {
        for turn in turns {
            try conversation.record(turn.role == .user ? .user : .assistant, turn.text)
        }
    }

    // MARK: - Recall

    /// Rendered evidence for `question` from the chat's turns outside the
    /// prompt (`keptTurns` most recent ones are in it), within `budget`
    /// tokens. Nil when recall is off, protected, empty, or failing.
    /// Cached per request, so every step of a tool loop gets the same text.
    func evidence(
        chatID: UUID, question: String, keptTurns: Int, budget: Int, modelPath: String
    ) async -> String? {
        await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume(returning: self.evidenceLocked(
                    chatID: chatID, question: question, keptTurns: keptTurns,
                    budget: budget, modelPath: modelPath))
            }
        }
    }

    private func evidenceLocked(
        chatID: UUID, question: String, keptTurns: Int, budget: Int, modelPath: String
    ) -> String? {
        guard allowedLocked else { return nil }
        let key = "\(chatID.uuidString)/\(keptTurns)/\(budget)/\(modelPath)/\(question.utf8.count)/\(question.hashValue)"
        if let cached = evidenceCache[key] { return cached.isEmpty ? nil : cached }
        guard let conversation = conversationLocked(chatID: chatID, create: true),
              let engine = engineLocked(modelPath: modelPath),
              let packet = try? engine.prepare(
                question: question, evidenceTokens: budget,
                conversation: conversation, keptTurns: keptTurns)
        else { return nil }
        let text = packet.isEmpty ? "" : packet.evidenceText
        evidenceCache[key] = text
        evidenceCacheOrder.append(key)
        if evidenceCacheOrder.count > 16 {
            evidenceCache.removeValue(forKey: evidenceCacheOrder.removeFirst())
        }
        return text.isEmpty ? nil : text
    }

    private func invalidateEvidenceLocked(chatID: UUID) {
        let prefix = chatID.uuidString + "/"
        evidenceCacheOrder.removeAll { key in
            guard key.hasPrefix(prefix) else { return false }
            evidenceCache.removeValue(forKey: key)
            return true
        }
    }

    // MARK: - Forgetting

    /// Deletes every turn of a chat the user deleted or cleared, and
    /// schedules the forced compaction that takes compacted turns' bytes off
    /// the disk. Does nothing when no store exists.
    func forget(chatID: UUID) {
        queue.async {
            guard FileManager.default.fileExists(atPath: self.directory.path),
                  let conversation = self.conversationLocked(
                    chatID: chatID, create: true, mustExist: true)
            else { return }
            _ = try? conversation.forget()
            self.dropConversationLocked(chatID)
            self.invalidateEvidenceLocked(chatID: chatID)
            self.scheduleCompactionLocked(force: true)
        }
    }

    /// Closes everything and deletes the store folder.
    func purge() async {
        await withCheckedContinuation { continuation in
            queue.async {
                self.purgeLocked()
                continuation.resume()
            }
        }
    }

    private func purgeLocked() {
        closeLocked()
        try? FileManager.default.removeItem(at: directory)
    }

    /// Releases every handle (shutdown, profile lock). The folder stays.
    func close() {
        queue.async { self.closeLocked() }
    }

    private func closeLocked() {
        compactTask?.cancel()
        compactTask = nil
        needsForcedCompaction = false
        conversations.removeAll()
        conversationOrder.removeAll()
        evidenceCache.removeAll()
        evidenceCacheOrder.removeAll()
        engine = nil
        store = nil
    }

    /// Removes a store left behind by a crash between turning protection on
    /// (or recall off) and the purge that should have followed.
    func purgeIfDisallowed() {
        queue.async {
            if !self.allowedLocked { self.purgeLocked() }
        }
    }

    // MARK: - Handles

    private func storeLocked(mustExist: Bool = false) -> SyntextContextStore? {
        if let store { return store }
        let fm = FileManager.default
        if !fm.fileExists(atPath: directory.path) {
            if mustExist { return nil }
            // Owner-only, and out of backups: it is a plaintext, rebuildable
            // copy of the transcript.
            do {
                try fm.createDirectory(
                    at: directory, withIntermediateDirectories: true,
                    attributes: [.posixPermissions: 0o700])
                var url = directory
                var values = URLResourceValues()
                values.isExcludedFromBackup = true
                try url.setResourceValues(values)
            } catch {
                return nil
            }
        }
        // No fsync: the transcript is the source of truth, a torn journal
        // tail is cut at the next open, and reconcile re-records the gap.
        let options = SyntextContextStoreOptions(mustExist: mustExist, sync: false)
        for attempt in 1...3 {
            do {
                let opened = try SyntextContextStore(directory: directory.path, options: options)
                store = opened
                return opened
            } catch let error as SyntextError where error.isContextStoreLocked && attempt < 3 {
                // Spurious lock failures happen under fork churn (the app
                // spawns git); a real second writer stays locked and fails.
                usleep(useconds_t(100_000 * attempt))
            } catch {
                return nil
            }
        }
        return nil
    }

    private func engineLocked(modelPath: String) -> SyntextContextEngine? {
        if let engine, engine.modelPath == modelPath { return engine.engine }
        guard let store = storeLocked() else { return nil }
        // The model's own tokenizer, opened separately: the session's
        // counter runs on the session queue, which generation holds for the
        // whole turn, so calling it from here could deadlock.
        let tokenizer = try? TurboSparkTokenizer(modelPath: modelPath)
        let counter: @Sendable (String) -> Int = { text in
            (try? tokenizer?.count(text)) ?? max(1, text.utf8.count / 3)
        }
        guard let created = try? SyntextContextEngine(
            store: store,
            options: SyntextContextEngineOptions(appendPolicy: false),
            tokenCounter: counter)
        else { return nil }
        engine = (modelPath, created)
        return created
    }

    private func conversationLocked(
        chatID: UUID, create: Bool, mustExist: Bool = false
    ) -> SyntextConversation? {
        if let open = conversations[chatID] {
            conversationOrder.removeAll { $0 == chatID }
            conversationOrder.append(chatID)
            return open
        }
        guard create, let store = storeLocked(mustExist: mustExist),
              let conversation = try? SyntextConversation(
                store: store, session: chatID.uuidString, mode: .app(Self.scope(for: chatID)))
        else { return nil }
        conversations[chatID] = conversation
        conversationOrder.append(chatID)
        if conversationOrder.count > Self.maxOpenConversations {
            conversations.removeValue(forKey: conversationOrder.removeFirst())
        }
        return conversation
    }

    private func dropConversationLocked(_ chatID: UUID) {
        conversations.removeValue(forKey: chatID)
        conversationOrder.removeAll { $0 == chatID }
    }

    /// Debounced so a burst of deletes rewrites the store once. A forced
    /// compaction rewrites every segment, so only deletion and clearing ask
    /// for one; rebuilds after an edit compact only when the store is due.
    private func scheduleCompactionLocked(force: Bool) {
        needsForcedCompaction = needsForcedCompaction || force
        compactTask?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self, let store = self.store else { return }
            _ = try? store.compact(force: self.needsForcedCompaction)
            self.needsForcedCompaction = false
            self.compactTask = nil
        }
        compactTask = work
        queue.asyncAfter(deadline: .now() + 2, execute: work)
    }

    /// Runs any scheduled compaction now. Tests use it.
    func compactNow() async {
        await withCheckedContinuation { continuation in
            queue.async {
                if let work = self.compactTask {
                    work.cancel()
                    _ = try? self.store?.compact(force: self.needsForcedCompaction)
                    self.needsForcedCompaction = false
                    self.compactTask = nil
                }
                continuation.resume()
            }
        }
    }
}
