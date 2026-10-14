import Foundation
import Syntext
import SyntextContext

/// What one user turn's evidence may read.
struct ContextEvidenceRequest: Sendable {
    let chatID: UUID
    /// Where the chat's turns are recorded; `nil` when recall is off for it
    /// (the setting, or a ghost chat).
    let recall: ContextChatLocation?
    /// The project whose files the turn may read; `nil` outside a project
    /// turn or with that project's engine off.
    let projectID: UUID?
    let question: String
    let evidenceTokens: Int
    /// The chat's recall rows before the question, reconciled first.
    let recallRows: [ContextRecall.Row]
    /// How many of the newest recall rows the prompt already holds.
    let keptTurns: Int
    /// Tokens one `context_*` tool result may use this turn.
    let toolResultTokens: Int
}

/// Evidence for one user turn, ready to put in its message.
struct ContextEvidence: Sendable, Equatable {
    let text: String
    let tokens: Int
    let blockCount: Int
    let lowestTrust: SyntextTrust?
}

extension ContextRecall.Row {
    init(recorded message: SyntextChatMessage) {
        self.init(role: message.role == .assistant ? .assistant : .user, text: message.content)
    }

    var syntextRole: SyntextChatRole { role == .assistant ? .assistant : .user }
}

extension ContextEngineManager {
    /// Conversations kept open at once; more are reopened on demand.
    static let maxOpenConversations = 64

    func conversation(for location: ContextChatLocation) async throws -> SyntextConversation {
        if let existing = conversations[location] { return existing }
        let open = try await openStore(location.store)
        if let existing = conversations[location] { return existing }
        let created = try SyntextConversation(
            store: open.store, session: location.session, mode: location.memoryMode)
        if conversations.count >= Self.maxOpenConversations { conversations.removeAll() }
        conversations[location] = created
        return created
    }

    /// Runs `body` after every earlier recall job of `location`, so two
    /// reconciles of one chat (a turn finishing, the next one starting)
    /// never record the same rows twice.
    func serializeRecall(
        _ location: ContextChatLocation,
        _ body: @escaping @Sendable () throws -> Void
    ) async throws {
        let previous = recallTasks[location]
        let task = Task.detached(priority: .utility) {
            _ = await previous?.result
            try body()
        }
        recallTasks[location] = task
        try await task.value
    }

    /// Brings the chat's recorded turns in line with `rows` (see
    /// `ContextRecall`). Returns the conversation for retrieval.
    @discardableResult
    func reconcile(
        _ location: ContextChatLocation, rows: [ContextRecall.Row]
    ) async throws -> SyntextConversation {
        let conversation = try await conversation(for: location)
        try await serializeRecall(location) {
            let recorded = try conversation.history().map(ContextRecall.Row.init(recorded:))
            switch ContextRecall.plan(recorded: recorded, desired: rows) {
            case .upToDate:
                return
            case .append(let new):
                for row in new { try conversation.record(row.syntextRole, row.text) }
            case .rebuild(let all):
                try conversation.forget()
                for row in all { try conversation.record(row.syntextRole, row.text) }
            }
        }
        return conversation
    }

    /// Records a finished turn: reconciles the chat with its whole
    /// transcript. Failures only cost recall, never the turn.
    func recordTurn(chatID: UUID, store: ContextChatLocation.Store, rows: [ContextRecall.Row]) async {
        _ = try? await reconcile(ContextChatLocation(store: store, chatID: chatID), rows: rows)
    }

    /// Evidence for one user turn: the project's files when the turn may
    /// read them, and the chat's own older turns when it records them.
    /// `nil` when there is nothing to read or nothing relevant.
    func prepareEvidence(_ request: ContextEvidenceRequest) async throws -> ContextEvidence? {
        toolOptions.removeValue(forKey: request.chatID)
        let key: ContextChatLocation.Store
        if let projectID = request.projectID {
            key = .project(projectID)
        } else if let recall = request.recall {
            key = recall.store
        } else {
            return nil
        }
        var conversation: SyntextConversation?
        if let recall = request.recall, recall.store == key {
            conversation = try await reconcile(recall, rows: request.recallRows)
        }
        let engine = try await engine(for: key)
        // Explicit scopes: project files only on a project turn. The
        // conversation's own scope is granted by `prepare` itself.
        let fileScopes = request.projectID == nil ? [String]() : ["default"]
        let packet = try await engine.prepareAsync(
            question: request.question, evidenceTokens: request.evidenceTokens,
            conversation: conversation, keptTurns: request.keptTurns, scopes: fileScopes)
        if request.projectID != nil {
            var scopes = fileScopes
            if conversation != nil, let recall = request.recall { scopes.append(recall.memoryScope) }
            var options = SyntextToolOptions(
                scopes: scopes, resultTokens: max(64, request.toolResultTokens))
            options.record(packet)
            toolOptions[request.chatID] = options
        }
        guard !packet.isEmpty else { return nil }
        return ContextEvidence(
            text: packet.evidenceText, tokens: packet.evidenceTokens,
            blockCount: packet.blocks.count, lowestTrust: packet.lowestTrust)
    }

    /// Runs a `context_search`, `context_read` or `context_outline` call
    /// against a project's store. Scopes and budgets come from this turn's
    /// options, never from the model's arguments.
    func callTool(
        name: String, arguments: String, chatID: UUID?, projectID: UUID
    ) async throws -> SyntextToolResult {
        let engine = try await engine(for: .project(projectID))
        var options = chatID.flatMap { toolOptions[$0] } ?? SyntextToolOptions(scopes: ["default"])
        let result = try await engine.callToolAsync(name: name, arguments: arguments, options: options)
        options.record(result)
        if let chatID { toolOptions[chatID] = options }
        return result
    }

    /// Deletes a chat's recorded turns wherever they may be: the chats store,
    /// and its project's store when it has one. Their bytes leave the disk
    /// at a forced compaction shortly after.
    public func forgetChat(_ chatID: UUID, projectID: UUID?) async {
        toolOptions.removeValue(forKey: chatID)
        var keys: [ContextChatLocation.Store] = [.chats]
        if let projectID { keys.append(.project(projectID)) }
        for key in keys {
            guard stores[key] != nil
                || FileManager.default.fileExists(atPath: Self.directory(for: key).path)
            else { continue }
            let location = ContextChatLocation(store: key, chatID: chatID)
            guard let conversation = try? await conversation(for: location) else { continue }
            try? await serializeRecall(location) { _ = try conversation.forget() }
            conversations.removeValue(forKey: location)
            scheduleForcedCompaction(key)
        }
    }

    /// Several chats deleted together cost one compaction per store.
    func scheduleForcedCompaction(_ key: ContextChatLocation.Store) {
        compactionTasks[key]?.cancel()
        compactionTasks[key] = Task {
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            guard !Task.isCancelled else { return }
            await self.forceCompaction(key)
        }
    }

    private func forceCompaction(_ key: ContextChatLocation.Store) async {
        compactionTasks.removeValue(forKey: key)
        guard let open = stores[key] else { return }
        let store = open.store
        await Task.detached(priority: .background) {
            _ = try? store.compact(force: true)
        }.value
        // A project store opened only to forget a chat is closed again.
        if case .project(let id) = key, activeProject?.id != id { closeStore(key) }
    }
}

extension ContextEngineManager {
    /// Drops the cached conversation handles, so a test reads what a fresh
    /// handle loads from the store.
    func removeConversationHandlesForTesting() {
        conversations.removeAll()
    }
}
