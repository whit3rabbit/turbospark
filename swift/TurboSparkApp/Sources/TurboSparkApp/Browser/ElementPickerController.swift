import Combine
import Foundation

@MainActor
public final class ElementPickerController: ObservableObject {
    public enum ErrorState: Equatable {
        case staleCandidate
    }

    public typealias CandidatePicker = @MainActor (BrowserTabID, CGPoint) async throws -> DOMSnapshotPickCandidate?
    public typealias CandidateValidator = @MainActor (BrowserTabID, DOMSnapshotPickCandidate) async throws -> Void
    public typealias TabReadiness = @MainActor (BrowserTabID) -> Bool
    public typealias UserTakeover = @MainActor (BrowserTabID) -> Void
    public typealias AttachmentSink = @MainActor (AppPromptAttachment, UUID) -> Bool

    @Published public private(set) var isComposerAvailable: Bool
    @Published public private(set) var isPicking = false
    @Published public private(set) var isConfirming = false
    @Published public private(set) var candidate: DOMSnapshotPickCandidate?
    @Published public private(set) var selectedCandidate: DOMSnapshotPickCandidate?
    @Published public private(set) var lastError: ErrorState?

    public var canConfirmSelection: Bool {
        isPicking && !isConfirming && canAttachNow() && selectedCandidate != nil
    }

    private let canAttachNow: @MainActor () -> Bool
    private let selectedChatID: @MainActor () -> UUID?
    private let isTabReady: TabReadiness
    private let prepareForUserInput: UserTakeover
    private let pickCandidateAt: CandidatePicker
    private let validateCandidate: CandidateValidator
    private let attach: AttachmentSink
    private var availabilityObservation: AnyCancellable?
    private var activeTabID: BrowserTabID?
    private var targetChatID: UUID?
    private var activeSessionID: UUID?
    private var activeConfirmationID: UUID?
    private var hoverTask: Task<Void, Never>?
    private var hoverRequestID = UUID()

    public convenience init(engine: WebKitBrowserEngine) {
        self.init(engine: engine, appModel: nil)
    }

    public convenience init(engine: WebKitBrowserEngine, appModel: AppModel) {
        self.init(engine: engine, appModel: Optional(appModel))
    }

    private convenience init(engine: WebKitBrowserEngine, appModel: AppModel?) {
        guard let appModel else {
            self.init(
                initialComposerAvailability: false,
                composerAvailability: Just(false).eraseToAnyPublisher(),
                canAttachNow: { false },
                selectedChatID: { nil },
                isTabReady: { _ in false },
                prepareForUserInput: { _ in },
                pickCandidateAt: { _, _ in nil },
                validateCandidate: { _, _ in },
                attach: { _, _ in false }
            )
            return
        }

        let availability = appModel.$generating
            .combineLatest(appModel.$submitting)
            .map { generating, submitting in !generating && !submitting }
            .removeDuplicates()
            .eraseToAnyPublisher()

        self.init(
            initialComposerAvailability: !appModel.generating && !appModel.submitting,
            composerAvailability: availability,
            canAttachNow: { !appModel.generating && !appModel.submitting },
            selectedChatID: { appModel.selectedChatID },
            isTabReady: { tabID in engine.snapshotService(for: tabID)?.canPickElement == true },
            prepareForUserInput: { tabID in engine.prepareForDirectUserInput(in: tabID) },
            pickCandidateAt: { tabID, point in
                guard let service = engine.snapshotService(for: tabID) else {
                    throw DOMSnapshotServiceError.unavailable
                }
                return try await service.pickCandidate(at: point)
            },
            validateCandidate: { tabID, candidate in
                guard let service = engine.snapshotService(for: tabID) else {
                    throw BrowserControlError.staleReference(reference: candidate.reference)
                }
                try await service.resolve(reference: candidate.reference, generation: candidate.generation)
            },
            attach: { attachment, chatID in
                guard !appModel.generating, !appModel.submitting else { return false }
                appModel.addPromptAttachment(attachment, toChatID: chatID)
                return true
            }
        )
    }

    init(
        initialComposerAvailability: Bool,
        composerAvailability: AnyPublisher<Bool, Never>,
        canAttachNow: @escaping @MainActor () -> Bool,
        selectedChatID: @escaping @MainActor () -> UUID?,
        isTabReady: @escaping TabReadiness,
        prepareForUserInput: @escaping UserTakeover,
        pickCandidateAt: @escaping CandidatePicker,
        validateCandidate: @escaping CandidateValidator,
        attach: @escaping AttachmentSink
    ) {
        self.isComposerAvailable = initialComposerAvailability
        self.canAttachNow = canAttachNow
        self.selectedChatID = selectedChatID
        self.isTabReady = isTabReady
        self.prepareForUserInput = prepareForUserInput
        self.pickCandidateAt = pickCandidateAt
        self.validateCandidate = validateCandidate
        self.attach = attach
        self.availabilityObservation = composerAvailability
            .removeDuplicates()
            .sink { [weak self] available in
                Task { @MainActor [weak self] in
                    self?.isComposerAvailable = available
                }
            }
    }

    public func canStartPicking(in tabID: BrowserTabID) -> Bool {
        canAttachNow() && selectedChatID() != nil && isTabReady(tabID)
    }

    @discardableResult
    public func startPicking(in tabID: BrowserTabID) -> Bool {
        guard canStartPicking(in: tabID), let chatID = selectedChatID() else { return false }
        cancel()
        activeSessionID = UUID()
        activeTabID = tabID
        targetChatID = chatID
        isPicking = true
        prepareForUserInput(tabID)
        return true
    }

    public func updateCandidate(at point: CGPoint) {
        guard isPicking,
              selectedCandidate == nil,
              canAttachNow(),
              let tabID = activeTabID else {
            candidate = nil
            return
        }

        // Never let a click confirm the previous element while this hover is resolving.
        candidate = nil
        lastError = nil
        hoverTask?.cancel()
        let requestID = UUID()
        hoverRequestID = requestID
        hoverTask = Task { @MainActor [weak self] in
            do {
                try await Task.sleep(nanoseconds: 35_000_000)
                guard !Task.isCancelled, let self, self.hoverRequestID == requestID else { return }
                let result = try await self.pickCandidateAt(tabID, point)
                guard !Task.isCancelled,
                      self.hoverRequestID == requestID,
                      self.isPicking,
                      self.selectedCandidate == nil else { return }
                self.candidate = result
                self.lastError = nil
            } catch {
                guard let self, self.hoverRequestID == requestID else { return }
                self.candidate = nil
                self.lastError = .staleCandidate
            }
        }
    }

    public func clearHoverCandidate() {
        guard selectedCandidate == nil else { return }
        hoverTask?.cancel()
        hoverTask = nil
        hoverRequestID = UUID()
        candidate = nil
    }

    func waitForPendingCandidateUpdate() async {
        await hoverTask?.value
    }

    @discardableResult
    public func selectCandidate() -> Bool {
        guard isPicking,
              canAttachNow(),
              let candidate else { return false }
        selectedCandidate = candidate
        lastError = nil
        hoverTask?.cancel()
        hoverTask = nil
        return true
    }

    @discardableResult
    public func attachSelectedCandidate() async -> Bool {
        guard canAttachNow(),
              !isConfirming,
              let sessionID = activeSessionID,
              let tabID = activeTabID,
              let chatID = targetChatID,
              let selectedCandidate else { return false }

        let confirmationID = UUID()
        activeConfirmationID = confirmationID
        isConfirming = true
        defer {
            if activeConfirmationID == confirmationID {
                activeConfirmationID = nil
                isConfirming = false
            }
        }

        do {
            try await validateCandidate(tabID, selectedCandidate)
        } catch {
            guard isPicking,
                  activeSessionID == sessionID,
                  activeConfirmationID == confirmationID,
                  activeTabID == tabID,
                  targetChatID == chatID,
                  self.selectedCandidate == selectedCandidate else { return false }
            lastError = .staleCandidate
            candidate = nil
            self.selectedCandidate = nil
            return false
        }

        guard isPicking,
              activeSessionID == sessionID,
              activeConfirmationID == confirmationID,
              activeTabID == tabID,
              targetChatID == chatID,
              self.selectedCandidate == selectedCandidate,
              canAttachNow() else { return false }
        let attachment = Self.attachment(for: selectedCandidate)
        guard attach(attachment, chatID) else { return false }
        cancel()
        return true
    }

    public func cancel() {
        hoverTask?.cancel()
        hoverTask = nil
        hoverRequestID = UUID()
        activeTabID = nil
        targetChatID = nil
        activeSessionID = nil
        activeConfirmationID = nil
        isConfirming = false
        isPicking = false
        candidate = nil
        selectedCandidate = nil
        lastError = nil
    }

    private static func attachment(for candidate: DOMSnapshotPickCandidate) -> AppPromptAttachment {
        let content = """
        <untrusted_browser_element>
        Reference: \(escapeMarkup(candidate.reference))
        Role: \(escapeMarkup(candidate.role))
        Description: \(escapeMarkup(candidate.name))
        </untrusted_browser_element>
        """
        return AppPromptAttachment(
            fileName: "browser-element.txt",
            formatLabel: "Browser element",
            extractedText: content,
            wasTruncatedDuringExtraction: false
        )
    }

    private static func escapeMarkup(_ value: String) -> String {
        value
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&apos;")
    }
}
