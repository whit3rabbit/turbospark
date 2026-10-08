import AppKit
import Combine
import SwiftUI
import WebKit

@MainActor
final class BrowserPaneModel: ObservableObject {
    @Published private(set) var tabs: [BrowserTab] = []
    @Published private(set) var selectedTabID: BrowserTabID?
    @Published private(set) var selectedTabState: BrowserEngineTabState?
    @Published private(set) var canGoBack = false
    @Published private(set) var canGoForward = false
    @Published private(set) var isAddressInvalid = false
    @Published private(set) var addressText = ""

    let engine: WebKitBrowserEngine

    private let externalBrowserOpener: (URL) -> Void
    private var isEditingAddress = false

    init(
        engine: WebKitBrowserEngine,
        externalBrowserOpener: @escaping (URL) -> Void = { _ = NSWorkspace.shared.open($0) }
    ) {
        self.engine = engine
        self.externalBrowserOpener = externalBrowserOpener
        self.tabs = engine.tabStore.tabs
        self.selectedTabID = engine.tabStore.activeTabID
        refresh()
    }

    func refresh() {
        let store = engine.tabStore
        tabs = store.tabs
        if let selectedTabID, !tabs.contains(where: { $0.id == selectedTabID }) {
            self.selectedTabID = store.activeTabID ?? tabs.first?.id
        } else if selectedTabID == nil {
            selectedTabID = store.activeTabID ?? tabs.first?.id
        }

        guard let tabID = selectedTabID else {
            selectedTabState = nil
            canGoBack = false
            canGoForward = false
            if !isEditingAddress {
                addressText = ""
            }
            return
        }

        selectedTabState = engine.tabStates[tabID]
        if let webView = engine.webView(for: tabID) {
            canGoBack = webView.canGoBack
            canGoForward = webView.canGoForward
        } else {
            canGoBack = false
            canGoForward = false
        }
        if !isEditingAddress {
            addressText = selectedTabState?.address ?? store.tab(id: tabID)?.address ?? ""
        }
    }

    func isAgentControlled(tabID: BrowserTabID) -> Bool {
        engine.tabStore.agentControlledTabID == tabID
    }

    @discardableResult
    func createTab() -> BrowserTabID {
        let tabID = engine.createTab(owner: .user)
        selectedTabID = tabID
        isEditingAddress = false
        isAddressInvalid = false
        refresh()
        return tabID
    }

    func selectTab(_ tabID: BrowserTabID) {
        guard tabs.contains(where: { $0.id == tabID }) else { return }
        do {
            try engine.tabStore.selectTab(tabID)
        } catch {
            return
        }
        selectedTabID = tabID
        isEditingAddress = false
        isAddressInvalid = false
        refresh()
    }

    func closeSelectedTab() {
        guard let selectedTabID else { return }
        closeTab(selectedTabID)
    }

    func closeTab(_ tabID: BrowserTabID) {
        guard tabs.contains(where: { $0.id == tabID }) else { return }
        let closesSelection = selectedTabID == tabID
        do {
            try engine.closeTab(tabID)
        } catch {
            refresh()
            return
        }
        if closesSelection {
            selectedTabID = engine.tabStore.activeTabID ?? engine.tabStore.tabs.first?.id
            isEditingAddress = false
            isAddressInvalid = false
        }
        refresh()
    }

    func updateAddressDraft(_ value: String) {
        addressText = value
        isEditingAddress = true
        isAddressInvalid = false
    }

    @discardableResult
    func submitAddress() -> Bool {
        guard let selectedTabID,
              let destination = addressDestination(from: addressText) else {
            isAddressInvalid = true
            return false
        }

        do {
            try engine.navigateAsUser(to: destination.url, in: selectedTabID)
        } catch {
            isAddressInvalid = true
            return false
        }
        isAddressInvalid = false
        isEditingAddress = false
        refresh()
        return true
    }

    func goBack() {
        performDirectUserAction { _ = $0.goBack() }
    }

    func goForward() {
        performDirectUserAction { _ = $0.goForward() }
    }

    func reload() {
        performDirectUserAction { _ = $0.reload() }
    }

    func stopLoading() {
        guard let selectedTabID else { return }
        engine.prepareForDirectUserInput(in: selectedTabID)
        engine.stopLoading(in: selectedTabID)
        refresh()
    }

    func prepareForViewportInput() {
        guard let selectedTabID,
              engine.tabStore.tab(id: selectedTabID)?.owner == .agent else { return }
        engine.prepareForDirectUserInput(in: selectedTabID)
        refresh()
    }

    @discardableResult
    func retryOrRecover() -> Bool {
        guard let selectedTabID else { return false }
        if case .crashed? = selectedTabState?.loadState {
            do {
                // The user's click is direct input: take the tab over first so
                // the reload is not judged as an agent navigation.
                engine.prepareForDirectUserInput(in: selectedTabID)
                try engine.recoverCrashedTab(selectedTabID)
                refresh()
                return true
            } catch {
                return false
            }
        }
        guard let address = selectedTabState?.address ?? engine.tabStore.tab(id: selectedTabID)?.address,
              let destination = addressDestination(from: address) else {
            return false
        }
        do {
            try engine.navigateAsUser(to: destination.url, in: selectedTabID)
            refresh()
            return true
        } catch {
            return false
        }
    }

    @discardableResult
    func openInSystemBrowser() -> Bool {
        guard let selectedTabID,
              let address = selectedTabState?.address ?? engine.tabStore.tab(id: selectedTabID)?.address,
              let destination = addressDestination(from: address) else {
            return false
        }
        engine.prepareForDirectUserInput(in: selectedTabID)
        externalBrowserOpener(destination.url)
        refresh()
        return true
    }

    private func performDirectUserAction(_ action: (WKWebView) -> Void) {
        guard let selectedTabID, let webView = engine.webView(for: selectedTabID) else { return }
        engine.prepareForDirectUserInput(in: selectedTabID)
        action(webView)
        refresh()
    }

    private func addressDestination(from address: String) -> BrowserNavigationDestination? {
        let trimmed = address.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed) else { return nil }
        return BrowserNavigationDestination(url: url)
    }
}

enum BrowserTabPresentation: Equatable {
    case content
    case loading(progress: Double)
    case failed(message: String)
    case crashed(message: String)
}

@MainActor
struct BrowserTabView: View {
    let engine: WebKitBrowserEngine
    @ObservedObject var viewportController: BrowserViewportController
    @ObservedObject var elementPicker: ElementPickerController
    let tabID: BrowserTabID
    let state: BrowserEngineTabState?
    let isAgentControlled: Bool
    let onRetry: () -> Void
    let onRecover: () -> Void

    var body: some View {
        ZStack(alignment: .topTrailing) {
            if let webView = engine.webView(for: tabID) {
                BrowserViewportWebView(controller: viewportController, webView: webView)
            } else {
                Color.clear
            }

            if elementPicker.isPicking, case .content = Self.presentation(for: state) {
                elementPickerHitLayer
                    .zIndex(2)
            }

            switch Self.presentation(for: state) {
            case .content:
                EmptyView()
            case .loading(let progress):
                ProgressView(value: progress)
                    .frame(maxWidth: 180)
                    .padding(12)
                    .background(.regularMaterial, in: Capsule())
                    .accessibilityLabel(Text("Loading", bundle: .module))
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .padding(12)
            case .failed(let message):
                failureCard(message: message, crashed: false)
            case .crashed(let message):
                failureCard(message: message, crashed: true)
            }

            if isAgentControlled {
                Label {
                    Text("Agent controlled", bundle: .module)
                } icon: {
                    Image(systemName: "hand.raised.fill")
                }
                .themedFont(.small)
                .padding(8)
                .background(.regularMaterial, in: Capsule())
                .padding(12)
            }
        }
    }

    private var elementPickerHitLayer: some View {
        GeometryReader { geometry in
            let scale = viewportController.isFixed ? CGFloat(viewportController.preference.zoom) : 1
            ZStack(alignment: .topLeading) {
                Color.clear
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let location):
                            elementPicker.updateCandidate(at: BrowserViewportController.cssPoint(
                                viewPoint: location,
                                scrollOffset: viewportController.scrollOffset,
                                scale: scale))
                        case .ended:
                            elementPicker.clearHoverCandidate()
                        }
                    }
                    .onTapGesture {
                        _ = elementPicker.selectCandidate()
                    }

                if let candidate = elementPicker.selectedCandidate ?? elementPicker.candidate {
                    let bounds = candidate.bounds
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.accentColor.opacity(0.12))
                        .overlay {
                            RoundedRectangle(cornerRadius: 4)
                                .stroke(Color.accentColor, lineWidth: 2)
                        }
                        .frame(
                            width: max(1, bounds.width * scale),
                            height: max(1, bounds.height * scale)
                        )
                        .position(
                            x: (bounds.x + bounds.width / 2) * scale - viewportController.scrollOffset.x,
                            y: (bounds.y + bounds.height / 2) * scale - viewportController.scrollOffset.y
                        )
                        .allowsHitTesting(false)
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
    }

    static func presentation(for state: BrowserEngineTabState?) -> BrowserTabPresentation {
        guard let state else { return .content }
        switch state.loadState {
        case .loading:
            let progress = state.progress.isFinite ? min(max(state.progress, 0), 1) : 0
            return .loading(progress: progress)
        case .failed(let reason):
            return .failed(message: state.loadError ?? reason)
        case .crashed:
            return .crashed(message: state.loadError ?? "")
        case .idle, .loaded, .restored:
            return .content
        }
    }

    @ViewBuilder
    private func failureCard(message: String, crashed: Bool) -> some View {
        VStack(spacing: 12) {
            if !message.isEmpty {
                Text(message)
                    .multilineTextAlignment(.center)
                    .textSelection(.enabled)
            }
            Button {
                if crashed {
                    onRecover()
                } else {
                    onRetry()
                }
            } label: {
                Label {
                    Text("Retry", bundle: .module)
                } icon: {
                    Image(systemName: "arrow.clockwise")
                }
            }
        }
        .padding(20)
        .frame(maxWidth: 360)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

@MainActor
public struct BrowserPaneView: View {
    @ObservedObject private var engine: WebKitBrowserEngine
    @StateObject private var model: BrowserPaneModel
    @StateObject private var viewportController: BrowserViewportController
    @StateObject private var elementPicker: ElementPickerController
    @State private var isViewportPopoverPresented = false

    public init(engine: WebKitBrowserEngine) {
        self.engine = engine
        self._model = StateObject(wrappedValue: BrowserPaneModel(engine: engine))
        self._viewportController = StateObject(wrappedValue: BrowserViewportController())
        self._elementPicker = StateObject(wrappedValue: ElementPickerController(engine: engine))
    }

    public init(engine: WebKitBrowserEngine, appModel: AppModel) {
        self.engine = engine
        self._model = StateObject(wrappedValue: BrowserPaneModel(engine: engine))
        self._viewportController = StateObject(wrappedValue: BrowserViewportController())
        self._elementPicker = StateObject(
            wrappedValue: ElementPickerController(engine: engine, appModel: appModel)
        )
    }

    public var body: some View {
        VStack(spacing: 0) {
            tabStrip
            browserToolbar
            if elementPicker.isPicking {
                elementPickerBar
            }
            if model.isAddressInvalid {
                Text("Enter a valid HTTP or HTTPS address.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 8)
            }
            if let tabID = model.selectedTabID {
                BrowserTabView(
                    engine: engine,
                    viewportController: viewportController,
                    elementPicker: elementPicker,
                    tabID: tabID,
                    state: model.selectedTabState,
                    isAgentControlled: model.isAgentControlled(tabID: tabID),
                    onRetry: { _ = model.retryOrRecover() },
                    onRecover: { _ = model.retryOrRecover() }
                )
                .id(tabID)
            } else {
                Button {
                    _ = model.createTab()
                } label: {
                    Label {
                        Text("New Tab", bundle: .module)
                    } icon: {
                        Image(systemName: "plus")
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear { model.refresh() }
        // @Published emits in willSet, so a synchronous refresh would read the
        // PRE-change tabStates. Hop to the next main-queue turn to see the new value.
        .onReceive(engine.$tabStates.receive(on: DispatchQueue.main)) { _ in model.refresh() }
        .onChange(of: model.selectedTabID) { _, _ in elementPicker.cancel() }
        // Show the page that raised a dialog before presenting it, so a
        // background tab cannot draw a prompt over the page being viewed.
        .onChange(of: engine.pendingDialogs.first?.tabID, initial: true) { _, tabID in
            if let tabID { model.selectTab(tabID) }
        }
        .overlay {
            if let request = engine.pendingDialogs.first {
                BrowserDialogApprovalView(request: request) { requestID, decision in
                    engine.resolvePendingDialog(requestID, decision: decision)
                }
                // Fresh view state per dialog: otherwise text typed into one
                // prompt carries over to the next queued dialog's origin.
                .id(request.id)
                .zIndex(4)
            }
        }
    }

    private var tabStrip: some View {
        HStack(spacing: 8) {
            ScrollView(.horizontal) {
                HStack(spacing: 6) {
                    ForEach(model.tabs) { tab in
                        tabButton(tab)
                    }
                }
                .padding(.horizontal, 8)
            }
            .scrollIndicators(.hidden)

            Button {
                _ = model.createTab()
            } label: {
                Image(systemName: "plus")
            }
            .buttonStyle(.borderless)
            .help(Text("New Tab", bundle: .module))
            .accessibilityLabel(Text("New Tab", bundle: .module))
        }
        .padding(.vertical, 6)
        .background(.bar)
    }

    private func tabButton(_ tab: BrowserTab) -> some View {
        HStack(spacing: 4) {
            Button {
                elementPicker.cancel()
                model.selectTab(tab.id)
            } label: {
                HStack(spacing: 6) {
                    if model.isAgentControlled(tabID: tab.id) {
                        Image(systemName: "hand.raised.fill")
                            .help(Text("Agent controlled", bundle: .module))
                            .accessibilityLabel(Text("Agent controlled", bundle: .module))
                    }
                    if let title = tab.title, !title.isEmpty {
                        Text(title)
                            .lineLimit(1)
                    } else {
                        Text("New Tab", bundle: .module)
                            .lineLimit(1)
                    }
                }
                .frame(maxWidth: 180, alignment: .leading)
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(
                    model.selectedTabID == tab.id ? Color.accentColor.opacity(0.18) : Color.clear,
                    in: RoundedRectangle(cornerRadius: 6)
                )
            }
            .buttonStyle(.plain)

            Button {
                model.closeTab(tab.id)
            } label: {
                Image(systemName: "xmark")
                    .themedFont(.tiny)
            }
            .buttonStyle(.borderless)
            .help(Text("Close Tab", bundle: .module))
            .accessibilityLabel(Text("Close Tab", bundle: .module))
        }
    }

    private var browserToolbar: some View {
        HStack(spacing: 8) {
            Button { elementPicker.cancel(); model.goBack() } label: { Image(systemName: "chevron.backward") }
                .help(Text("Previous", bundle: .module))
                .accessibilityLabel(Text("Previous", bundle: .module))
                .disabled(!model.canGoBack)
            Button { elementPicker.cancel(); model.goForward() } label: { Image(systemName: "chevron.forward") }
                .help(Text("Next", bundle: .module))
                .accessibilityLabel(Text("Next", bundle: .module))
                .disabled(!model.canGoForward)
            Button { elementPicker.cancel(); model.reload() } label: { Image(systemName: "arrow.clockwise") }
                .help(Text("Reload", bundle: .module))
                .accessibilityLabel(Text("Reload", bundle: .module))
                .disabled(model.selectedTabID == nil)
            Button { elementPicker.cancel(); model.stopLoading() } label: { Image(systemName: "xmark") }
                .help(Text("Stop", bundle: .module))
                .accessibilityLabel(Text("Stop", bundle: .module))
                .disabled(model.selectedTabState?.loadState != .loading)

            TextField(
                text: Binding(
                    get: { model.addressText },
                    set: { model.updateAddressDraft($0) }
                )
            ) {
                Text("Address", bundle: .module)
            }
            .textFieldStyle(.roundedBorder)
            .onSubmit { elementPicker.cancel(); _ = model.submitAddress() }
            .disabled(model.selectedTabID == nil)

            Button { elementPicker.cancel(); _ = model.openInSystemBrowser() } label: {
                Image(systemName: "safari")
            }
            .help(Text("Open in System Browser", bundle: .module))
            .accessibilityLabel(Text("Open in System Browser", bundle: .module))
            .disabled(model.selectedTabID == nil)

            if elementPicker.isPicking {
                Button {
                    elementPicker.cancel()
                } label: {
                    Image(systemName: "xmark.circle")
                }
                .help(Text("Cancel", bundle: .module))
                .accessibilityLabel(Text("Cancel", bundle: .module))
            } else {
                Button {
                    if let tabID = model.selectedTabID {
                        _ = elementPicker.startPicking(in: tabID)
                    }
                } label: {
                    Image(systemName: "scope")
                }
                .help(Text("Pick Element", bundle: .module))
                .accessibilityLabel(Text("Pick Element", bundle: .module))
                .disabled(model.selectedTabID.map { !elementPicker.canStartPicking(in: $0) } ?? true)
            }

            Button {
                isViewportPopoverPresented.toggle()
            } label: {
                Image(systemName: "aspectratio")
            }
            .help(Text("Viewport", bundle: .module))
            .accessibilityLabel(Text("Viewport", bundle: .module))
            .popover(isPresented: $isViewportPopoverPresented, arrowEdge: .bottom) {
                viewportControls
            }
        }
        .buttonStyle(.borderless)
        .padding(8)
    }

    private var elementPickerBar: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                if elementPicker.lastError == .staleCandidate {
                    Text("This element changed. Pick it again.", bundle: .module)
                        .lineLimit(1)
                } else if let candidate = elementPicker.selectedCandidate ?? elementPicker.candidate {
                    Text(verbatim: "\(candidate.role): \(candidate.name)")
                        .lineLimit(1)
                        .truncationMode(.middle)
                } else {
                    Text("Click a highlighted element to select it.", bundle: .module)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 8)
            Button {
                Task { _ = await elementPicker.attachSelectedCandidate() }
            } label: {
                Text("Attach", bundle: .module)
            }
            .disabled(!elementPicker.canConfirmSelection)
            Button {
                elementPicker.cancel()
            } label: {
                Text("Cancel", bundle: .module)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
    }

    private var viewportControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(
                isOn: Binding(
                    get: { viewportController.isFixed },
                    set: {
                        elementPicker.cancel()
                        model.prepareForViewportInput()
                        viewportController.setMode($0 ? .fixed : .responsive)
                    }
                )
            ) {
                Text("Fixed CSS viewport", bundle: .module)
            }

            Divider()

            Stepper(
                value: Binding(
                    get: { viewportController.preference.width },
                    set: {
                        elementPicker.cancel()
                        model.prepareForViewportInput()
                        viewportController.updatePreference(width: $0)
                    }
                ),
                in: BrowserViewportController.widthRange,
                step: 80
            ) {
                HStack {
                    Text("Width", bundle: .module)
                    Spacer()
                    Text(verbatim: "\(viewportController.preference.width) px")
                }
            }
            .disabled(!viewportController.isFixed)

            Stepper(
                value: Binding(
                    get: { viewportController.preference.height },
                    set: {
                        elementPicker.cancel()
                        model.prepareForViewportInput()
                        viewportController.updatePreference(height: $0)
                    }
                ),
                in: BrowserViewportController.heightRange,
                step: 80
            ) {
                HStack {
                    Text("Height", bundle: .module)
                    Spacer()
                    Text(verbatim: "\(viewportController.preference.height) px")
                }
            }
            .disabled(!viewportController.isFixed)

            Stepper(
                value: Binding(
                    get: { viewportController.preference.zoom },
                    set: {
                        elementPicker.cancel()
                        model.prepareForViewportInput()
                        viewportController.updatePreference(zoom: $0)
                    }
                ),
                in: BrowserViewportController.zoomRange,
                step: 0.1
            ) {
                HStack {
                    Text("Zoom", bundle: .module)
                    Spacer()
                    Text(verbatim: "\(viewportController.preference.zoom.formatted(.number.precision(.fractionLength(1))))\u{00D7}")
                }
            }
            .disabled(!viewportController.isFixed)
        }
        .padding(14)
        .frame(width: 260)
    }
}
