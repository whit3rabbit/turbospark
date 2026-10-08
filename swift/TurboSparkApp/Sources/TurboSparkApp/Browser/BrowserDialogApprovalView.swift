import SwiftUI

@MainActor
struct BrowserDialogApprovalActions {
    let request: BrowserDialogRequest
    let resolve: @MainActor (UUID, BrowserDialogDecision) -> Bool

    @discardableResult
    func accept(promptResponse: String) -> Bool {
        resolve(
            request.id,
            .accept(promptText: request.kind == .prompt ? promptResponse : nil)
        )
    }

    @discardableResult
    func dismiss() -> Bool {
        resolve(request.id, .dismiss)
    }
}

@MainActor
struct BrowserDialogApprovalView: View {
    let request: BrowserDialogRequest
    let actions: BrowserDialogApprovalActions
    @State private var promptResponse: String

    init(
        request: BrowserDialogRequest,
        resolve: @escaping @MainActor (UUID, BrowserDialogDecision) -> Bool
    ) {
        self.request = request
        self.actions = BrowserDialogApprovalActions(request: request, resolve: resolve)
        self._promptResponse = State(initialValue: request.defaultText ?? "")
    }

    var body: some View {
        ZStack {
            Color.black.opacity(0.28)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 16) {
                Text(titleKey, bundle: .module)
                    .themedFont(.base, weight: .semibold)

                // The page that raised the dialog, like Chrome and Safari
                // label theirs: a background tab must not be able to pose as
                // the page the user is looking at.
                Text("Message from \(Self.originLabel(request))", bundle: .module)
                    .themedCode(.small, weight: .semibold)
                    .foregroundStyle(.appSecondary)

                Text(request.message)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)

                if request.kind == .prompt {
                    TextField("", text: $promptResponse)
                        .textFieldStyle(.roundedBorder)
                        .accessibilityLabel(Text("Page prompt", bundle: .module))
                }

                HStack {
                    Button {
                        actions.dismiss()
                    } label: {
                        Text("Dismiss", bundle: .module)
                    }
                    .keyboardShortcut(.cancelAction)

                    Spacer()

                    Button {
                        actions.accept(promptResponse: promptResponse)
                    } label: {
                        Text("Accept", bundle: .module)
                    }
                    .keyboardShortcut(.defaultAction)
                }
            }
            .padding(20)
            .frame(maxWidth: 440, alignment: .leading)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
            .shadow(radius: 24)
            .padding(24)
            .accessibilityAddTraits(.isModal)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityIdentifier("browser-dialog-approval")
    }

    /// The origin shown in the dialog; `?` when the engine could not name
    /// one, which is itself a signal worth surfacing.
    static func originLabel(_ request: BrowserDialogRequest) -> String {
        request.sourceOrigin?.canonicalString ?? "?"
    }

    private var titleKey: LocalizedStringKey {
        switch request.kind {
        case .alert: "Page alert"
        case .confirm: "Page confirmation"
        case .prompt: "Page prompt"
        }
    }
}
