import AppKit
import SwiftUI
import TurboSpark

/// Hugging Face authentication token management card with live whoami-v2 validation.
/// Follows the UX conventions of Unsloth: masked monospace input with reveal toggle,
/// debounced status validation against the Hugging Face API, and user identification badge.
// Isolated explicitly: its helpers call each other, and only `body` is
// isolated by the protocol on the macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
public struct HfAuthTokenCardView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject public var model: AppModel

    @State private var tokenInput: String = ""
    @State private var isRevealed: Bool = false
    @State private var savedToken: String? = nil
    @State private var tokenSource: String? = nil
    @State private var validationStatus: HfTokenValidationStatus? = nil
    @State private var isValidating: Bool = false
    @State private var validationTask: Task<Void, Never>? = nil
    @State private var showingRemoveConfirmation: Bool = false

    public init(model: AppModel) {
        self.model = model
    }

    private var isDirty: Bool {
        let current = tokenInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let saved = savedToken?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return current != saved
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            headerRow
            tokenInputRow
            statusRow
            footerActionsRow
            Divider().padding(.vertical, 2)
            HfMirrorEndpointRowView(model: model)
        }
        // The same card as every other Models & Storage section, instead
        // of an inline copy that had already drifted to its own padding.
        .modelsSettingsCard()
        .confirmationDialog(
            Text("Remove Token", bundle: .module),
            isPresented: $showingRemoveConfirmation,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                clearToken()
            } label: { Text("Remove Token", bundle: .module) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("This removes the saved Hugging Face token from this Mac.", bundle: .module)
        }
        .onAppear(perform: loadSavedToken)
    }

    // MARK: - Header
    private var headerRow: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text("Hugging Face API Token", bundle: .module)
                        .settingsControl("Hugging Face API Token", pane: .models, timing: .action)
                        .font(theme.ui(.base, weight: .semibold))
                        .accessibilityAddTraits(.isHeader)
                    if savedToken != nil {
                        Text("Configured", bundle: .module)
                            .font(theme.ui(.tiny, weight: .medium))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(theme.accent.opacity(0.15))
                            .foregroundStyle(theme.accent)
                            .clipShape(Capsule())

                        if let source = tokenSource {
                            // The source is an identifier (environment
                            // variable, CLI cache path), not prose.
                            Text("via \(source)", bundle: .module)
                                .font(theme.ui(.tiny))
                                .foregroundStyle(.appSecondary)
                        }
                    }
                }
                Text("Authenticate to inspect and download gated or private repositories.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Link(destination: URL(string: "https://huggingface.co/settings/tokens")!) {
                HStack(spacing: 4) {
                    Text("Get Token", bundle: .module)
                        .font(theme.ui(.small))
                    Image(systemName: "arrow.up.forward.app")
                        .themedFont(.tiny)
                }
            }
            .buttonStyle(.link)
            .appPointerCursor()
        }
    }

    // MARK: - Input Field
    private var tokenInputRow: some View {
        HStack(spacing: 8) {
            HStack {
                // The placeholder is the token's literal prefix, which has
                // nothing to translate; the label is the VoiceOver name.
                if isRevealed {
                    TextField(text: $tokenInput, prompt: Text(verbatim: "hf_...")) {
                        Text("Hugging Face API Token", bundle: .module)
                    }
                    .textFieldStyle(.plain)
                    .font(theme.code(.base))
                } else {
                    SecureField(text: $tokenInput, prompt: Text(verbatim: "hf_...")) {
                        Text("Hugging Face API Token", bundle: .module)
                    }
                    .textFieldStyle(.plain)
                    .font(theme.code(.base))
                }

                AppIconButton(
                    isRevealed ? "eye.slash" : "eye",
                    label: isRevealed ? Text("Hide token", bundle: .module) : Text("Reveal token", bundle: .module)
                ) {
                    isRevealed.toggle()
                }
                .buttonStyle(.plain)
                .foregroundStyle(.appSecondary)
                .appPointerCursor()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .stroke(.appBorder, lineWidth: 1)
            )
            .onChange(of: tokenInput) {
                scheduleDebouncedValidation()
            }

            if !tokenInput.isEmpty {
                AppIconButton("xmark.circle.fill", label: Text("Clear field", bundle: .module)) {
                    tokenInput = ""
                    scheduleDebouncedValidation()
                }
                .buttonStyle(.plain)
                .foregroundStyle(.appSecondary)
                .appPointerCursor()
            }
        }
    }

    // MARK: - Live Validation Status
    @ViewBuilder
    private var statusRow: some View {
        if isValidating {
            HStack(spacing: 6) {
                AppProgressBar(value: nil, label: Text("Validating token with Hugging Face...", bundle: .module), size: .compact)
                Text("Validating token with Hugging Face...", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
        } else if let status = validationStatus {
            switch status {
            case .valid(let name, let fullname, let email):
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(theme.accent)
                    VStack(alignment: .leading, spacing: 1) {
                        HStack(spacing: 4) {
                            Text("Valid token", bundle: .module)
                                .font(theme.ui(.small, weight: .semibold))
                                .foregroundStyle(theme.accent)
                            if let name, !name.isEmpty {
                                Text("signed in as @\(name)", bundle: .module)
                                    .font(theme.ui(.small))
                                    .foregroundStyle(.appSecondary)
                            }
                        }
                        if let detail = userDetailString(fullname: fullname, email: email) {
                            Text(detail)
                                .font(theme.ui(.tiny))
                                .foregroundStyle(.appSecondary)
                        }
                    }
                }
            case .invalid(let message):
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(AppAttention.color)
                    Group {
                        // The service's own message wins when it sent one.
                        if let message {
                            Text(verbatim: message)
                        } else {
                            Text("Invalid token. Hugging Face rejected these credentials.", bundle: .module)
                        }
                    }
                    .font(theme.ui(.small))
                    .foregroundStyle(AppAttention.color)
                }
            case .rateLimited(let retryAfter):
                HStack(spacing: 6) {
                    Image(systemName: "clock.arrow.circlepath")
                        .foregroundStyle(AppAttention.color)
                    // `Int`, not the payload's `UInt64`: the catalog key is
                    // spelled `%lld`, and a `UInt64` interpolates as `%llu`,
                    // which matched no key and rendered English everywhere.
                    Text("Hugging Face API rate limit reached. Retry in \(Int(retryAfter ?? 60))s.", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(AppAttention.color)
                }
            case .unavailable(let message):
                HStack(spacing: 6) {
                    Image(systemName: "wifi.slash")
                        .foregroundStyle(.appSecondary)
                    Text("Could not reach Hugging Face: \(message)", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                }
            case .missing:
                EmptyView()
            }
        }
    }

    // MARK: - Actions
    private var footerActionsRow: some View {
        HStack {
            Button {
                saveToken()
            } label: { Text("Save Token", bundle: .module) }
            .appAction(.primary, size: .small)
            .disabled(tokenInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !isDirty || isValidating)
            .help(Text("Save Token", bundle: .module))
            .appPointerCursor()

            if savedToken != nil {
                Button(role: .destructive) {
                    showingRemoveConfirmation = true
                } label: { Text("Remove Token", bundle: .module) }
                .appAction(.destructive, size: .small)
                .appPointerCursor()
            }

            Spacer()
        }
    }

    // MARK: - Helpers & Data Flow
    private func userDetailString(fullname: String?, email: String?) -> String? {
        let parts = [fullname, email].compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        return parts.isEmpty ? nil : parts.joined(separator: " - ")
    }

    private func loadSavedToken() {
        if let info = try? TurboSparkCatalog.getHfTokenInfo() {
            savedToken = info.token
            tokenSource = info.source
            tokenInput = info.token
            triggerValidation(info.token)
        } else if let token = try? TurboSparkCatalog.getHfToken(), !token.isEmpty {
            savedToken = token
            tokenSource = nil
            tokenInput = token
            triggerValidation(token)
        } else {
            savedToken = nil
            tokenSource = nil
        }
    }

    private func scheduleDebouncedValidation() {
        validationTask?.cancel()
        let candidate = tokenInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !candidate.isEmpty else {
            validationStatus = .missing
            isValidating = false
            return
        }

        isValidating = true
        validationTask = Task {
            try? await Task.sleep(nanoseconds: 500_000_000)
            guard !Task.isCancelled else { return }
            triggerValidation(candidate)
        }
    }

    private func triggerValidation(_ token: String) {
        isValidating = true
        Task.detached {
            let status = (try? TurboSparkCatalog.validateHfToken(token)) ?? .unavailable(message: AppStrings.localized("Network request failed"))
            await MainActor.run {
                self.validationStatus = status
                self.isValidating = false
            }
        }
    }

    private func saveToken() {
        let trimmed = tokenInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        do {
            try TurboSparkCatalog.setHfToken(trimmed)
            savedToken = trimmed
            model.showToast(AppStrings.localized("Hugging Face API token saved"), style: .success)
            // Re-resolve: HF_TOKEN or a huggingface-cli token can outrank the
            // store, so show the token that is actually in effect.
            loadSavedToken()
        } catch {
            model.showToast(AppStrings.localized("Failed to save token: \(error.localizedDescription)"), style: .error)
        }
    }

    private func clearToken() {
        do {
            try TurboSparkCatalog.clearHfToken()
            savedToken = nil
            tokenInput = ""
            validationStatus = nil
            // Re-resolve: removing the stored token can expose an environment
            // or huggingface-cli token that is still sent on installs.
            loadSavedToken()
            if savedToken != nil {
                model.showToast(
                    AppStrings.localized("Stored token removed, but another token is still in effect (environment variable or Hugging Face CLI cache)."),
                    style: .info)
            } else {
                model.showToast(AppStrings.localized("Hugging Face token removed"), style: .info)
            }
        } catch {
            model.showToast(AppStrings.localized("Failed to clear token: \(error.localizedDescription)"), style: .error)
        }
    }
}
