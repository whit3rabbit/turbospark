import AppKit
import SwiftUI
import TurboSpark

/// Hugging Face authentication token management card with live whoami-v2 validation.
/// Follows the UX conventions of Unsloth: masked monospace input with reveal toggle,
/// debounced status validation against the Hugging Face API, and user identification badge.
public struct HfAuthTokenCardView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject public var model: AppModel

    @State private var tokenInput: String = ""
    @State private var isRevealed: Bool = false
    @State private var savedToken: String? = nil
    @State private var tokenSource: String? = nil
    @State private var mirrorEndpointInput: String = ""
    @State private var savedMirrorEndpoint: String = ""
    @State private var validationStatus: HfTokenValidationStatus? = nil
    @State private var isValidating: Bool = false
    @State private var validationTask: Task<Void, Never>? = nil

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
            Divider().padding(.vertical, 2)
            mirrorEndpointRow
            footerActionsRow
        }
        .padding(16)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .onAppear {
            loadSavedToken()
            loadMirrorEndpoint()
        }
    }

    // MARK: - Header
    private var headerRow: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text("Hugging Face API Token", bundle: .module)
                        .font(theme.ui(.base, weight: .semibold))
                    if savedToken != nil {
                        Text("Configured", bundle: .module)
                            .font(theme.ui(.tiny, weight: .medium))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.15))
                            .foregroundStyle(.green)
                            .clipShape(Capsule())

                        if let source = tokenSource {
                            Text("via \(source)", bundle: .module)
                                .font(theme.ui(.tiny))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                Text("Authenticate to inspect and download gated or private repositories.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Link(destination: URL(string: "https://huggingface.co/settings/tokens")!) {
                HStack(spacing: 4) {
                    Text("Get Token", bundle: .module)
                        .font(theme.ui(.small))
                    Image(systemName: "arrow.up.forward.app")
                        .themedFont(points: 11)
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
                if isRevealed {
                    TextField("hf_...", text: $tokenInput)
                        .textFieldStyle(.plain)
                        .font(theme.code(.base))
                } else {
                    SecureField("hf_...", text: $tokenInput)
                        .textFieldStyle(.plain)
                        .font(theme.code(.base))
                }

                Button {
                    isRevealed.toggle()
                } label: {
                    Image(systemName: isRevealed ? "eye.slash" : "eye")
                        .foregroundStyle(.secondary)
                        .themedFont(points: 13)
                }
                .buttonStyle(.plain)
                .help(isRevealed ? "Hide token" : "Reveal token")
                .appPointerCursor()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
            )
            .onChange(of: tokenInput) {
                scheduleDebouncedValidation()
            }

            if !tokenInput.isEmpty {
                Button {
                    tokenInput = ""
                    scheduleDebouncedValidation()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Clear field")
                .appPointerCursor()
            }
        }
    }

    // MARK: - Live Validation Status
    @ViewBuilder
    private var statusRow: some View {
        if isValidating {
            HStack(spacing: 6) {
                ProgressView()
                    .controlSize(.small)
                Text("Validating token with Hugging Face...", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.secondary)
            }
        } else if let status = validationStatus {
            switch status {
            case .valid(let name, let fullname, let email):
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    VStack(alignment: .leading, spacing: 1) {
                        HStack(spacing: 4) {
                            Text("Valid token", bundle: .module)
                                .font(theme.ui(.small, weight: .semibold))
                                .foregroundStyle(.green)
                            if let name, !name.isEmpty {
                                Text("signed in as @\(name)", bundle: .module)
                                    .font(theme.ui(.small))
                                    .foregroundStyle(.secondary)
                            }
                        }
                        if let detail = userDetailString(fullname: fullname, email: email) {
                            Text(detail)
                                .font(theme.ui(.tiny))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            case .invalid(let message):
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(message ?? "Invalid token. Hugging Face rejected these credentials.")
                        .font(theme.ui(.small))
                        .foregroundStyle(.red)
                }
            case .rateLimited(let retryAfter):
                HStack(spacing: 6) {
                    Image(systemName: "clock.arrow.circlepath")
                        .foregroundStyle(.orange)
                    Text("Hugging Face API rate limit reached. Retry in \(retryAfter ?? 60)s.", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.orange)
                }
            case .unavailable(let message):
                HStack(spacing: 6) {
                    Image(systemName: "wifi.slash")
                        .foregroundStyle(.secondary)
                    Text("Could not reach Hugging Face: \(message)", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.secondary)
                }
            case .missing:
                EmptyView()
            }
        }
    }

    // MARK: - Mirror Endpoint
    private var mirrorEndpointRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Mirror Endpoint ($HF_ENDPOINT)", bundle: .module)
                    .font(theme.ui(.small, weight: .medium))
                Spacer()
                if !savedMirrorEndpoint.isEmpty && savedMirrorEndpoint != "https://huggingface.co" {
                    Button("Reset to Default") {
                        resetMirrorEndpoint()
                    }
                    .buttonStyle(.plain)
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.secondary)
                    .appPointerCursor()
                }
            }
            HStack(spacing: 8) {
                TextField("https://huggingface.co", text: $mirrorEndpointInput)
                    .textFieldStyle(.plain)
                    .font(theme.code(.base))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(Color(nsColor: .textBackgroundColor).opacity(0.8))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                    )

                Button("Save Mirror") {
                    saveMirrorEndpoint()
                }
                .buttonStyle(.bordered)
                .disabled(mirrorEndpointInput.trimmingCharacters(in: .whitespacesAndNewlines) == savedMirrorEndpoint)
                .appPointerCursor()
            }
            Text("Useful for regions with restricted Hugging Face access, e.g. https://hf-mirror.com", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Actions
    private var footerActionsRow: some View {
        HStack {
            Button("Save Token") {
                saveToken()
            }
            .buttonStyle(.borderedProminent)
            .disabled(tokenInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !isDirty || isValidating)
            .appPointerCursor()

            if savedToken != nil {
                Button("Remove Token", role: .destructive) {
                    clearToken()
                }
                .buttonStyle(.bordered)
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
            let status = (try? TurboSparkCatalog.validateHfToken(token)) ?? .unavailable(message: "Network request failed")
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
            model.showToast("Hugging Face API token saved", style: .success)
        } catch {
            model.showToast("Failed to save token: \(error.localizedDescription)", style: .error)
        }
    }

    private func clearToken() {
        do {
            try TurboSparkCatalog.clearHfToken()
            savedToken = nil
            tokenInput = ""
            validationStatus = nil
            model.showToast("Hugging Face token removed", style: .info)
        } catch {
            model.showToast("Failed to clear token: \(error.localizedDescription)", style: .error)
        }
    }

    private func loadMirrorEndpoint() {
        let endpoint = model.hfEndpointInput.trimmingCharacters(in: .whitespacesAndNewlines)
        savedMirrorEndpoint = endpoint.isEmpty ? "https://huggingface.co" : endpoint
        mirrorEndpointInput = savedMirrorEndpoint
    }

    private func saveMirrorEndpoint() {
        let trimmed = mirrorEndpointInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let effective = trimmed.isEmpty ? "https://huggingface.co" : trimmed
        model.hfEndpointInput = effective
        savedMirrorEndpoint = effective
        do {
            try TurboSparkCatalog.setHfEndpoint(effective == "https://huggingface.co" ? nil : effective)
            model.showToast("Mirror endpoint updated", style: .success)
        } catch {
            model.showToast("Failed to set mirror endpoint: \(error.localizedDescription)", style: .error)
        }
    }

    private func resetMirrorEndpoint() {
        model.hfEndpointInput = "https://huggingface.co"
        mirrorEndpointInput = "https://huggingface.co"
        savedMirrorEndpoint = "https://huggingface.co"
        do {
            try TurboSparkCatalog.setHfEndpoint(nil)
            model.showToast("Mirror endpoint reset to default", style: .info)
        } catch {
            model.showToast("Failed to reset mirror endpoint: \(error.localizedDescription)", style: .error)
        }
    }
}
