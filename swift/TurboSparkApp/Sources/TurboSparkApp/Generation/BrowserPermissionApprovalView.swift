import SwiftUI

struct BrowserPermissionApprovalView: View {
    let request: BrowserPermissionApprovalRequest
    let onAllowOnce: () -> BrowserPermissionApprovalActionResult
    let onAlwaysAllow: () -> BrowserPermissionApprovalActionResult
    let onDeny: () -> Void

    @State private var refusal: BrowserPermissionGrantResult?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "globe")
                    .foregroundStyle(.appAccent)
                Text(ToolPresentation.resolve(request.toolName).localizedLabel)
                    .themedFont(.small, weight: .medium)
                Text(verbatim: request.origin.canonicalString)
                    .themedFont(.tiny, systemDesign: .monospaced)
                    .textSelection(.enabled)
            }
            .lineLimit(1)

            Text("This action requires your confirmation to execute.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            Text("Site grants match exact HTTP(S) origins and are limited to 256 per project. Actions ask by default.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)

            HStack(spacing: 8) {
                Button {
                    handle(onAllowOnce())
                } label: {
                    Text("Approve Once", bundle: .module)
                }
                .buttonStyle(.borderedProminent)

                Button {
                    handle(onAlwaysAllow())
                } label: {
                    Text("Always Allow", bundle: .module)
                }
                .buttonStyle(.bordered)

                Button(role: .cancel, action: onDeny) {
                    Text("Deny", bundle: .module)
                }
                .buttonStyle(.bordered)
            }

            if let refusal {
                if refusal == .limitReached {
                    Text("Project limit reached (256 origins). Remove one before adding another.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.red)
                } else if refusal == .invalidOrigin {
                    Text("Enter a valid HTTP(S) origin.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.red)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
    }

    private func handle(_ result: BrowserPermissionApprovalActionResult) {
        switch result {
        case .approved, .unavailable:
            refusal = nil
        case .grantRefused(let grantResult):
            refusal = grantResult
        }
    }
}
