import SwiftUI

/// Settings > Browser > Site permissions: the saved per-site records with
/// search and per-record removal, and the recent decisions (the audit log).
/// This pane is the one audit viewer.
@MainActor
struct SitePermissionsSettingsSection: View {
    @ObservedObject var store: SitePermissionStore
    @State private var query = ""

    /// The audit rows shown at once. The store keeps more; this keeps the
    /// grouped form short.
    static let visibleAuditRows = 50

    var body: some View {
        Section(header: Text("Site permissions", bundle: .module)) {
            Text("Pages that ask for the camera, microphone, or location are denied unless you allow them. Tabs the agent controls are always denied. A saved choice applies to one exact site in this profile only.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)

            AppSearchField(text: $query, prompt: Text("Search", bundle: .module))

            let records = SitePermissionStore.filter(store.records, query: query)
            if records.isEmpty {
                Text("No saved site permissions.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            } else {
                ForEach(records) { record in
                    recordRow(record)
                }
            }

            DisclosureGroup {
                if store.auditLog.isEmpty {
                    Text("No decisions yet.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                } else {
                    ForEach(store.auditLog.prefix(Self.visibleAuditRows)) { event in
                        auditRow(event)
                    }
                }
            } label: {
                Text("Recent decisions", bundle: .module)
            }
        }
        .onAppear { store.loadIfNeeded() }
    }

    private func recordRow(_ record: SitePermissionRecord) -> some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: record.origin)
                    .themedCode(.small, weight: .semibold)
                    .textSelection(.enabled)
                SitePermissionText.name(record.permissionClass)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            Spacer(minLength: 8)
            verdictBadge(record.verdict)
            // Removing a record loses nothing the user cannot grant again, so
            // it is a quiet action with a neutral glyph, not a destructive one.
            AppIconButton("minus.circle", label: Text("Remove \(record.origin)", bundle: .module)) {
                store.removeRecord(id: record.id)
            }
            .buttonStyle(.borderless)
        }
        .accessibilityElement(children: .contain)
    }

    private func verdictBadge(_ verdict: SitePermissionVerdict) -> some View {
        // Glyph plus text, never color alone.
        HStack(spacing: 4) {
            switch verdict {
            case .allow:
                Image(systemName: "checkmark.circle").accessibilityHidden(true)
                Text("Allowed", bundle: .module)
            case .deny:
                Image(systemName: "slash.circle").accessibilityHidden(true)
                Text("Denied", bundle: .module)
            }
        }
        .themedFont(.small)
        .foregroundStyle(.appSecondary)
    }

    private func auditRow(_ event: BrowserPermissionDiagnosticEvent) -> some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: event.canonicalOrigin)
                    .themedCode(.small)
                HStack(spacing: 6) {
                    if let permissionClass = event.permissionClass {
                        SitePermissionText.name(permissionClass)
                    }
                    SitePermissionText.decision(event.decision)
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }
            Spacer(minLength: 8)
            Text(event.timestamp, format: .dateTime.month().day().hour().minute())
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
        .accessibilityElement(children: .combine)
    }
}
