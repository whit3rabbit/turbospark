import SwiftUI

/// Shown under a fetched marketplace so the user can remember its source.
/// Fetching never saves anything: the manifest's own name is attacker-
/// controlled, so the name here is only a suggestion the user may change, and
/// nothing is written until they press Add. A name that already points at a
/// different source is refused (see `AppModel.addMarketplaceSource`).
struct FetchedSourceSaveBar: View {
    @ObservedObject var model: AppModel
    let source: MarketplaceSource
    let kind: MarketplaceKind
    let projectID: UUID?
    let suggestedName: String

    @State private var name: String
    @State private var message: String?
    @State private var failed = false

    init(model: AppModel, source: MarketplaceSource, kind: MarketplaceKind,
         projectID: UUID?, suggestedName: String) {
        self.model = model
        self.source = source
        self.kind = kind
        self.projectID = projectID
        self.suggestedName = suggestedName
        self._name = State(initialValue: suggestedName)
    }

    var body: some View {
        HStack(spacing: 8) {
            Text("Save this source as", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            TextField(text: $name) { Text("Name", bundle: .module) }
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 220)
            Button { add() } label: { Text("Add source", bundle: .module) }
                .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            if let message {
                if failed {
                    Text(message).themedFont(.small).foregroundStyle(.orange).lineLimit(2)
                } else {
                    Text(message).themedFont(.small).foregroundStyle(.appSecondary).lineLimit(2)
                }
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
    }

    private func add() {
        do {
            try model.addMarketplaceSource(name: name, source: source, kind: kind, projectID: projectID)
            failed = false
            message = String(localized: "Source saved.", bundle: .module)
        } catch {
            failed = true
            message = error.localizedDescription
        }
    }
}
