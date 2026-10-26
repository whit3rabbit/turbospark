import SwiftUI
import TurboSpark

/// Management sheet for custom vocabulary terms and jargon.
///
/// Phase 1:
/// Allows users to register domain-specific words, acronyms, and common
/// speech-to-text misrecognitions that are normalized across transcripts.
@MainActor
struct AudioCustomVocabularySheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.appTheme) private var theme
    @FocusState private var isFieldFocused: Bool

    @State private var terms: [AudioVocabularyTerm] = []
    @State private var newTerm: String = ""
    @State private var newAliases: String = ""
    @State private var newCategory: String = ""
    @State private var searchQuery: String = ""

    private var scale: CGFloat { AppMetrics.textScale(theme) }

    private var filteredTerms: [AudioVocabularyTerm] {
        let query = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return terms }
        return terms.filter { item in
            item.term.lowercased().contains(query) ||
            item.aliases.contains { $0.lowercased().contains(query) } ||
            (item.category?.lowercased().contains(query) == true)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.horizontal, 24)
                .padding(.top, 20)
                .padding(.bottom, 14)
            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    addTermCard
                    termsListSection
                }
                .padding(24)
            }

            Divider()
            footer
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
        }
        .frame(
            minWidth: (480 * scale).rounded(),
            idealWidth: (620 * scale).rounded(),
            maxWidth: (840 * scale).rounded(),
            minHeight: (460 * scale).rounded(),
            idealHeight: (640 * scale).rounded()
        )
        .background(.appPage)
        .onAppear {
            terms = AudioCustomVocabularyStore.load()
            isFieldFocused = true
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Custom Vocabulary & Jargon", bundle: .module)
                    .themedFont(.title2, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Text("\(terms.count) terms", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .monospacedDigit()
            }
            Text("Ensure domain-specific terms, acronyms, and company names are recognized accurately. Misrecognized phonetic variants are automatically normalized.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
    }

    // MARK: - Add Term Card

    private var addTermCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add New Term", bundle: .module)
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)

            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Canonical Term", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    TextField(text: $newTerm, prompt: Text("e.g. TurboSpark, kubectl", bundle: .module)) {
                        Text("Term", bundle: .module)
                    }
                    .textFieldStyle(.roundedBorder)
                    .focused($isFieldFocused)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Category (Optional)", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    TextField(text: $newCategory, prompt: Text("e.g. Project, Tech, Name", bundle: .module)) {
                        Text("Category", bundle: .module)
                    }
                    .textFieldStyle(.roundedBorder)
                    .frame(
                        minWidth: (100 * scale).rounded(),
                        idealWidth: (140 * scale).rounded(),
                        maxWidth: (200 * scale).rounded()
                    )
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Sounds Like / Misrecognitions (comma separated)", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                TextField(text: $newAliases, prompt: Text("e.g. turbo spark, turbospark, cube cuddle", bundle: .module)) {
                    Text("Sounds like", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)
            }

            HStack {
                Spacer()
                Button {
                    addTerm()
                } label: {
                    Label {
                        Text("Add Term", bundle: .module)
                    } icon: {
                        Image(systemName: "plus")
                    }
                }
                .appAction(.primary, size: .small)
                .disabled(newTerm.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .appCardSurface(padding: 16)
    }

    // MARK: - Terms List Section

    private var termsListSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            AppAdaptiveHeader(
                leading: {
                    Text("Registered Terms", bundle: .module)
                        .themedFont(.base, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                },
                actions: {
                    AppSearchField(
                        text: $searchQuery,
                        prompt: Text("Filter terms...", bundle: .module),
                        minWidth: (140 * scale).rounded(),
                        idealWidth: (180 * scale).rounded(),
                        maxWidth: (240 * scale).rounded()
                    )
                }
            )

            if filteredTerms.isEmpty {
                if searchQuery.isEmpty {
                    AppEmptyState(
                        systemImage: "character.book.closed",
                        title: Text("No vocabulary terms registered yet", bundle: .module)
                    )
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 28)
                    .appCardSurface(padding: 16)
                } else {
                    AppEmptyState(
                        systemImage: "magnifyingglass",
                        title: Text("No terms matching \"\(searchQuery)\"", bundle: .module)
                    )
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 28)
                    .appCardSurface(padding: 16)
                }
            } else {
                VStack(spacing: 8) {
                    ForEach(filteredTerms) { item in
                        termRow(item)
                    }
                }
            }
        }
    }

    private func termRow(_ item: AudioVocabularyTerm) -> some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Text(verbatim: item.term)
                        .themedFont(.base, weight: .semibold)
                    if let category = item.category, !category.isEmpty {
                        Text(verbatim: category)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.appSurface)
                            .clipShape(Capsule())
                    }
                }

                if !item.aliases.isEmpty {
                    FlowLayout(spacing: 4, lineSpacing: 4) {
                        ForEach(item.aliases, id: \.self) { alias in
                            Text(verbatim: "→ \(alias)")
                                .themedFont(.small)
                                .foregroundStyle(.appSecondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.appSurface)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                    }
                }
            }

            Spacer()

            AppIconButton("minus.circle", label: Text("Remove term from vocabulary", bundle: .module), role: .destructive) {
                deleteTerm(id: item.id)
            }
        }
        .appCardSurface(padding: 12)
    }

    // MARK: - Footer

    private var footer: some View {
        HStack {
            Spacer()
            Button {
                dismiss()
            } label: {
                Text("Done", bundle: .module)
            }
            .appAction(.primary)
            .keyboardShortcut(.defaultAction)
        }
    }

    // MARK: - Actions

    private func addTerm() {
        let term = newTerm.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !term.isEmpty else { return }

        let rawAliases = newAliases.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        let category = newCategory.trimmingCharacters(in: .whitespacesAndNewlines)

        AudioCustomVocabularyStore.add(
            term: term,
            aliases: rawAliases,
            category: category.isEmpty ? nil : category
        )

        terms = AudioCustomVocabularyStore.load()
        newTerm = ""
        newAliases = ""
        newCategory = ""
    }

    private func deleteTerm(id: String) {
        AudioCustomVocabularyStore.delete(id: id)
        terms = AudioCustomVocabularyStore.load()
    }
}
