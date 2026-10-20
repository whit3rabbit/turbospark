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
        .frame(minWidth: 480, idealWidth: 620 * scale, maxWidth: 840 * scale, minHeight: 460, idealHeight: 640 * scale)
        .background(.appPage)
        .onAppear {
            terms = AudioCustomVocabularyStore.load()
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(verbatim: "Custom Vocabulary & Jargon")
                    .themedFont(.title2, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Text(verbatim: "\(terms.count) terms")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .monospacedDigit()
            }
            Text(verbatim: "Ensure domain-specific terms, acronyms, and company names are recognized accurately. Misrecognized phonetic variants are automatically normalized.")
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
    }

    // MARK: - Add Term Card

    private var addTermCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(verbatim: "Add New Term")
                .themedFont(.base, weight: .semibold)

            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(verbatim: "Canonical Term")
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    TextField(text: $newTerm, prompt: Text(verbatim: "e.g. TurboSpark, kubectl")) {
                        Text(verbatim: "Term")
                    }
                    .textFieldStyle(.roundedBorder)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(verbatim: "Category (Optional)")
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    TextField(text: $newCategory, prompt: Text(verbatim: "e.g. Project, Tech, Name")) {
                        Text(verbatim: "Category")
                    }
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140 * scale)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(verbatim: "Sounds Like / Misrecognitions (comma separated)")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                TextField(text: $newAliases, prompt: Text(verbatim: "e.g. turbo spark, turbospark, cube cuddle")) {
                    Text(verbatim: "Sounds like")
                }
                .textFieldStyle(.roundedBorder)
            }

            HStack {
                Spacer()
                Button {
                    addTerm()
                } label: {
                    Label {
                        Text(verbatim: "Add Term")
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
            HStack {
                Text(verbatim: "Registered Terms")
                    .themedFont(.base, weight: .semibold)
                Spacer()
                searchField
            }

            if filteredTerms.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "character.book.closed")
                        .font(.system(size: 28))
                        .foregroundStyle(.appSecondary)
                    Text(verbatim: searchQuery.isEmpty ? "No vocabulary terms registered yet" : "No terms matching '\(searchQuery)'")
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 28)
                .appCardSurface(padding: 16)
            } else {
                VStack(spacing: 8) {
                    ForEach(filteredTerms) { item in
                        termRow(item)
                    }
                }
            }
        }
    }

    private var searchField: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.appSecondary)
            TextField(text: $searchQuery, prompt: Text(verbatim: "Filter terms...")) {
                Text(verbatim: "Search")
            }
            .textFieldStyle(.plain)
            if !searchQuery.isEmpty {
                Button {
                    searchQuery = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .frame(maxWidth: 200 * scale)
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

            Button {
                deleteTerm(id: item.id)
            } label: {
                Image(systemName: "trash")
                    .foregroundStyle(.appSecondary)
            }
            .buttonStyle(.plain)
            .help(Text(verbatim: "Remove term from vocabulary"))
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
                Text(verbatim: "Done")
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
