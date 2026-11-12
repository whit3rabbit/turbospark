import XCTest
import TurboSpark
@testable import TurboSparkApp

final class AudioCustomVocabularyTests: XCTestCase {

    func testAudioCustomVocabularyNormalization() {
        let terms = [
            AudioVocabularyTerm(term: "TurboSpark", aliases: ["turbo spark", "turbospark"], category: "Project"),
            AudioVocabularyTerm(term: "CoreAudio", aliases: ["core audio"], category: "Tech"),
            AudioVocabularyTerm(term: "kubectl", aliases: ["cube control", "cube cuddle", "kube control"], category: "CLI"),
            AudioVocabularyTerm(term: "PostgreSQL", aliases: ["postgres ql", "postgres"], category: "Database"),
        ]

        let input = "Today we deployed turbo spark using kube control and connected to postgres ql with core audio enabled."
        let normalized = AudioCustomVocabulary.apply(terms, to: input)

        XCTAssertEqual(normalized, "Today we deployed TurboSpark using kubectl and connected to PostgreSQL with CoreAudio enabled.")
    }

    func testWordBoundarySafety() {
        let terms = [
            AudioVocabularyTerm(term: "TurboSpark", aliases: ["turbo"], category: "Project"),
            AudioVocabularyTerm(term: "kubectl", aliases: ["cube"], category: "CLI"),
        ]

        // Partial word matches like "turbofan", "cubed" must NOT be modified
        let input = "The turbofan engine was tested in a cubed room."
        let normalized = AudioCustomVocabulary.apply(terms, to: input)

        XCTAssertEqual(normalized, "The turbofan engine was tested in a cubed room.")
    }

    func testCustomVocabularySegmentsNormalization() {
        let terms = [
            AudioVocabularyTerm(term: "TurboSpark", aliases: ["turbo spark"], category: "Project"),
            AudioVocabularyTerm(term: "MLX", aliases: ["m l x"], category: "Tech"),
        ]

        let seg1 = AudioLibrarySegment(id: UUID(), start: 0, end: 4, text: "Welcome to turbo spark.")
        let seg2 = AudioLibrarySegment(id: UUID(), start: 4, end: 8, text: "We run on m l x resident kernels.")

        let updated = AudioCustomVocabulary.apply(terms, to: [seg1, seg2])

        XCTAssertEqual(updated[0].text, "Welcome to TurboSpark.")
        XCTAssertEqual(updated[1].text, "We run on MLX resident kernels.")
    }

    func testPromptContextGeneration() {
        let terms = [
            AudioVocabularyTerm(term: "TurboSpark"),
            AudioVocabularyTerm(term: "CoreAudio"),
            AudioVocabularyTerm(term: "GGUF"),
        ]

        let context = AudioCustomVocabulary.promptContext(terms: terms)
        XCTAssertEqual(context, "TurboSpark, CoreAudio, GGUF")
    }

    func testCustomVocabularyStoreRoundTrip() {
        let testTerm = AudioVocabularyTerm(
            term: "SyntestingVocab",
            aliases: ["syn testing", "syntest"],
            category: "Test"
        )

        AudioCustomVocabularyStore.save([testTerm])
        let loaded = AudioCustomVocabularyStore.load()

        XCTAssertTrue(loaded.contains { $0.term == "SyntestingVocab" })

        AudioCustomVocabularyStore.delete(id: testTerm.id)
        let afterDelete = AudioCustomVocabularyStore.load()
        XCTAssertFalse(afterDelete.contains { $0.id == testTerm.id })
    }
}
