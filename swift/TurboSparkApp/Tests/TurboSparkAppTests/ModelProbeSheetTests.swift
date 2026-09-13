import XCTest
import TurboSpark

@testable import TurboSparkApp

final class ModelProbeSheetTests: XCTestCase {

    private func decode<T: Decodable>(_ json: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    func testSanitizeRepoStripsUrlsAndPrefixes() {
        XCTAssertEqual(
            ModelProbeSheet.sanitizeRepo("https://huggingface.co/unsloth/Qwen3.8-27B-GGUF"),
            "unsloth/Qwen3.8-27B-GGUF"
        )
        XCTAssertEqual(
            ModelProbeSheet.sanitizeRepo("http://hf.co/unsloth/Qwen3.8-27B-GGUF/"),
            "unsloth/Qwen3.8-27B-GGUF"
        )
        XCTAssertEqual(
            ModelProbeSheet.sanitizeRepo("www.huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit"),
            "mlx-community/Qwen3.6-35B-A3B-4bit"
        )
        XCTAssertEqual(
            ModelProbeSheet.sanitizeRepo("https://www.huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/"),
            "mlx-community/Qwen3.6-35B-A3B-4bit"
        )
        XCTAssertEqual(
            ModelProbeSheet.sanitizeRepo("unsloth/Qwen3.8-27B-GGUF"),
            "unsloth/Qwen3.8-27B-GGUF"
        )
        XCTAssertEqual(
            ModelProbeSheet.sanitizeRepo("  https://huggingface.co/org/repo@rev/  "),
            "org/repo@rev"
        )
    }

    func testProbeInstallDecisionBlocksOnError() {
        let err = "Probe failed: HTTP 404"
        let decision = ModelProbeGating.installDecision(
            repo: "unsloth/Qwen3.8-27B-GGUF",
            probeError: err,
            report: nil,
            variants: nil,
            ggufFile: "",
            selectedVariant: nil,
            freeDiskBytes: 100_000_000_000
        )
        XCTAssertTrue(decision.isBlocked)
        XCTAssertEqual(decision.reason, err)
    }

    func testProbeInstallDecisionBlocksOnMultiVariantWithoutSelection() throws {
        let json = """
        {
            "repo": "unsloth/Qwen3.8-27B-GGUF",
            "revision": "main",
            "variants": [
                { "file": "a.gguf", "bytes": 1000, "quantLabel": "Q4_K_M", "ladderRank": 4, "executable": true },
                { "file": "b.gguf", "bytes": 2000, "quantLabel": "Q8_0", "ladderRank": 0, "executable": true }
            ],
            "shardedSkipped": 0
        }
        """
        let variants: RepoVariants = try decode(json)
        let decision = ModelProbeGating.installDecision(
            repo: "unsloth/Qwen3.8-27B-GGUF",
            probeError: nil,
            report: nil,
            variants: variants,
            ggufFile: "",
            selectedVariant: nil,
            freeDiskBytes: 100_000_000_000
        )
        XCTAssertTrue(decision.isBlocked)
        XCTAssertTrue(decision.reason?.contains("multiple GGUF variants") ?? false)
    }

    func testProbeInstallDecisionRequiresProbeBeforeInstall() {
        let decision = ModelProbeGating.installDecision(
            repo: "unsloth/Qwen3.8-27B-GGUF",
            probeError: nil,
            report: nil,
            variants: nil,
            ggufFile: "a.gguf",
            selectedVariant: nil,
            freeDiskBytes: 100_000_000_000
        )
        XCTAssertTrue(decision.isBlocked)
        XCTAssertTrue(decision.reason?.contains("Run probe") ?? false)
    }

    func testProbeInstallDecisionBlocksWhenReportNotRunnable() throws {
        let why = "no kernels for block type(s) Q3_K"
        let json = """
        {
            "repo": "unsloth/Qwen3.8-27B-GGUF",
            "revision": "main",
            "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
            "downloadBytes": 15000000000,
            "architecture": "qwen35",
            "family": "qwen35",
            "runnable": false,
            "refusedBecause": "\(why)",
            "types": [],
            "affine": null,
            "expertStride": null,
            "trainedContext": 32768,
            "slotCacheBytes": [],
            "sidecarsPresent": [],
            "sidecarsMissing": [],
            "chatTemplate": null,
            "warnings": [],
            "fit": null
        }
        """
        let report: ProbeReport = try decode(json)
        let decision = ModelProbeGating.installDecision(
            repo: "unsloth/Qwen3.8-27B-GGUF",
            probeError: nil,
            report: report,
            variants: nil,
            ggufFile: "Qwen3.8-27B-UD-Q4_K_M.gguf",
            selectedVariant: nil,
            freeDiskBytes: 100_000_000_000
        )
        XCTAssertTrue(decision.isBlocked)
        XCTAssertEqual(decision.reason, why)
    }

    func testProbeInstallDecisionAllowsWhenReportRunnableAndFits() throws {
        let json = """
        {
            "repo": "unsloth/Qwen3.8-27B-GGUF",
            "revision": "main",
            "file": "Qwen3.8-27B-UD-Q6_K.gguf",
            "downloadBytes": 20000000000,
            "architecture": "qwen35",
            "family": "qwen35",
            "runnable": true,
            "refusedBecause": null,
            "types": [],
            "affine": null,
            "expertStride": null,
            "trainedContext": 32768,
            "slotCacheBytes": [],
            "sidecarsPresent": ["tokenizer.json"],
            "sidecarsMissing": [],
            "chatTemplate": "chat_template.jinja",
            "warnings": [],
            "fit": {
                "verdict": "resident",
                "verdictSummary": "fits in unified memory",
                "runs": true,
                "countedBytes": 15000000000,
                "countedSource": "estimated",
                "mappedBytes": 20000000000,
                "mappedSource": "download",
                "slotCacheSlots": 0,
                "slotCacheBytes": 0,
                "kvBytes": 1000000000,
                "residentBytes": 15000000000,
                "largestContext": 32768,
                "context": 4096,
                "contextLadder": [],
                "trainedContext": 32768
            }
        }
        """
        let report: ProbeReport = try decode(json)
        let decision = ModelProbeGating.installDecision(
            repo: "unsloth/Qwen3.8-27B-GGUF",
            probeError: nil,
            report: report,
            variants: nil,
            ggufFile: "Qwen3.8-27B-UD-Q6_K.gguf",
            selectedVariant: nil,
            freeDiskBytes: 100_000_000_000
        )
        XCTAssertEqual(decision, ModelInstallDecision.allowed)
    }
}
