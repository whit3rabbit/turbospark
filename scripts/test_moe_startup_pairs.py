import unittest

from moe_startup_pairs import compare_tokens, validate_rows


def record():
    return {
        "protocol": "moe-startup-v1", "discarded_warmups": 0,
        "first_token_ms": 10, "load_to_first_token_ms": 20, "new_tokens": 1,
        "request_index": 0, "precompiled_requested": True,
        "pipeline_cache_requested": True,
        "compiler_after_request": {"library_loads": 2, "library_compiles": 0, "cache_enabled": True, "archive_hits": 3, "archive_misses": 0, "archive_errors": 0},
        "manifest_sha256": "model", "case": "short", "context": 4096,
        "max_new": 64, "expert_cache_slots": 16, "residency": "streamed",
        "shaping": "greedy", "seed": 42, "token_sha256": "tokens", "stop": "maxTokens",
    }


class EvidenceGuards(unittest.TestCase):
    def test_requested_optimization_must_actually_engage(self):
        for field, value in [("library_loads", 0), ("library_compiles", 1), ("cache_enabled", False), ("archive_hits", 0), ("archive_misses", 1), ("archive_errors", 1)]:
            row = record()
            row["compiler_after_request"][field] = value
            with self.assertRaises(ValueError):
                validate_rows([row], "ir-archive")
        validate_rows([record()], "ir-archive")

    def test_different_outputs_or_contracts_cannot_be_paired(self):
        for field, value in [("token_sha256", "other"), ("manifest_sha256", "other"), ("seed", 7), ("stop", "endOfTurn")]:
            row = {**record(), field: value}
            with self.assertRaises(ValueError):
                compare_tokens([record(), row])
        compare_tokens([record(), record()])

    def test_discarded_warmup_cannot_masquerade_as_startup(self):
        with self.assertRaises(ValueError):
            validate_rows([{**record(), "discarded_warmups": 1}], "ir-archive")
        with self.assertRaises(ValueError):
            validate_rows([{**record(), "first_token_ms": None}], "ir-archive")


if __name__ == "__main__":
    unittest.main()
