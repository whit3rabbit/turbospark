"""Tokenizer framing and truncation parity checks for IG1 conditioning fixtures."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "docs/verification/z-image-ig0-contracts.json"
CAPTURE_ROOT = ROOT / "docs/verification/z-image-ig0-captures"
MODEL_TOKENIZER = ROOT / "target" / "ig0" / "model" / "tokenizer"

PROMPTS = {
    "empty": "",
    "composition": "A red ceramic teapot to the left of a blue cup on a wooden table, a window behind them.",
    "typography": 'A shop sign with the exact words "FRESH BREAD" in clear black letters.',
    "detail": "Macro photograph of a honeybee on lavender, fine hairs and translucent wing veins.",
    "lighting": "A lighthouse in winter at dusk, warm windows reflected on wet snow, cold blue shadows.",
    "unicode": "\u96ea\u4e2d\u306e\u706f\u53f0, caf\u00e9 au cr\u00e9puscule",
    "overlong": "small red lighthouse " * 600,
}


class TokenizerParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACTS.read_text())
        cls.tokenizer = None
        if MODEL_TOKENIZER.exists():
            try:
                from transformers import AutoTokenizer
                cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER, local_files_only=True)
            except Exception:
                cls.tokenizer = None
        cls.token_cases = cls.contract["token_cases"]
        cls.contract_arrays = cls.contract["arrays"]

    def _case_from_name(self, name):
        return CAPTURE_ROOT / name / "encode.json"

    def test_contract_token_case_counts(self):
        self.assertEqual(set(self.contract["token_cases"]), set(PROMPTS))
        for case in PROMPTS:
            observed = self.token_cases[case]
            self.assertIn("untruncated_tokens", observed)
            self.assertIn("retained_tokens", observed)
            self.assertGreaterEqual(observed["untruncated_tokens"], observed["retained_tokens"])
            self.assertLessEqual(observed["retained_tokens"], 512)
        self.assertEqual(self.token_cases["empty"]["retained_tokens"], 8)
        self.assertEqual(self.token_cases["unicode"]["retained_tokens"], 19)
        self.assertEqual(self.token_cases["overlong"]["untruncated_tokens"], 2409)
        self.assertEqual(self.token_cases["overlong"]["retained_tokens"], 512)
        self.assertEqual(self.token_cases["overlong"]["untruncated_tokens"],
                         self.token_cases["overlong"]["retained_tokens"] + 1897)

    def test_capture_contract_schemas(self):
        expected_arrays = {"token_ids", "attention_mask", "conditioning"}
        for case in PROMPTS:
            path = self._case_from_name(case)
            self.assertTrue(path.exists(), path)
            manifest = json.loads(path.read_text())
            arrays = manifest["arrays"]
            self.assertSetEqual(set(arrays), expected_arrays)
            self.assertEqual(manifest["stage"], "encode")
            self.assertEqual(manifest["settings"]["prompt"], PROMPTS[case])
            for name in expected_arrays:
                self.assertIn(name, arrays)
            self.assertEqual(arrays["token_ids"]["shape"], [1, 512])
            self.assertEqual(arrays["token_ids"]["dtype"], "int64")
            self.assertEqual(arrays["attention_mask"]["shape"], [1, 512])
            self.assertEqual(arrays["attention_mask"]["dtype"], "int64")
            self.assertEqual(arrays["conditioning"]["source_dtype"], "torch.bfloat16")
            contract_ids = self.contract_arrays[f"{case}_ids"]
            self.assertEqual(arrays["token_ids"]["sha256"], contract_ids["sha256"])
            contract_mask = self.contract_arrays[f"{case}_mask"]
            self.assertEqual(arrays["attention_mask"]["sha256"], contract_mask["sha256"])

    def test_conditioning_rows_follow_retained_tokens(self):
        for case in PROMPTS:
            path = self._case_from_name(case)
            manifest = json.loads(path.read_text())
            arrays = manifest["arrays"]
            conditioning_shape = arrays["conditioning"]["shape"]
            expected_rows = self.token_cases[case]["retained_tokens"]
            self.assertEqual(conditioning_shape[1], 2560)
            self.assertEqual(conditioning_shape[0], expected_rows)

    def test_framing_and_truncation_parity(self):
        if self.tokenizer is None:
            self.skipTest("local tokenizer checkout is unavailable")
        for case in PROMPTS:
            prompt = PROMPTS[case]
            expected = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            path = self._case_from_name(case)
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["framed_prompt"], expected)
            tokens = self.tokenizer(prompt=expected, padding="max_length", max_length=512,
                                    truncation=True, return_tensors="np")
            self.assertEqual(tokens["input_ids"].shape, (1, 512))
            self.assertEqual(tokens["attention_mask"].shape, (1, 512))
            self.assertEqual(int(tokens["attention_mask"].sum()), self.token_cases[case]["retained_tokens"])


if __name__ == "__main__":
    unittest.main()
