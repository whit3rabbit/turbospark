import unittest

from prefill_bank_replay import lfu_misses, order_misses, order_replay, replay
from test_expert_policy_replay import capture


class PrefillBankReplay(unittest.TestCase):
    def test_pipelining_protects_previous_experts_until_the_batch_boundary(self):
        groups = [[0], [0], [1], [2], [0]]
        self.assertEqual(lfu_misses(groups, 16, 2, 3, 1), 4)
        self.assertEqual(lfu_misses(groups, 3, 2, 3, 1), 3)

    def test_shared_bank_reloads_at_each_layer_and_microbatch(self):
        # The extra pass is decode and selects a new expert on both layers.
        trace = capture([[[0]] * 129 + [[1]]] * 2, 2, 1)
        layout = {
            "expertsPerLayer": 2,
            "numLayers": 2,
            "expertStride": 30,
            "layers": [{"layer": 0, "expertStride": 10}, {"layer": 1}],
        }
        result = replay(trace, layout, 1, 129)
        self.assertEqual(
            result["requested_bytes"], {"lfu_16": 40, "lfu_128": 40, "bank_128": 80}
        )
        self.assertFalse(result["passes_screen"])

    def test_long_prompt_clears_screen_when_lfu_repeatedly_evicts(self):
        trace = capture([[[i % 4] for i in range(6000)]], 4, 1)
        layout = {
            "expertsPerLayer": 4,
            "numLayers": 1,
            "expertStride": 10,
            "layers": [{"layer": 0}],
        }
        result = replay(trace, layout, 2, 6000)
        self.assertEqual(
            result["requested_bytes"], {"lfu_16": 60000, "lfu_128": 60000, "bank_128": 1880}
        )
        self.assertTrue(result["passes_screen"])

    def test_layer_major_beats_a_bank_emptied_between_chunks(self):
        # Four experts cycling, chunks of 4, four slots: the shared bank
        # rereads all four every chunk, layer-major reads them once.
        groups = [[i % 4] for i in range(16)]
        got = order_misses(groups, 4, 4, 4)
        self.assertEqual(got, {"batch_major_lfu": 4, "batch_major_bank": 16,
                               "layer_major": 4, "compulsory": 4})

    def test_slots_bound_what_either_order_can_keep(self):
        # Two slots: each chunk of 4 splits into two routed sub-batches, and
        # four experts cannot be held by any order.
        groups = [[i % 4] for i in range(8)]
        got = order_misses(groups, 4, 2, 4)
        self.assertEqual(got["layer_major"], got["batch_major_lfu"])
        self.assertEqual(got["layer_major"], 8)
        self.assertEqual(got["compulsory"], 4)

    def test_the_sweep_prices_bytes_and_skips_long_prompts(self):
        trace = capture([[[i % 4] for i in range(16)]] * 2, 4, 1)
        layout = {"expertsPerLayer": 4, "numLayers": 2, "expertStride": 30,
                  "layers": [{"layer": 0, "expertStride": 10}, {"layer": 1}]}
        out = order_replay(trace, layout, [4], [4], [16, 99])
        self.assertEqual(out["skipped_prompts"], [99])
        cell = out["cells"][0]
        self.assertEqual(cell["bytes"]["batch_major_bank"], 16 * (10 + 30))
        self.assertEqual(cell["bytes"]["layer_major"], 4 * (10 + 30))
        self.assertEqual(cell["saving_vs_bank_percent"], 75.0)
        self.assertEqual(cell["saving_vs_lfu_percent"], 0.0)
        with self.assertRaises(ValueError):
            order_replay(trace, layout, [4], [4], [99])


if __name__ == "__main__":
    unittest.main()
