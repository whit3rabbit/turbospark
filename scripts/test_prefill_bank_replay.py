import unittest

from prefill_bank_replay import lfu_misses, replay
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


if __name__ == "__main__":
    unittest.main()
