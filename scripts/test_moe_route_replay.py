import copy
import unittest
from moe_route_replay import analyze, routed_sub_batches, union_growth, verify_summary


def capture(n=17):
    routes = [[0, 1] for _ in range(n)]
    return {"num_experts": 4, "num_layers": 1, "top_k": 2,
            "trace": [[e for r in routes for e in r]],
            "batches": {"version": 1, "truncated": False, "rows": [{
                "request": 1, "batch": 1, "phase": "prefill", "layer": 0,
                "token_start": 0, "token_count": n, "trace_start": 0,
                "trace_end": n * 2, "routes": routes, "completed": True,
                "cache_hits": 2*n-2, "cache_misses": 2, "bytes_requested": 64,
                "io_samples": n}]}}


class ReplayTests(unittest.TestCase):
    def test_threshold_boundary_and_conservation(self):
        for n, expected in [(16, 0), (17, 2)]:
            row = analyze(capture(n))["batches"][0]
            self.assertEqual(row["selections"], 2 * n)
            self.assertEqual(row["candidates"]["17"]["experts"], expected)
            self.assertEqual(row["measured_io"]["bytes_requested"], 64)
            self.assertTrue(row["executable"])

    def test_windows_are_hypothetical_and_do_not_invent_io(self):
        rows = analyze(capture(), window=8)["batches"]
        self.assertEqual([r["tokens"] for r in rows], [8, 8, 1])
        self.assertTrue(all(not r["executable"] and r["measured_io"] is None for r in rows))

    def test_legacy_and_truncated_refused(self):
        with self.assertRaises(ValueError): analyze({"num_layers": 1, "num_experts": 4, "top_k": 2})
        c = capture(); c["batches"]["truncated"] = True
        with self.assertRaises(ValueError): analyze(c)
        self.assertEqual(len(analyze(c, allow_truncated=True)["batches"]), 1)

    def test_malformed_captures(self):
        changes = {"layer": 3, "trace_end": 9, "token_count": 2,
                   "cache_hits": 1, "io_samples": 2, "routes": [[0, 0]]}
        for key, value in changes.items():
            with self.subTest(key=key):
                c = capture(); c["batches"]["rows"][0][key] = value
                with self.assertRaises(ValueError): analyze(c)

    def test_decode_and_failed_batches_excluded(self):
        for key, value in [("phase", "decode"), ("completed", False)]:
            c = capture(); c["batches"]["rows"][0][key] = value
            self.assertEqual(analyze(c)["batches"], [])

    def test_windows_do_not_cross_requests_or_gaps(self):
        c = capture(2); del c["trace"]
        row = c["batches"]["rows"][0]
        second = dict(copy.deepcopy(row), batch=2, token_start=9, trace_start=4, trace_end=8)
        third = dict(copy.deepcopy(row), request=2, batch=3, trace_start=8, trace_end=12)
        c["batches"]["rows"] += [second, third]
        self.assertEqual([r["tokens"] for r in analyze(c, window=64)["batches"]], [2, 2, 2])

    def test_trace_mismatch_and_duplicate_batches_refused(self):
        c = capture(); c["trace"][0][0] = 3
        with self.assertRaises(ValueError): analyze(c)
        c = capture(); c["batches"]["rows"] *= 2
        with self.assertRaises(ValueError): analyze(c)

    def test_modeled_bytes_are_explicit(self):
        row = analyze(capture(), expert_bytes=128)["batches"][0]
        self.assertEqual(row["candidates"]["17"]["modeled_weight_bytes"], 256)

    def test_union_cap_matches_greedy_shrink(self):
        routes = [[0, 1], [1, 2], [3, 4], [0, 1]]
        # cap 3: {0,1} + {2} = 3 fits; {3,4} would make 5, so it starts a new
        # sub-batch; {3,4} + {0,1} = 4 > 3, so the last token stands alone.
        self.assertEqual(routed_sub_batches(routes, 3), [2, 1, 1])
        self.assertEqual(routed_sub_batches(routes, 5), [4])
        self.assertEqual(routed_sub_batches(routes, 4), [2, 2])
        with self.assertRaises(ValueError): routed_sub_batches(routes, 1)

    def test_union_growth_counts_the_shared_step_per_layer(self):
        def plain(row):
            return {"num_experts": 8, "num_layers": 1, "top_k": 2, "trace": [row]}
        a, b = plain([0, 1, 0, 1]), plain([1, 2, 3, 4])
        out = union_growth([a, b], (1, 2))["growth"]
        self.assertEqual([r["mean_union"] for r in out], [2.0, 3.5])
        self.assertEqual(out[1]["source"], "captures")
        self.assertAlmostEqual(out[1]["ceiling"], 2 * 2 / 3.5)
        # One capture is cut into N segments: [0,1] beside [2,3] for N=2.
        seg = union_growth([plain([0, 1, 2, 3])], (2,))["growth"][0]
        self.assertEqual((seg["source"], seg["steps"], seg["mean_union"]),
                         ("segments", 1, 4.0))
        # The prompt is dropped before anything is counted.
        self.assertEqual(union_growth([b], (1,), skip=1)["growth"][0]["steps"], 1)
        with self.assertRaises(ValueError): union_growth([a, b], (4,))
        with self.assertRaises(ValueError): union_growth([a], (8,))

    def test_verify_rows_are_summarized_per_verify(self):
        c = capture(2); del c["trace"]
        first = c["batches"]["rows"][0]
        first.update(phase="verify", routes=[[0, 1], [1, 2]], io_samples=0)
        # The same verify on a second layer, then a second verify.
        second = dict(copy.deepcopy(first), layer=1, routes=[[0, 1], [0, 1]])
        third = dict(copy.deepcopy(first), batch=2, trace_start=4, trace_end=8,
                     routes=[[0, 1], [2, 3]])
        c["num_layers"] = 2
        c["batches"]["rows"] = [first, second, third]
        self.assertEqual(analyze(c)["batches"], [])
        got = verify_summary(analyze(c, phase="verify")["batches"], 100, 3)
        self.assertEqual(got["verifies"], 2)
        self.assertEqual(got["mean_distinct_experts"], (3 + 2 + 4) / 2)
        self.assertEqual(got["requested_bytes_per_committed_token"], 900 / 3)

    def test_union_cap_reported_per_row(self):
        row = analyze(capture(), union_cap=2)["batches"][0]
        self.assertEqual(row["routed_sub_batches"], [17])
        self.assertIsNone(analyze(capture())["batches"][0]["routed_sub_batches"])
