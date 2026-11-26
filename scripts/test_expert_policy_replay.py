import itertools
import json
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest

from expert_policy_replay import (
    AdmitLfu,
    ClassLru,
    ClassLruPerLayer,
    ClassLruPool,
    DecayedLfu,
    EngineLfu,
    Lru,
    Oracle,
    compulsory,
    felt_verdict,
    felt_wait,
    from_capture,
    keep_replay,
    next_use_index,
    parse_admit,
    parse_keep,
    replay,
    schedule,
    tally,
)


def capture(layers, num_experts, top_k):
    """An in-memory TURBOSPARK_ROUTER_HIST capture from per-layer groups."""
    counts, trace = [], []
    for groups in layers:
        row = [e for g in groups for e in g]
        trace.append(row)
        counts.append([row.count(e) for e in range(num_experts)])
    return {"num_layers": len(layers), "num_experts": num_experts,
            "top_k": top_k, "counts": counts, "trace": trace}


def misses_by_pass(steps, factory, pool=None):
    return [m for _, _, _, m in replay(steps, factory, pool=pool)]


def oracle_factory(steps, slots):
    per_layer = {}
    for (layer, expert), times in next_use_index(steps).items():
        per_layer.setdefault(layer, {})[expert] = times
    return lambda layer: Oracle(slots, per_layer[layer])


def brute_force_min(groups, slots):
    """The true minimum miss count, by trying every legal eviction."""
    best = [float("inf")]

    def walk(t, resident, missed):
        if missed >= best[0]:
            return
        if t == len(groups):
            best[0] = missed
            return
        group = set(groups[t])
        new = group - resident
        need = len(resident) + len(new) - slots
        if need <= 0:
            walk(t + 1, resident | new, missed + len(new))
            return
        for out in itertools.combinations(sorted(resident - group), need):
            walk(t + 1, (resident - set(out)) | new, missed + len(new))

    walk(0, frozenset(), 0)
    return best[0]


class OracleIsTheOptimum(unittest.TestCase):
    def one_layer(self, groups, num_experts, top_k):
        cap = from_capture(capture([groups], num_experts, top_k), "synthetic")
        return schedule([cap])

    def test_the_textbook_reference_string(self):
        # 1 2 3 4 1 2 5 1 2 3 4 5 over three frames: OPT faults 7 times,
        # LRU 10. Any paging text has this example; it pins both arms to
        # an answer known before this file existed.
        refs = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
        steps = self.one_layer([[r] for r in refs], 6, 1)
        self.assertEqual(sum(misses_by_pass(steps, oracle_factory(steps, 3))), 7)
        self.assertEqual(sum(misses_by_pass(steps, lambda layer: Lru(3))), 10)

    def test_it_matches_brute_force_on_grouped_requests(self):
        # The engine requests top_k experts AT ONCE and all must be
        # resident together, which is not the textbook's one-page request.
        # Belady's rule restricted to non-requested slots is checked here
        # against exhaustive search rather than argued.
        rng = random.Random(1)
        for _ in range(300):
            slots, top_k, num_experts = rng.choice([(3, 2, 6), (4, 2, 7), (4, 3, 7)])
            groups = [rng.sample(range(num_experts), top_k) for _ in range(rng.randint(4, 9))]
            steps = self.one_layer(groups, num_experts, top_k)
            got = sum(misses_by_pass(steps, oracle_factory(steps, slots)))
            self.assertEqual(got, brute_force_min(groups, slots), groups)


class StaleCountsAfterADomainChange(unittest.TestCase):
    """The engine's LFU never ages a count, so a long-lived runner carries
    one workload's hot set into the next. Phase A routes {0,1} and {2,3}
    alternately; phase B routes {4,5} and {6,7}. Four slots hold all of
    phase B, and an aging policy keeps them; the engine's policy keeps the
    two stale phase-A experts whose counts B cannot outvote yet, and
    evicts B's own experts every pass."""

    def setUp(self):
        a = [[0, 1], [2, 3]] * 25
        b = [[4, 5], [6, 7]] * 25
        cap_a = from_capture(capture([a], 8, 2), "a")
        cap_b = from_capture(capture([b], 8, 2), "b")
        self.steps = schedule([cap_a, cap_b])
        self.skips = [0, 0]

    def phase_b_misses(self, factory):
        """Misses per phase-B pass, in pass order."""
        return [m for c, _, _, m in replay(self.steps, factory) if c == 1]

    def test_the_engine_policy_thrashes_and_aging_policies_do_not(self):
        engine = self.phase_b_misses(lambda layer: EngineLfu(4, 8))
        lru = self.phase_b_misses(lambda layer: Lru(4))
        decayed = self.phase_b_misses(lambda layer: DecayedLfu(4, 4))
        floor = compulsory(self.steps, self.skips)[1]
        self.assertEqual(floor, 4)
        self.assertEqual(sum(lru), floor)
        # The decayed counts need a few passes to fall below B's, and then
        # B's four experts stay resident for good.
        self.assertEqual(sum(decayed[10:]), 0, decayed)
        # The engine's counts need ~50 passes for B to outvote A: it misses
        # both requests of nearly every phase-B pass, including late ones.
        self.assertGreater(sum(engine), 90)
        self.assertEqual(sum(engine[10:40]), 60, engine)
        self.assertEqual(tally(replay(self.steps, lambda layer: Lru(4)), self.skips)[1],
                         (100, floor))


class ThePoolLendsSlotsAcrossLayers(unittest.TestCase):
    def test_a_concentrated_layer_frees_slots_for_a_spread_one(self):
        # Layer 0 cycles four experts through three slots, which LRU misses
        # every time; layer 1 only ever routes expert 0. Pooled, the six
        # slots hold all five keys and only the first touches miss.
        layer0 = [[e] for e in [0, 1, 2, 3] * 10]
        layer1 = [[0]] * 40
        cap = from_capture(capture([layer0, layer1], 4, 1), "two-layer")
        steps = schedule([cap])
        per_layer = sum(misses_by_pass(steps, lambda layer: Lru(3)))
        pooled = sum(misses_by_pass(steps, None, pool=Lru(6)))
        self.assertEqual(pooled, compulsory(steps, [0])[0])
        self.assertEqual(pooled, 5)
        self.assertEqual(per_layer, 40 + 1)


class AdmissionKeepsTheHotSet(unittest.TestCase):
    def reads(self, margin, bounce, slots=3):
        # Experts 0 and 1 are hot; 2, 3, 4, ... each appear once.
        groups = []
        for scan in range(2, 22):
            groups += [[0], [1], [scan]]
        steps = schedule([from_capture(capture([groups], 32, 1), "scan")])
        return sum(misses_by_pass(steps, lambda layer: AdmitLfu(slots, margin, bounce)))

    def test_a_one_off_expert_is_bounced_instead_of_evicting_a_hot_one(self):
        # Two residents plus one bounce slot: the scan is read and dropped,
        # so only 0, 1 and the 20 one-offs miss.
        self.assertEqual(self.reads(1, 1), 22)
        # With a margin no count difference can fail, each scan evicts a hot
        # expert, which then misses again.
        self.assertGreater(self.reads(-1000, 1), 40)

    def test_residents_exclude_the_bounce_ring(self):
        cache = AdmitLfu(4, 1, 2)
        for t, e in enumerate([0, 1, 2, 3, 4]):
            cache.plan([e], t)
        self.assertEqual(len(cache.resident), 2)

    def test_more_failures_than_the_ring_holds_are_admitted(self):
        cache = AdmitLfu(3, 100, 1)
        cache.plan([0, 1], 0)
        self.assertEqual(cache.plan([2, 3], 1), 2)
        # One of the two had to take a resident slot for the dispatch.
        self.assertEqual(len(cache.resident & {2, 3}), 1)

    def test_the_arm_syntax(self):
        self.assertEqual(parse_admit("admit:0.5"), (0.5, 2))
        self.assertEqual(parse_admit("admit:2,bounce=1"), (2.0, 1))
        for bad in ("lru", "admit:1,ring=2", "admit:"):
            with self.assertRaises(ValueError):
                parse_admit(bad)


def keep_capture(groups, weights, pred=None, layers=1):
    cap = capture([groups] * layers, 8, len(groups[0]))
    cap["gate_weights"] = [[w for g in weights for w in g]] * layers
    if pred is not None:
        cap["pred"] = [[]] + [[e for g in pred for e in g]] * (layers - 1)
    return from_capture(cap, "keep")


class KeepDropsOnlyColdLowRanks(unittest.TestCase):
    def run_keep(self, cap, j, r=None, skip=0, slots=4):
        return keep_replay([cap], schedule([cap]), slots, 8, j, r, [skip])

    def test_a_cold_low_rank_is_dropped_and_a_resident_one_is_used(self):
        groups = [[0, 1, 2], [3, 4, 5], [3, 4, 0]]
        weights = [[0.5, 0.3, 0.2]] * 3
        # Pass 0 warms the cache in full (skip 1). Pass 1: rank 2 (expert 5)
        # is cold and dropped. Pass 2: rank 2 (expert 0) is resident.
        out = self.run_keep(keep_capture(groups, weights), 2, skip=1, slots=6)
        self.assertEqual((out["reads"], out["full_reads"]), (2, 3))
        self.assertAlmostEqual(out["dropped_mass"], 0.2 / 2)
        self.assertAlmostEqual(out["tokens_with_drop"], 0.5)

    def test_keeping_every_rank_is_the_engine_lfu(self):
        rng = random.Random(3)
        groups = [rng.sample(range(8), 3) for _ in range(40)]
        out = self.run_keep(keep_capture(groups, [[0.5, 0.3, 0.2]] * 40), 3)
        self.assertEqual(out["reads"], out["full_reads"])
        self.assertEqual(out["dropped_mass"], 0.0)

    def test_without_weights_the_mass_is_unknown_not_zero(self):
        cap = from_capture(capture([[[0, 1]]], 8, 2), "v1")
        self.assertIsNone(self.run_keep(cap, 1)["dropped_mass"])

    def test_r_narrows_later_layers_unless_announced(self):
        groups = [[0, 1]]
        weights = [[0.75, 0.25]]
        # J=2 R=2: layer 0 keeps both; layer 1's rank 2 is cold.
        plain = self.run_keep(keep_capture(groups, weights, layers=2), 2, 2)
        self.assertEqual(plain["reads"], 3)
        # Announced by the previous layer's lookahead, it is read.
        told = self.run_keep(
            keep_capture(groups, weights, pred=[[1, 7]], layers=2), 2, 2)
        self.assertEqual(told["reads"], 4)

    def test_the_keep_syntax(self):
        self.assertEqual(parse_keep("5", 8), (5, None))
        self.assertEqual(parse_keep("5,4", 8), (5, 4))
        for bad in ("9", "0", "5,6", "5,4,3"):
            with self.assertRaises(ValueError):
                parse_keep(bad, 8)

    def test_mismatched_weights_are_refused(self):
        cap = capture([[[0, 1]]], 8, 2)
        cap["gate_weights"] = [[0.5]]
        with self.assertRaises(SystemExit):
            from_capture(cap, "short")


class FeltWaitWeightsLayers(unittest.TestCase):
    def test_per_layer_costs_felt_fraction_and_skip(self):
        steps = schedule([from_capture(capture([[[0], [1]], [[0], [0]]], 4, 1), "f")])
        rows = list(replay(steps, lambda layer: Lru(1)))
        # Counted from pass 0: layer 0 misses twice, layer 1 once.
        got = felt_wait(steps, rows, [0], {0: 10.0, 1: 2.0}, 0.5)
        self.assertAlmostEqual(got[0], (2 * 10 + 1 * 2) * 0.5 / 2)
        later = felt_wait(steps, rows, [1], {0: 10.0, 1: 2.0}, 0.5)
        self.assertAlmostEqual(later[0], 5.0)

    def test_the_gate_and_the_predicted_gain(self):
        v = felt_verdict({0: 10.0, 1: 10.0}, {0: 7.0, 1: 10.2}, token_ms=100.0)
        self.assertAlmostEqual(v["mean"], 0.14)
        self.assertAlmostEqual(v["worst"], -0.02)
        self.assertTrue(v["passes"])
        self.assertAlmostEqual(v["predicted_gain"], (100 / 97 - 1 + 100 / 100.2 - 1) / 2)
        self.assertFalse(felt_verdict({0: 10.0, 1: 10.0}, {0: 5.0, 1: 10.5})["passes"])
        self.assertFalse(felt_verdict({0: 10.0}, {0: 9.5})["passes"])
        with self.assertRaises(ValueError):
            felt_verdict({0: 10.0}, {0: 1.0}, token_ms=5.0)


class TheCommandRuns(unittest.TestCase):
    def test_every_new_flag_end_to_end(self):
        rng = random.Random(5)
        groups = [rng.sample(range(16), 4) for _ in range(60)]
        cap = capture([groups, groups], 16, 4)
        cap["gate_weights"] = [[0.25] * 240] * 2
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "c.json"
            path.write_text(json.dumps(cap))
            out = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).with_name(
                    "expert_policy_replay.py")), str(path), "--slots", "8",
                 "--skip", "10", "--arm", "admit:1,bounce=2", "--keep", "3,2",
                 "--felt-fraction", "0.5", "--layer-cost", "1,2",
                 "--token-ms", "100"],
                capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        for needle in ("admit:1,bounce=2", "keep 3,2", "dropped gate mass",
                       "predicted_gain", "not a performance result"):
            self.assertIn(needle, out.stdout)


class DecayRebaseKeepsTheOrder(unittest.TestCase):
    def test_rebasing_changes_no_decision(self):
        # Half-life 1 keeps every weight an exact power of two, so a rebase
        # by 2^-8 is exact and must reproduce the un-rebased run miss for
        # miss. 300 passes stays far below the default threshold.
        rng = random.Random(7)
        groups = [rng.sample(range(10), 2) for _ in range(300)]
        steps = schedule([from_capture(capture([groups], 10, 2), "r")])

        class Eager(DecayedLfu):
            REBASE_EXPONENT = 8

        plain = misses_by_pass(steps, lambda layer: DecayedLfu(4, 1))
        eager_cache = {}

        def eager(layer):
            eager_cache[layer] = Eager(4, 1)
            return eager_cache[layer]

        rebased = misses_by_pass(steps, eager)
        self.assertGreater(eager_cache[0].base, 0, "the rebase never ran")
        self.assertEqual(plain, rebased)


class ClassLruMatchesUpstream(unittest.TestCase):
    """The ClassLRU arm is a port of upstream llama.cpp's own policy, so it
    is pinned to upstream's numbers rather than to this repo's opinion of
    them.

    The golden counts below were produced by driving UPSTREAM's simulator
    (`llama.cpp` `scripts/moe-cache-tune.py`, `class ClassLRU`, the Python
    model of its own C++ `moe_cache_lru`) over the same deterministic
    routing sequence, with upstream's CPU-share branch disabled
    (`n_cpu = 0`, `n_zero = 0`) so the arm is a pure cache-replacement
    policy. Upstream's file is not vendored here, so the numbers are
    pinned; the port must reproduce them miss for miss.

    Upstream's policy is llama.cpp PR #31233, which the MoE-cache
    CPU-share work of PR #30966 builds on.
    """

    LAYERS, NUM_EXPERTS, TOP_K, PASSES = 3, 16, 2, 40

    def sequence(self):
        rng = random.Random(1234)
        steps = []
        for p in range(self.PASSES):
            for layer in range(self.LAYERS):
                steps.append((layer, rng.sample(range(self.NUM_EXPERTS),
                                                self.TOP_K)))
        return steps

    def test_the_per_layer_placement_matches_upstream(self):
        for slots, expected in ((4, 182), (8, 128)):
            got = 0
            for layer in range(self.LAYERS):
                cache = ClassLru([layer], self.NUM_EXPERTS, slots)
                for lay, group in self.sequence():
                    if lay != layer:
                        continue
                    got += cache.plan(layer, list(group))
            self.assertEqual(got, expected, f"per-layer at {slots} slots")

    def test_the_pool_placement_matches_upstream(self):
        for slots, expected in ((4, 189), (8, 133)):
            cache = ClassLru(list(range(self.LAYERS)), self.NUM_EXPERTS,
                             slots * self.LAYERS)
            got = sum(cache.plan(layer, list(group))
                      for layer, group in self.sequence())
            self.assertEqual(got, expected, f"pool at {slots} slots")

    def test_it_never_misses_less_than_the_oracle(self):
        # The oracle bound is the harness's own invariant, and the new arms
        # are checked against it here so a port that somehow beat Belady
        # would fail here rather than in a replay nobody reads.
        steps = self.one_layer_steps()
        for slots in (4, 8):
            bound = sum(misses_by_pass(steps, oracle_factory(steps, slots)))
            got = sum(misses_by_pass(steps, lambda layer: ClassLruPerLayer(
                layer, slots, self.NUM_EXPERTS)))
            self.assertGreaterEqual(got, bound, f"at {slots} slots")

    def one_layer_steps(self):
        groups = [g for _, g in self.sequence()][:20]
        cap = from_capture(capture([groups], self.NUM_EXPERTS, self.TOP_K),
                            "synthetic")
        return schedule([cap])


if __name__ == "__main__":
    unittest.main()
