import itertools
import random
import unittest

from expert_policy_replay import (
    DecayedLfu,
    EngineLfu,
    Lru,
    Oracle,
    compulsory,
    from_capture,
    next_use_index,
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


if __name__ == "__main__":
    unittest.main()
