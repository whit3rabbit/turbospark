#!/usr/bin/env python3
"""Offline tests for kld_mlx_affine.py's dump provenance guard."""

import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest


def load_driver():
    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    sys.modules["numpy"] = numpy
    kld = types.ModuleType("kld")
    kld.divergences = lambda *_: None
    kld.perplexity = lambda *_: None
    sys.modules["kld"] = kld
    path = pathlib.Path(__file__).with_name("kld_mlx_affine.py")
    spec = importlib.util.spec_from_file_location("kld_mlx_affine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER = load_driver()


class DumpIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.install = pathlib.Path(self.temp.name)
        self.spec = {
            "repo": "publisher/model-a",
            "revision": "a" * 40,
        }
        (self.install / "manifest.json").write_text(
            json.dumps({"modelID": self.spec["repo"]})
        )
        self.meta = {"install": str(self.install)}

    def tearDown(self):
        self.temp.cleanup()

    def test_matching_manifest_identity_passes_without_a_receipt(self):
        DRIVER.assert_dump_matches_checkpoint(self.meta, self.spec)

    def test_swapped_checkpoint_name_is_refused(self):
        wrong = {**self.spec, "repo": "publisher/model-b"}
        with self.assertRaisesRegex(SystemExit, "dump install does not match"):
            DRIVER.assert_dump_matches_checkpoint(self.meta, wrong)

    def test_receipt_revision_is_checked_when_present(self):
        (self.install / "verified-install.json").write_text(
            json.dumps(
                {
                    "sourceRepoId": self.spec["repo"],
                    "sourceRevision": "b" * 40,
                }
            )
        )
        with self.assertRaisesRegex(SystemExit, "source revision does not match"):
            DRIVER.assert_dump_matches_checkpoint(self.meta, self.spec)


if __name__ == "__main__":
    unittest.main()
