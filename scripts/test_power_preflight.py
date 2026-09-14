"""Exercise power.sh before sudo, without sampling or launching a model."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class PowerPreflightTests(unittest.TestCase):
    def run_preflight(self, process, output="explicit"):
        script = Path(os.environ.get("POWER_SCRIPT", Path(__file__).with_name("power.sh")))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Use grep's ERE engine, as pgrep does, against a controlled process
            # list. A sentinel stops the real script at its sudo boundary.
            shims = {
                "pgrep": '#!/bin/sh\nprintf "%s\\n" "$TEST_PROCESS" | /usr/bin/grep -E "$2"\n',
                "pmset": '#!/bin/sh\necho "Now drawing from AC Power"\n',
                "sudo": '#!/bin/sh\necho SUDO_SENTINEL >&2\nexit 1\n',
            }
            for name, content in shims.items():
                path = root / name
                path.write_text(content)
                path.chmod(0o755)
            env = dict(os.environ)
            env.update(
                PATH=f"{root}:/usr/bin:/bin",
                TEST_PROCESS=process,
                MODEL=directory,
                RUST_BENCH="/usr/bin/true",
                TMPDIR=str(root),
                LABEL="ac",
                COOLING="auto",
                ARMS="seq,chunked",
            )
            capture = root / "capture"
            if output in ("explicit", "unsafe"):
                env["OUT"] = str(capture)
            else:
                env.pop("OUT", None)
            if output == "unsafe":
                capture.mkdir(mode=0o777)
                capture.chmod(0o777)
            result = subprocess.run(
                ["/bin/bash", str(script)], env=env, capture_output=True,
                text=True, timeout=10,
            )
            captures = list(root.glob("turbospark-power.*"))
            capture_modes = [path.stat().st_mode & 0o777 for path in captures]
            default_rows = [(path / "rows.tsv").is_file() for path in captures]
            return (result, capture.exists(), capture_modes, default_rows,
                    (capture / "rows.tsv").exists())

    def test_model_processes_refused_before_capture(self):
        for name in (
            "turbospark-check", "turbospark-server", "turbospark-bench",
            "TurboSparkApp", "MferenceCLI", "mference-bench", "mlx_lm.generate",
        ):
            with self.subTest(process=name):
                process = f"123 /Applications/Test/{name}"
                result, created, _, _, _ = self.run_preflight(process)
                self.assertEqual(result.returncode, 2)
                self.assertIn("another model process is running", result.stderr)
                self.assertIn(process, result.stderr)
                self.assertNotIn("SUDO_SENTINEL", result.stderr)
                self.assertFalse(created)

    def test_idle_process_list_reaches_authentication(self):
        for process in ("", "123 /usr/bin/editor", "123 /bin/bash scripts/power.sh"):
            with self.subTest(process=process):
                result, created, _, _, _ = self.run_preflight(process)
                self.assertEqual(result.returncode, 2)
                self.assertIn("SUDO_SENTINEL", result.stderr)
                self.assertNotIn("another model process is running", result.stderr)
                self.assertTrue(created)

    def test_default_output_is_private_and_unpredictable(self):
        result, created, modes, rows, _ = self.run_preflight("", output="default")
        self.assertEqual(result.returncode, 2)
        self.assertIn("SUDO_SENTINEL", result.stderr)
        self.assertFalse(created)
        self.assertEqual(modes, [0o700])
        self.assertEqual(rows, [True])

    def test_world_accessible_explicit_output_is_refused(self):
        result, created, _, _, rows_created = self.run_preflight("", output="unsafe")
        self.assertEqual(result.returncode, 2)
        self.assertTrue(created)
        self.assertIn("group and other permissions must be disabled", result.stderr)
        self.assertNotIn("SUDO_SENTINEL", result.stderr)
        self.assertFalse(rows_created)


if __name__ == "__main__":
    unittest.main()
