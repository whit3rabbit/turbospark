"""Exercise power.sh before sudo, without sampling or launching a model."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class PowerPreflightTests(unittest.TestCase):
    def run_preflight(self, process, cooling="auto", max_status="0"):
        script = Path(os.environ.get("POWER_SCRIPT", Path(__file__).with_name("power.sh")))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Use grep's ERE engine, as pgrep does, against a controlled process
            # list. A sentinel stops the real script at its sudo boundary.
            shims = {
                "pgrep": '#!/bin/sh\nprintf "%s\\n" "$TEST_PROCESS" | /usr/bin/grep -E "$2"\n',
                "pmset": '#!/bin/sh\necho "Now drawing from AC Power"\n',
                "sudo": '#!/bin/sh\necho SUDO_SENTINEL >&2\nexit 1\n',
                "thermalforge": (
                    '#!/bin/sh\necho "$1" >> "$THERMALFORGE_TRACE"\n'
                    '[ "$1" = max ] && exit "$TEST_MAX_STATUS"\nexit 0\n'
                ),
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
                OUT=str(root / "capture"),
                LABEL="ac",
                COOLING=cooling,
                ARMS="seq,chunked",
                TEST_MAX_STATUS=max_status,
                THERMALFORGE_TRACE=str(root / "thermalforge.trace"),
            )
            result = subprocess.run(
                ["/bin/bash", str(script)], env=env, capture_output=True,
                text=True, timeout=10,
            )
            trace_path = root / "thermalforge.trace"
            trace = trace_path.read_text().splitlines() if trace_path.exists() else []
            return result, (root / "capture").exists(), trace

    def test_model_processes_refused_before_capture(self):
        for name in (
            "turbospark-check", "turbospark-server", "turbospark-bench",
            "TurboSparkApp", "MferenceCLI", "mference-bench", "mlx_lm.generate",
        ):
            with self.subTest(process=name):
                process = f"123 /Applications/Test/{name}"
                result, created, _ = self.run_preflight(process)
                self.assertEqual(result.returncode, 2)
                self.assertIn("another model process is running", result.stderr)
                self.assertIn(process, result.stderr)
                self.assertNotIn("SUDO_SENTINEL", result.stderr)
                self.assertFalse(created)

    def test_idle_process_list_reaches_authentication(self):
        for process in ("", "123 /usr/bin/editor", "123 /bin/bash scripts/power.sh"):
            with self.subTest(process=process):
                result, created, _ = self.run_preflight(process)
                self.assertEqual(result.returncode, 2)
                self.assertIn("SUDO_SENTINEL", result.stderr)
                self.assertNotIn("another model process is running", result.stderr)
                self.assertTrue(created)

    def test_partial_fan_pin_failure_restores_automatic_curve(self):
        result, created, trace = self.run_preflight(
            "", cooling="max", max_status="42",
        )
        self.assertEqual(result.returncode, 2)
        self.assertTrue(created)
        self.assertEqual(trace, ["max", "auto"])
        self.assertIn("thermalforge max failed", result.stderr)
        self.assertIn("fans restored to the machine's own curve", result.stdout)


if __name__ == "__main__":
    unittest.main()
