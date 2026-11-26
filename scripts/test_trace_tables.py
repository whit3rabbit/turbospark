import pathlib
import sys
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "crates" / "lab" / "tools"))
import trace_tables  # noqa: E402

COLUMNS = ["start", "duration", "channel-name", "process", "cmdbuffer-id"]


def table(rows):
    """A metal-gpu-intervals export, with xctrace's id/ref compression."""
    cols = "".join(f"<col><mnemonic>{c}</mnemonic></col>" for c in COLUMNS)
    return ET.fromstring(
        f"<trace-query-result><node><schema>{cols}</schema>{''.join(rows)}"
        "</node></trace-query-result>")


def row(start_ms, dur_ms, buffer, channel="Compute", pid=7):
    return (
        f"<row><start-time>{int(start_ms * 1e6)}</start-time>"
        f"<duration>{int(dur_ms * 1e6)}</duration>"
        f'<channel fmt="{channel}">{channel}</channel>'
        f"<process><pid>{pid}</pid></process>{buffer}</row>")


def run(rows, tokens=None):
    with mock.patch.object(trace_tables, "export", return_value=table(rows)):
        return trace_tables.encoders("unused", 7, tokens)


class PerToken(unittest.TestCase):
    rows = [
        row(0, 2, '<cb id="9">100</cb>'),
        # The same buffer again, through a ref; overlapping the first.
        row(1, 2, '<cb ref="9"/>'),
        row(5, 1, "<cb>101</cb>"),
        # No buffer recorded, another process, another channel.
        row(6, 2, "<cb>0</cb>"),
        row(8, 9, "<cb>200</cb>", pid=8),
        row(8, 9, "<cb>201</cb>", channel="Vertex"),
    ]

    def test_totals_count_the_union_and_distinct_buffers(self):
        got = run(self.rows)
        self.assertEqual(got["count"], 4)
        # Busy 0..3 and 5..8 of an 8 ms window.
        self.assertAlmostEqual(got["busy_ms"], 6.0)
        self.assertAlmostEqual(got["idle_ms"], 2.0)
        self.assertEqual(got["command_buffers"], 2)
        self.assertIsNone(got["idle_ms_per_token"])
        self.assertIsNone(got["command_buffers_per_token"])

    def test_per_token_figures_divide_by_the_given_count(self):
        got = run(self.rows, tokens=4)
        self.assertEqual(got["window_tokens"], 4)
        self.assertAlmostEqual(got["busy_ms_per_token"], 1.5)
        self.assertAlmostEqual(got["idle_ms_per_token"], 0.5)
        self.assertAlmostEqual(got["command_buffers_per_token"], 0.5)

    def test_no_compute_rows(self):
        self.assertEqual(run([], tokens=4), {"count": 0})


if __name__ == "__main__":
    unittest.main()
