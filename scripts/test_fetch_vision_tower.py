"""Offline security tests for fetch_vision_tower.py."""

import importlib.util
from pathlib import Path
import struct
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("fetch_vision_tower.py")
SPEC = importlib.util.spec_from_file_location("fetch_vision_tower", SCRIPT)
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


class Response:
    def __init__(self, body, start, end, total, status=206):
        self.body = body
        self.position = 0
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}

    def getcode(self):
        return self.status

    def read(self, size=-1):
        if size < 0:
            size = len(self.body) - self.position
        result = self.body[self.position : self.position + size]
        self.position += len(result)
        return result

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FetchVisionTowerTests(unittest.TestCase):
    def test_rejects_oversized_header_before_requesting_it(self):
        prefix = struct.pack("<Q", fetcher.HEADER_LIMIT + 1)
        response = Response(prefix, 0, 7, fetcher.HEADER_LIMIT + 9)
        with mock.patch.object(
            fetcher.urllib.request, "urlopen", return_value=response
        ) as get:
            with self.assertRaisesRegex(ValueError, "header size"):
                fetcher._read_header("https://example.invalid/shard")
        self.assertEqual(get.call_count, 1)

    def test_rejects_server_that_ignores_range(self):
        response = Response(b"abcdefgh", 0, 7, 8, status=200)
        with mock.patch.object(
            fetcher.urllib.request, "urlopen", return_value=response
        ):
            with self.assertRaisesRegex(ValueError, "did not honor"):
                fetcher._ranged_bytes("https://example.invalid/shard", 0, 7, 8)

    def test_rejects_tensor_offset_outside_shard(self):
        with self.assertRaisesRegex(ValueError, "outside the shard"):
            fetcher._tensor_range({"data_offsets": [1, 11]}, 10)


if __name__ == "__main__":
    unittest.main()
