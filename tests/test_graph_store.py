from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_assist.graph_store import Neo4jGraphStore


class Neo4jPropertyTests(unittest.TestCase):
    def test_nested_metadata_is_sanitized_for_neo4j(self) -> None:
        props = Neo4jGraphStore._neo4j_props(
            {
                "kind": "document",
                "text": "OpenAI partnered with Microsoft.",
                "metadata": {
                    "format": "txt",
                    "source_file": "sample_data/commanda.txt",
                },
                "aliases": ["OpenAI", {"bad": "value"}],
            },
            exclude={"kind"},
        )

        self.assertEqual(props["text"], "OpenAI partnered with Microsoft.")
        self.assertIsInstance(props["metadata"], str)
        self.assertIn('"format": "txt"', props["metadata"])
        self.assertEqual(props["aliases"][0], "OpenAI")
        self.assertIsInstance(props["aliases"][1], str)


if __name__ == "__main__":
    unittest.main()
