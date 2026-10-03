from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_assist.pipeline import SmartAssistPipeline
from smart_assist.models import Document, Entity, Relation


class GraphReasoningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = SmartAssistPipeline()
        self.pipeline.extractor._client = None
        self.pipeline._openai_client = None
        sample_path = ROOT / "sample_data" / "documents.json"
        documents = self.pipeline.load_documents(sample_path)
        self.pipeline.ingest(documents)

    def tearDown(self) -> None:
        self.pipeline.close()

    def test_acquisition_query_returns_microsoft(self) -> None:
        result = self.pipeline.reason("What company aquired GitHub?")
        self.assertIn("Microsoft", result.answer)
        self.assertIn("GitHub", result.answer)

    def test_reverse_acquisition_query_returns_github_subject(self) -> None:
        result = self.pipeline.reason("Who was GitHub acquired by?")
        self.assertIn("GitHub was acquired by Microsoft", result.answer)

    def test_collaboration_query_returns_openai_answer(self) -> None:
        result = self.pipeline.reason("Who collaborates with OpenAI?")
        self.assertIn("OpenAI", result.answer)
        self.assertIn("Microsoft", result.answer)
        self.assertTrue(result.answer)

    def test_json_records_without_text_field_use_content(self) -> None:
        documents = self.pipeline.load_documents(
            [
                {
                    "id": "doc-x",
                    "content": "OpenAI collaborated with DataFlow Labs on embeddings.",
                    "topic": "ai",
                }
            ]
        )
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].text, "OpenAI collaborated with DataFlow Labs on embeddings.")
        self.assertEqual(documents[0].metadata["topic"], "ai")

    def test_reason_can_hydrate_from_neo4j_without_input(self) -> None:
        pipeline = SmartAssistPipeline()
        pipeline.extractor._client = None
        pipeline._openai_client = None
        pipeline.documents = []

        class FakeNeo4jStore:
            def load(self, graph):
                document = Document(id="doc-neo4j", text="OpenAI collaborated with Microsoft.", metadata={})
                graph.add_document(document)
                graph.add_entity(
                    Entity(
                        id="ent-openai",
                        name="OpenAI",
                        label="Organization",
                        source_document_id="doc-neo4j",
                        confidence=1.0,
                        aliases=[],
                        metadata={"documents": ["doc-neo4j"]},
                    )
                )
                graph.add_entity(
                    Entity(
                        id="ent-microsoft",
                        name="Microsoft",
                        label="Organization",
                        source_document_id="doc-neo4j",
                        confidence=1.0,
                        aliases=[],
                        metadata={"documents": ["doc-neo4j"]},
                    )
                )
                graph.add_relation(
                    Relation(
                        source_id="ent-openai",
                        target_id="ent-microsoft",
                        relation_type="collaborate_with",
                        evidence="OpenAI collaborated with Microsoft.",
                        source_document_id="doc-neo4j",
                        confidence=1.0,
                        metadata={},
                    )
                )
                return [document]

            def close(self) -> None:
                return None

        pipeline.neo4j_store = FakeNeo4jStore()
        pipeline.neo4j_enabled = True

        result = pipeline.reason("Who collaborates with OpenAI?")
        self.assertIn("OpenAI", result.answer)
        self.assertIn("Microsoft", result.answer)


if __name__ == "__main__":
    unittest.main()
