from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_assist.code_graph import CodeGraphIngestor
from smart_assist.graph_store import KnowledgeGraph
from smart_assist.pipeline import SmartAssistPipeline


class CodeGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = SmartAssistPipeline(enable_code_graph=True)
        self.pipeline.extractor._client = None
        self.pipeline._openai_client = None

    def tearDown(self) -> None:
        self.pipeline.close()

    def test_python_and_java_source_are_loaded_into_the_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            (temp_dir / "sample.py").write_text(
                "import os\n\n"
                "class Greeter:\n"
                "    def say_hello(self, name):\n"
                "        return greet(name)\n\n"
                "def greet(name):\n"
                "    return f'Hello {name}'\n",
                encoding="utf-8",
            )
            (temp_dir / "BaseSample.java").write_text(
                "package demo;\n\n"
                "public class BaseSample {\n"
                "    public void base() {}\n"
                "}\n",
                encoding="utf-8",
            )
            (temp_dir / "Utils.java").write_text(
                "package demo;\n\n"
                "public class Utils {\n"
                "    public static void execute() {}\n"
                "}\n",
                encoding="utf-8",
            )
            (temp_dir / "Sample.java").write_text(
                "package demo;\n"
                "import java.util.List;\n\n"
                "public class Sample extends BaseSample {\n"
                "    public void run() {\n"
                "        execute();\n"
                "    }\n"
                "}\n",
                encoding="utf-8",
            )
            (temp_dir / "ItemController.java").write_text(
                "package demo;\n\n"
                "import org.springframework.web.bind.annotation.GetMapping;\n"
                "import org.springframework.web.bind.annotation.RestController;\n\n"
                "@RestController\n"
                "public class ItemController {\n"
                "    @GetMapping(\"/items/{id}\")\n"
                "    public String getItem() {\n"
                "        return execute();\n"
                "    }\n"
                "}\n",
                encoding="utf-8",
            )

            documents = self.pipeline.load_documents(temp_dir, source_type="code")
            results = self.pipeline.ingest(documents)

        self.assertEqual(len(results), 5)
        self.assertTrue(self.pipeline.export_summary()["code_graph_enabled"])
        self.assertTrue(self.pipeline.graph.find_entity_by_name("Greeter"))
        self.assertTrue(self.pipeline.graph.find_entity_by_name("Sample"))
        self.assertTrue(self.pipeline.graph.find_entity_by_name("BaseSample"))
        self.assertTrue(self.pipeline.graph.find_entity_by_name("Utils"))
        self.assertTrue(self.pipeline.graph.find_entity_by_name("ItemController"))
        self.assertTrue(self.pipeline.graph.find_entity_by_name("greet"))
        self.assertTrue(self.pipeline.graph.find_entity_by_name("run"))
        self.assertTrue(self.pipeline.graph.find_entity_by_name("getItem"))
        self.assertTrue(self.pipeline.graph.find_entity_by_name("execute"))

        sample_entities = self.pipeline.graph.find_entity_by_name("Sample")
        execute_entities = self.pipeline.graph.find_entity_by_name("execute")
        controller_entities = self.pipeline.graph.find_entity_by_name("ItemController")
        get_item_entities = self.pipeline.graph.find_entity_by_name("getItem")
        self.assertTrue(sample_entities)
        self.assertTrue(execute_entities)
        self.assertTrue(controller_entities)
        self.assertTrue(get_item_entities)

        sample_id = sample_entities[0]["id"]
        execute_id = execute_entities[0]["id"]
        get_item_id = get_item_entities[0]["id"]
        controller_id = controller_entities[0]["id"]

        relation_types = {
            (source, target, data.get("relation_type"))
            for source, target, _, data in self.pipeline.graph.graph.edges(keys=True, data=True)
            if data.get("kind") == "relation"
        }
        self.assertIn((sample_id, next(item["id"] for item in self.pipeline.graph.find_entity_by_name("run")), "contains"), relation_types)
        self.assertIn(
            (
                next(item["id"] for item in self.pipeline.graph.find_entity_by_name("run")),
                execute_id,
                "calls",
            ),
            relation_types,
        )

        controller_node = self.pipeline.graph.graph.nodes[controller_id]
        get_item_node = self.pipeline.graph.graph.nodes[get_item_id]
        self.assertIn("RestController", controller_node.get("metadata", {}).get("annotations", []))
        self.assertIn("GetMapping", get_item_node.get("metadata", {}).get("annotations", []))
        self.assertEqual(get_item_node.get("metadata", {}).get("endpoint", {}).get("http_method"), "GET")
        self.assertEqual(get_item_node.get("metadata", {}).get("endpoint", {}).get("path"), "/items/{id}")

    def test_code_loading_is_disabled_by_default(self) -> None:
        plain_pipeline = SmartAssistPipeline()
        try:
            with self.assertRaises(RuntimeError):
                plain_pipeline.load_documents("does-not-matter", source_type="code")
        finally:
            plain_pipeline.close()

    def test_code_metadata_export_and_ingest_round_trip(self) -> None:
        ingestor = CodeGraphIngestor(enabled=True)
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            source_dir = temp_dir / "source"
            source_dir.mkdir()
            (source_dir / "sample.py").write_text(
                "class Greeter:\n"
                "    def say_hello(self):\n"
                "        return greet()\n\n"
                "def greet():\n"
                "    return 'hello'\n",
                encoding="utf-8",
            )
            (source_dir / "Controller.java").write_text(
                "package demo;\n\n"
                "import org.springframework.web.bind.annotation.GetMapping;\n"
                "import org.springframework.web.bind.annotation.RestController;\n\n"
                "@RestController\n"
                "public class Controller {\n"
                "    @GetMapping(\"/items/{id}\")\n"
                "    public String getItem() {\n"
                "        return \"ok\";\n"
                "    }\n"
                "}\n",
                encoding="utf-8",
            )

            metadata_path = ingestor.export_metadata(source_dir, output_dir=temp_dir)
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["kind"], "code_graph_metadata")
            self.assertEqual(len(payload["documents"]), 2)
            self.assertTrue(payload["documents"][0]["entities"])
            self.assertTrue(payload["documents"][0]["relations"])
            self.assertTrue(metadata_path.name.startswith("code_graph_metadata_"))

            controller_doc = next(doc for doc in payload["documents"] if doc["source_file"].endswith("Controller.java"))
            controller_class = next(entity for entity in controller_doc["entities"] if entity["name"] == "Controller")
            controller_method = next(entity for entity in controller_doc["entities"] if entity["name"] == "getItem")
            self.assertIn("spring", controller_class["metadata"])
            self.assertTrue(controller_class["metadata"]["spring"].get("is_controller"))
            self.assertIn("annotations", controller_method["metadata"])
            self.assertIn("GetMapping", controller_method["metadata"]["annotations"])
            self.assertIn("spring", controller_method["metadata"])
            self.assertEqual(controller_method["metadata"]["spring"]["endpoint"]["full_path"], "/items/{id}")
            self.assertEqual(controller_method["metadata"]["endpoint"]["http_method"], "GET")
            self.assertEqual(controller_method["metadata"]["endpoint"]["path"], "/items/{id}")

            graph = KnowledgeGraph()
            results = ingestor.ingest_metadata(metadata_path, graph)

        self.assertEqual(len(results), 2)
        self.assertTrue(graph.find_entity_by_name("Greeter"))
        self.assertTrue(graph.find_entity_by_name("say_hello"))
        self.assertTrue(graph.find_entity_by_name("Controller"))
        self.assertTrue(graph.find_entity_by_name("getItem"))
        relation_types = {
            data.get("relation_type")
            for _, _, _, data in graph.graph.edges(keys=True, data=True)
            if data.get("kind") == "relation"
        }
        self.assertIn("contains", relation_types)
        self.assertIn("calls", relation_types)


if __name__ == "__main__":
    unittest.main()
