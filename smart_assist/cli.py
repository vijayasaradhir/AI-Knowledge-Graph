from __future__ import annotations

import argparse
import json
from pathlib import Path

from .code_graph import CodeGraphIngestor
from .pipeline import SmartAssistPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smart-assist", description="Smart Assist knowledge graph demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest documents and build the graph")
    ingest.add_argument("--input", required=True, help="Path to a folder or file with documents")
    ingest.add_argument(
        "--source-type",
        default="auto",
        choices=["auto", "text", "code"],
        help="How to interpret the input path",
    )

    query = subparsers.add_parser("query", help="Run a graph query against ingested data")
    query.add_argument("--input", required=False, help="Path to a folder or file with documents")
    query.add_argument("--text", required=True, help="Query text")
    query.add_argument(
        "--source-type",
        default="auto",
        choices=["auto", "text", "code"],
        help="How to interpret the input path",
    )

    demo = subparsers.add_parser("demo", help="Run the app against bundled sample data")

    code_export = subparsers.add_parser("code-export", help="Export code graph metadata to JSON")
    code_export.add_argument("--input", required=True, help="Path to a folder or file with source code")
    code_export.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the timestamped metadata JSON file should be written",
    )

    code_ingest = subparsers.add_parser("code-ingest", help="Ingest previously exported code graph metadata JSON")
    code_ingest.add_argument("--input", required=True, help="Path to the code graph metadata JSON file")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = SmartAssistPipeline()
    code_graph = CodeGraphIngestor(enabled=True)

    try:
        if args.command == "demo":
            input_path = Path(__file__).resolve().parents[1] / "sample_data" / "documents.json"
            documents = pipeline.load_documents(input_path)
            results = pipeline.ingest(documents)
            print(json.dumps(pipeline.export_summary(), indent=2))
            print()
            for item in results:
                print(f"{item.document.id}: {len(item.entities)} entities, {len(item.relations)} relations")
            print()
            for query in ["OpenAI", "BrightPath AI", "GitHub"]:
                answer = pipeline.reason(query)
                print(f"Query: {query}")
                print(f"Answer: {answer.answer}")
                print(answer.explanation)
                print()
            return

        if args.command == "code-export":
            output_path = code_graph.export_metadata(args.input, output_dir=Path(args.output_dir))
            print(json.dumps({"output_file": str(output_path)}, indent=2))
            return

        if args.command == "code-ingest":
            metadata_pipeline = SmartAssistPipeline(enable_code_graph=True)
            metadata_pipeline.extractor._client = None
            metadata_pipeline._openai_client = None
            try:
                code_graph.ingest_metadata(args.input, metadata_pipeline.graph)
                if metadata_pipeline.neo4j_store:
                    try:
                        metadata_pipeline.neo4j_store.sync(metadata_pipeline.graph)
                    except Exception as exc:
                        print(f"Neo4j sync failed: {exc}")
                print(json.dumps(metadata_pipeline.export_summary(), indent=2))
            finally:
                metadata_pipeline.close()
            return

        if getattr(args, "input", None):
            documents = pipeline.load_documents(args.input, source_type=getattr(args, "source_type", "auto"))
            pipeline.ingest(documents)

        if args.command == "ingest":
            if not getattr(args, "input", None):
                raise SystemExit("The ingest command requires --input.")
            print(json.dumps(pipeline.export_summary(), indent=2))
        elif args.command == "query":
            if not getattr(args, "input", None) and not pipeline.neo4j_enabled:
                raise SystemExit("Query without --input requires Neo4j to be configured and already populated.")
            result = pipeline.reason(args.text)
            print(f"Answer: {result.answer}")
            print()
            print(json.dumps(
                {
                    "answer": result.answer,
                    "explanation": result.explanation,
                    "matched_entities": [entity.name for entity in result.matched_entities],
                    "similar_entities": [entity.name for entity in result.similar_entities],
                    "supporting_relations": [
                        {
                            "source_id": relation.source_id,
                            "target_id": relation.target_id,
                            "type": relation.relation_type,
                            "evidence": relation.evidence,
                        }
                        for relation in result.supporting_relations
                    ],
                },
                indent=2,
            ))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
