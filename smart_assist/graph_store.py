from __future__ import annotations

import json
import re
from typing import Optional

import networkx as nx

from .models import Document, Entity, Relation
from .utils import dedupe_preserve_order


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_document(self, document: Document) -> None:
        self.graph.add_node(
            document.id,
            kind="document",
            text=document.text,
            metadata=document.metadata,
        )

    def add_entity(self, entity: Entity) -> None:
        self.graph.add_node(
            entity.id,
            kind="entity",
            name=entity.name,
            label=entity.label,
            source_document_id=entity.source_document_id,
            confidence=entity.confidence,
            aliases=entity.aliases,
            metadata=entity.metadata,
        )

    def add_relation(self, relation: Relation) -> None:
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            key=relation.relation_type,
            kind="relation",
            relation_type=relation.relation_type,
            evidence=relation.evidence,
            confidence=relation.confidence,
            source_document_id=relation.source_document_id,
            metadata=relation.metadata,
        )

    def add_mention(self, document_id: str, entity_id: str) -> None:
        self.graph.add_edge(document_id, entity_id, key=f"mentions:{entity_id}", kind="mentions")

    def add_similarity(self, left_id: str, right_id: str, score: float, reason: str) -> None:
        self.graph.add_edge(
            left_id,
            right_id,
            key=f"similar:{right_id}",
            kind="similarity",
            score=score,
            reason=reason,
        )

    def entities(self) -> list[dict]:
        return [dict(id=node_id, **data) for node_id, data in self.graph.nodes(data=True) if data.get("kind") == "entity"]

    def relations(self) -> list[dict]:
        rows = []
        for source, target, _, data in self.graph.edges(keys=True, data=True):
            if data.get("kind") == "relation":
                rows.append({"source": source, "target": target, **data})
        return rows

    def related_entities(self, entity_id: str) -> list[str]:
        neighbors = []
        for _, target, _, data in self.graph.out_edges(entity_id, keys=True, data=True):
            if data.get("kind") == "relation":
                neighbors.append(target)
        return neighbors

    def find_entity_by_name(self, name: str) -> list[dict]:
        lowered = name.lower()
        return [
            dict(id=node_id, **data)
            for node_id, data in self.graph.nodes(data=True)
            if data.get("kind") == "entity"
            and (
                lowered in str(data.get("name", "")).lower()
                or any(lowered in str(alias).lower() for alias in data.get("aliases", []))
            )
        ]

    def find_entities_in_text(self, text: str) -> list[dict]:
        lowered = text.lower()
        matches = []
        seen = set()
        candidates = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("kind") != "entity":
                continue
            candidates.append((node_id, data))

        candidates.sort(key=lambda item: len(str(item[1].get("name", ""))), reverse=True)
        for node_id, data in candidates:
            names = dedupe_preserve_order([str(data.get("name", "")), *(str(alias) for alias in data.get("aliases", []))])
            for candidate in names:
                candidate = candidate.strip()
                if not candidate:
                    continue
                if candidate.isalnum():
                    pattern = rf"(?<!\w){re.escape(candidate.lower())}(?!\w)"
                    matched = re.search(pattern, lowered) is not None
                else:
                    matched = candidate.lower() in lowered
                if matched:
                    if node_id not in seen:
                        seen.add(node_id)
                        matches.append(dict(id=node_id, **data))
                    break
        return matches


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self.database = database
        self._driver = None
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception as exc:  # pragma: no cover - optional dependency/runtime connection
            raise RuntimeError(f"Unable to initialize Neo4j driver: {exc}") from exc

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def sync(self, graph: KnowledgeGraph) -> None:
        if not self._driver:
            return
        with self._driver.session(database=self.database) as session:
            for node_id, data in graph.graph.nodes(data=True):
                session.execute_write(self._merge_node, node_id, data)
            for source, target, _, data in graph.graph.edges(keys=True, data=True):
                session.execute_write(self._merge_edge, source, target, data)

    def load(self, graph: KnowledgeGraph) -> list[Document]:
        if not self._driver:
            return []

        graph.graph.clear()
        documents: list[Document] = []

        with self._driver.session(database=self.database) as session:
            node_rows = session.run(
                """
                MATCH (n:KGNode)
                RETURN n.id AS id, properties(n) AS props
                """
            )
            for record in node_rows:
                node_id = record["id"]
                props = record["props"]
                if not isinstance(props, dict):
                    continue
                kind = props.get("kind", "node")
                node_data = self._node_data_from_props(props)
                if kind == "document":
                    document = Document(
                        id=node_id,
                        text=str(node_data.get("text", "")),
                        metadata=dict(node_data.get("metadata", {})),
                    )
                    graph.add_document(document)
                    documents.append(document)
                elif kind == "entity":
                    graph.add_entity(
                        Entity(
                            id=node_id,
                            name=str(node_data.get("name", "")),
                            label=str(node_data.get("label", "Concept")),
                            source_document_id=str(node_data.get("source_document_id", "")),
                            confidence=float(node_data.get("confidence", 0.5) or 0.5),
                            aliases=[str(item) for item in node_data.get("aliases", []) if item is not None],
                            metadata=dict(node_data.get("metadata", {})),
                        )
                    )
                else:
                    graph.graph.add_node(node_id, **node_data)

            edge_rows = session.run(
                """
                MATCH (a:KGNode)-[r]->(b:KGNode)
                RETURN a.id AS source, b.id AS target, type(r) AS rel_type, properties(r) AS props
                """
            )
            for record in edge_rows:
                source = record["source"]
                target = record["target"]
                rel_type = str(record["rel_type"])
                props = record["props"]
                if not isinstance(props, dict):
                    props = {}
                if rel_type == "MENTIONS":
                    graph.graph.add_edge(source, target, key=f"mentions:{target}", kind="mentions")
                elif rel_type == "SIMILAR_TO":
                    graph.graph.add_edge(
                        source,
                        target,
                        key=f"similar:{target}",
                        kind="similarity",
                        score=props.get("score", 0.0),
                        reason=props.get("reason", "neo4j"),
                    )
                else:
                    graph.graph.add_edge(
                        source,
                        target,
                        key=rel_type,
                        kind="relation",
                        relation_type=rel_type,
                        evidence=str(props.get("evidence", "")),
                        confidence=float(props.get("confidence", 0.5) or 0.5),
                        source_document_id=str(props.get("source_document_id", "")),
                        metadata=self._metadata_from_value(props.get("metadata", {})),
                    )

        return documents

    @staticmethod
    def _merge_node(tx, node_id: str, data: dict) -> None:
        kind = data.get("kind", "node")
        tx.run(
            """
            MERGE (n:KGNode {id: $id})
            SET n += $props,
                n.kind = $kind
            """,
            id=node_id,
            kind=kind,
            props=Neo4jGraphStore._neo4j_props(data, exclude={"kind"}),
        )

    @staticmethod
    def _merge_edge(tx, source: str, target: str, data: dict) -> None:
        rel_type = data.get("relation_type") or data.get("kind", "RELATED_TO")
        if rel_type == "mentions":
            rel_type = "MENTIONS"
        elif rel_type == "similarity":
            rel_type = "SIMILAR_TO"
        tx.run(
            f"""
            MATCH (a:KGNode {{id: $source}})
            MATCH (b:KGNode {{id: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $props
            """,
            source=source,
            target=target,
            props=Neo4jGraphStore._neo4j_props(data, exclude={"kind", "relation_type"}),
        )

    @staticmethod
    def _neo4j_props(data: dict, exclude: set[str] | None = None) -> dict:
        excluded = exclude or set()
        return {k: Neo4jGraphStore._neo4j_value(v) for k, v in data.items() if k not in excluded}

    @staticmethod
    def _neo4j_value(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (list, tuple, set)):
            normalized = []
            for item in value:
                coerced = Neo4jGraphStore._neo4j_value(item)
                if isinstance(coerced, (dict, list, tuple, set)):
                    coerced = json.dumps(coerced, ensure_ascii=False, sort_keys=True)
                normalized.append(coerced)
            return normalized
        return str(value)

    @staticmethod
    def _node_data_from_props(props: dict) -> dict:
        node_data = dict(props)
        metadata = node_data.get("metadata")
        if isinstance(metadata, str):
            node_data["metadata"] = Neo4jGraphStore._metadata_from_value(metadata)
        aliases = node_data.get("aliases")
        if isinstance(aliases, list):
            node_data["aliases"] = [str(item) for item in aliases if item is not None]
        return node_data

    @staticmethod
    def _metadata_from_value(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                return {"value": value}
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        return {}
