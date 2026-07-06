from __future__ import annotations

import re
from typing import List, Optional

from .graph_store import KnowledgeGraph
from .models import Entity, QueryIntent, QueryResult, Relation


class GraphReasoner:
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def query(self, text: str, top_k: int = 5) -> QueryResult:
        intent = self._infer_intent(text)
        matched_entities = self._matched_entities(text, top_k=top_k)
        similar_entities = self._similar_entities(matched_entities, top_k=top_k)
        supporting_relations = self._supporting_relations(matched_entities, intent)
        explanation = self._build_explanation(matched_entities, similar_entities, supporting_relations)
        answer = self.compose_answer(matched_entities, supporting_relations, intent)
        return QueryResult(
            matched_entities=matched_entities,
            similar_entities=similar_entities[:top_k],
            supporting_relations=supporting_relations[:top_k],
            explanation=explanation,
            answer=answer,
            intent=intent,
        )

    def compose_answer(self, matched_entities: List[Entity], relations: List[Relation], intent: Optional[QueryIntent] = None) -> str:
        return self._build_answer(matched_entities, relations, intent or QueryIntent())

    def _matched_entities(self, text: str, top_k: int) -> List[Entity]:
        matched = self.graph.find_entities_in_text(text)
        if not matched:
            matched = self.graph.find_entity_by_name(text)
        return [self._entity_from_node(item) for item in matched[:top_k]]

    def _similar_entities(self, matched_entities: List[Entity], top_k: int) -> List[Entity]:
        similar_entities: list[Entity] = []
        seen: set[str] = set()
        for entity in matched_entities:
            for _, target, _, data in self.graph.graph.out_edges(entity.id, keys=True, data=True):
                if data.get("kind") != "similarity":
                    continue
                if target in seen:
                    continue
                node = self.graph.graph.nodes[target]
                if node.get("kind") != "entity":
                    continue
                seen.add(target)
                similar_entities.append(self._entity_from_node(dict(id=target, **node)))
                if len(similar_entities) >= top_k:
                    return similar_entities
        return similar_entities

    def _supporting_relations(self, entities: List[Entity], intent: QueryIntent) -> List[Relation]:
        relations: List[Relation] = []
        entity_ids = {entity.id for entity in entities}
        if not entity_ids:
            return relations

        for source, target, _, data in self.graph.graph.edges(keys=True, data=True):
            if data.get("kind") != "relation":
                continue
            relation = Relation(
                source_id=source,
                target_id=target,
                relation_type=data.get("relation_type", "related_to"),
                evidence=data.get("evidence", ""),
                source_document_id=data.get("source_document_id", ""),
                confidence=data.get("confidence", 0.5),
                metadata=data.get("metadata", {}),
            )
            if self._relation_matches_intent(relation, entity_ids, intent):
                relations.append(relation)
        return self._dedupe_relations(relations)

    def _relation_matches_intent(self, relation: Relation, entity_ids: set[str], intent: QueryIntent) -> bool:
        if relation.source_id not in entity_ids and relation.target_id not in entity_ids:
            return False

        if not intent.relation_type:
            return True

        normalized = relation.relation_type.lower()
        if intent.relation_type == "collaborate_with":
            return normalized in {"collaborate_with", "partner_with", "co_occurs_with"}
        if intent.relation_type == "acquire":
            return normalized == "acquire"
        if intent.relation_type == "found":
            return normalized == "found"
        if intent.relation_type == "work_at":
            return normalized == "work_at"
        if intent.relation_type == "use":
            return normalized == "use"
        return True

    def _build_answer(self, matched_entities: List[Entity], relations: List[Relation], intent: QueryIntent) -> str:
        if not matched_entities:
            return "No direct answer could be inferred from the graph."

        focus = matched_entities[0]
        focus_name = focus.name or "The graph"
        collaboration_names: list[str] = []
        related_names: list[str] = []
        direct_sentences: list[str] = []
        seen_names: set[str] = set()
        seen_sentences: set[str] = set()

        for relation in relations:
            source_name = self._entity_name(relation.source_id)
            target_name = self._entity_name(relation.target_id)
            if not source_name or not target_name:
                continue

            if relation.relation_type in {"collaborate_with", "partner_with"}:
                if relation.source_id == focus.id:
                    counterpart = target_name
                elif relation.target_id == focus.id:
                    counterpart = source_name
                else:
                    continue
                if counterpart not in seen_names:
                    seen_names.add(counterpart)
                    collaboration_names.append(counterpart)
                continue

            if relation.relation_type == "co_occurs_with":
                counterpart = target_name if relation.source_id == focus.id else source_name
                if counterpart and counterpart not in seen_names:
                    seen_names.add(counterpart)
                    related_names.append(counterpart)
                continue

            sentence = self._directional_sentence(relation, source_name, target_name, intent)
            if sentence and sentence not in seen_sentences:
                seen_sentences.add(sentence)
                direct_sentences.append(sentence)

        parts: list[str] = []
        parts.extend(direct_sentences)

        if collaboration_names:
            parts.append(f"{focus_name} collaborates with {self._join_names(collaboration_names)}.")
        if related_names:
            parts.append(f"{focus_name} is mentioned with {self._join_names(related_names)}.")

        if parts:
            return " ".join(parts)

        if intent.relation_type:
            return f"No {intent.relation_type.replace('_', ' ')} fact was found for {focus_name}."

        return f"No direct answer could be inferred for {focus_name}."

    def _directional_sentence(self, relation: Relation, source_name: str, target_name: str, intent: QueryIntent) -> str:
        if relation.relation_type == "acquire":
            if intent.answer_role == "target":
                return f"{target_name} was acquired by {source_name}."
            return f"{source_name} acquired {target_name}."
        if relation.relation_type == "found":
            if intent.answer_role == "target":
                return f"{target_name} was founded by {source_name}."
            return f"{source_name} founded {target_name}."
        if relation.relation_type == "work_at":
            return f"{source_name} works at {target_name}."
        if relation.relation_type == "use":
            return f"{source_name} used {target_name}."
        return ""

    def _build_explanation(self, matched: List[Entity], similar: List[Entity], relations: List[Relation]) -> str:
        parts = []
        if matched:
            parts.append(f"Matched {len(matched)} entity(s) in the graph.")
        if similar:
            parts.append(f"Found {len(similar)} similarity-linked entity(s).")
        if relations:
            parts.append(f"Identified {len(relations)} relation(s) supporting the query.")
        if not parts:
            parts.append("No strong graph matches were found.")
        return " ".join(parts)

    def _infer_intent(self, text: str) -> QueryIntent:
        lowered = text.lower()
        keywords = [word for word in re.findall(r"[a-z][a-z0-9_]+", lowered) if len(word) > 2]

        relation_type: Optional[str] = None
        answer_role = "source"

        if re.search(r"\b(acquired|acquire|acquires|aquired|purchase|purchased|buy|bought)\b", lowered):
            relation_type = "acquire"
            answer_role = "target" if re.search(r"\b(acquired|purchased|bought)\s+by\b", lowered) else "source"
        elif re.search(r"\b(partner(?:ed|s)?|collaborat(?:ed|es|ing)?)\b", lowered) or re.search(r"\bwork(?:s|ed|ing)?\s+with\b", lowered):
            relation_type = "collaborate_with"
            answer_role = "other"
        elif re.search(r"\b(found(?:ed|s)?|founded|founds?)\b", lowered):
            relation_type = "found"
            answer_role = "source" if "by" not in lowered else "target"
        elif re.search(r"\bwork(?:s|ed|ing)?\s+(?:at|for)\b", lowered):
            relation_type = "work_at"
            answer_role = "source"
        elif re.search(r"\buse(?:d|s|ing)?\b", lowered):
            relation_type = "use"
            answer_role = "source"

        return QueryIntent(
            relation_type=relation_type,
            answer_role=answer_role,
            focus_mode="entity",
            keywords=keywords,
        )

    def _entity_from_node(self, item: dict) -> Entity:
        return Entity(
            id=item["id"],
            name=item.get("name", ""),
            label=item.get("label", "Concept"),
            source_document_id=item.get("metadata", {}).get("documents", [""])[0] if item.get("metadata") else "",
            confidence=item.get("confidence", 0.5),
            aliases=item.get("aliases", []),
            metadata=item.get("metadata", {}),
        )

    def _entity_name(self, entity_id: str) -> str:
        if entity_id not in self.graph.graph.nodes:
            return ""
        node = self.graph.graph.nodes[entity_id]
        if node.get("kind") != "entity":
            return ""
        return str(node.get("name", ""))

    def _join_names(self, names: List[str]) -> str:
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{', '.join(names[:-1])} and {names[-1]}"

    def _dedupe_relations(self, relations: List[Relation]) -> List[Relation]:
        seen = set()
        deduped: List[Relation] = []
        for relation in relations:
            key = (relation.source_id, relation.target_id, relation.relation_type, relation.evidence)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(relation)
        return deduped
