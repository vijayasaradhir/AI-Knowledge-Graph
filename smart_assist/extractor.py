from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .models import Document, Entity, ExtractionResult, Relation
from .utils import split_sentences, stable_id

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


RELATION_ALIASES = {
    "partner": "partner_with",
    "partner_with": "partner_with",
    "collaborate": "collaborate_with",
    "collaborate_with": "collaborate_with",
    "acquire": "acquire",
    "acquired": "acquire",
    "found": "found",
    "founded": "found",
    "use": "use",
    "used": "use",
    "work": "work_at",
    "work_at": "work_at",
    "works_at": "work_at",
    "cooccur": "co_occurs_with",
    "co_occurs_with": "co_occurs_with",
}


@dataclass(slots=True)
class _HeuristicCandidate:
    text: str
    label: str
    source: str


class EntityRelationExtractor:
    def __init__(
        self,
        model_name: Optional[str] = None,
        client: Optional["OpenAI"] = None,
    ) -> None:
        self._entity_index: Dict[Tuple[str, str], Entity] = {}
        self._model_name = model_name or os.getenv("KG_OPENAI_MODEL", "gpt-4.1-mini")
        self._client = client or self._build_client()

    def extract(self, document: Document) -> ExtractionResult:
        self._entity_index = {}
        entities: List[Entity] = []
        relations: List[Relation] = []
        sentence_entities: Dict[str, List[Entity]] = defaultdict(list)

        payload = self._extract_with_llm(document.text)
        if payload is None:
            return self._extract_with_fallback(document)

        extracted_entities = payload.get("entities", [])
        extracted_relations = payload.get("relations", [])
        entities = self._materialize_llm_entities(document, extracted_entities)
        sentence_entities[document.text] = list(entities)
        relations = self._materialize_llm_relations(document, extracted_relations, entities)

        if not relations:
            relations.extend(self._cooccurrence_relations(document, sentence_entities))

        return ExtractionResult(document=document, entities=entities, relations=relations)

    def _build_client(self) -> Optional["OpenAI"]:
        if OpenAI is None:
            return None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            return OpenAI(api_key=api_key)
        except Exception:
            return None

    def _extract_with_llm(self, text: str) -> Optional[dict]:
        print(f"self._client {self._client}")
        if self._client is None:
            return None

        schema = {
            "name": "smart_assist_kg_extraction",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "label": {
                                    "type": "string",
                                    "enum": ["Person", "Organization", "Technology", "Location", "Event", "Concept"],
                                },
                                "confidence": {"type": "number"},
                                "aliases": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["name", "label", "confidence", "aliases"],
                        },
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "source_name": {"type": "string"},
                                "target_name": {"type": "string"},
                                "relation_type": {
                                    "type": "string",
                                    "enum": [
                                        "partner_with",
                                        "collaborate_with",
                                        "acquire",
                                        "found",
                                        "use",
                                        "work_at",
                                        "co_occurs_with",
                                    ],
                                },
                                "evidence": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": [
                                "source_name",
                                "target_name",
                                "relation_type",
                                "evidence",
                                "confidence",
                            ],
                        },
                    },
                },
                "required": ["entities", "relations"],
            },
            "strict": True,
        }

        system_prompt = (
            "You extract a knowledge graph from business/technology text.\n"
            "Identify distinct entities and the relations between them.\n"
            "Use concise canonical names, and avoid inventing facts not present in the text.\n"
            "Only return data that matches the provided JSON schema."
        )
        user_prompt = (
            "Extract entities and relations from the text below.\n\n"
            f"Text:\n{text}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_schema", "json_schema": schema},
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as exc:
            print(f"error working with llm {exc}")
            return None

    def _extract_with_fallback(self, document: Document) -> ExtractionResult:
        entities: List[Entity] = []
        relations: List[Relation] = []
        sentence_entities: Dict[str, List[Entity]] = defaultdict(list)

        for sentence in split_sentences(document.text):
            candidates = self._heuristic_candidates(sentence)
            extracted = self._materialize_heuristic_entities(document, candidates)
            for entity in extracted:
                if entity.id not in {item.id for item in entities}:
                    entities.append(entity)
                sentence_entities[sentence].append(entity)
            relations.extend(self._extract_heuristic_relations(document, sentence, extracted))

        if not relations:
            relations.extend(self._cooccurrence_relations(document, sentence_entities))

        return ExtractionResult(document=document, entities=entities, relations=relations)

    def _materialize_llm_entities(self, document: Document, extracted_entities: Sequence[dict]) -> List[Entity]:
        entities: List[Entity] = []
        for item in extracted_entities:
            name = self._clean_text(str(item.get("name", "")))
            if not name:
                continue
            label = self._normalize_label(str(item.get("label", "Concept")))
            key = (label, name.lower())
            confidence = self._clamp_confidence(item.get("confidence", 0.8), default=0.8)
            aliases = self._unique_texts([name, *(item.get("aliases", []) or [])])
            if key in self._entity_index:
                entity = self._entity_index[key]
                docs = list(entity.metadata.get("documents", []))
                if document.id not in docs:
                    docs.append(document.id)
                    entity.metadata["documents"] = docs
                entity.confidence = max(entity.confidence, confidence)
                entity.aliases = self._unique_texts([*entity.aliases, *aliases])
            else:
                entity = Entity(
                    id=stable_id("ent", f"{label}:{name.lower()}"),
                    name=name,
                    label=label,
                    source_document_id=document.id,
                    confidence=confidence,
                    aliases=aliases,
                    metadata={
                        "documents": [document.id],
                        "source": "openai",
                    },
                )
                self._entity_index[key] = entity
            entities.append(entity)
        return entities

    def _materialize_llm_relations(self, document: Document, extracted_relations: Sequence[dict], entities: Sequence[Entity]) -> List[Relation]:
        relations: List[Relation] = []
        lookup = self._build_entity_lookup(entities)
        for item in extracted_relations:
            src_name = self._clean_text(str(item.get("source_name", "")))
            tgt_name = self._clean_text(str(item.get("target_name", "")))
            relation_type = self._normalize_relation_type(str(item.get("relation_type", "")))
            evidence = self._clean_text(str(item.get("evidence", ""))) or document.text
            if not src_name or not tgt_name or not relation_type:
                continue
            source_entity = self._resolve_entity(src_name, lookup)
            target_entity = self._resolve_entity(tgt_name, lookup)
            if not source_entity or not target_entity or source_entity.id == target_entity.id:
                continue
            relations.append(
                Relation(
                    source_id=source_entity.id,
                    target_id=target_entity.id,
                    relation_type=relation_type,
                    evidence=evidence,
                    source_document_id=document.id,
                    confidence=self._clamp_confidence(item.get("confidence", 0.8), default=0.8),
                    metadata={"source": "openai"},
                )
            )
        return self._dedupe_relations(relations)

    def _heuristic_candidates(self, sentence: str) -> List[_HeuristicCandidate]:
        candidates: List[_HeuristicCandidate] = []
        pattern = re.compile(r"\b(?:[A-Z][\w&.-]*)(?:\s+[A-Z][\w&.-]*)*\b")
        for match in pattern.finditer(sentence):
            text = self._clean_text(match.group(0))
            if not text or all(token in {"The", "A", "An"} for token in text.split()):
                continue
            candidates.append(_HeuristicCandidate(text=text, label=self._infer_label(text), source="regex"))
        return self._dedupe_heuristic_candidates(candidates)

    def _materialize_heuristic_entities(self, document: Document, candidates: Sequence[_HeuristicCandidate]) -> List[Entity]:
        entities: List[Entity] = []
        for candidate in candidates:
            key = (candidate.label, candidate.text.lower())
            if key in self._entity_index:
                entity = self._entity_index[key]
                docs = list(entity.metadata.get("documents", []))
                if document.id not in docs:
                    docs.append(document.id)
                    entity.metadata["documents"] = docs
            else:
                entity = Entity(
                    id=stable_id("ent", f"{candidate.label}:{candidate.text.lower()}"),
                    name=candidate.text,
                    label=candidate.label,
                    source_document_id=document.id,
                    confidence=0.6,
                    aliases=[candidate.text.lower()],
                    metadata={"documents": [document.id], "source": candidate.source},
                )
                self._entity_index[key] = entity
            entities.append(entity)
        return entities

    def _extract_heuristic_relations(self, document: Document, sentence: str, entities: Sequence[Entity]) -> List[Relation]:
        relations: List[Relation] = []
        entity_lookup = self._build_entity_lookup(entities)
        relation_patterns: Sequence[Tuple[str, re.Pattern[str]]] = (
            ("partner_with", re.compile(r"(?P<src>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)\s+partner(?:ed|s)?\s+with\s+(?P<tgt>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)", re.I)),
            ("collaborate_with", re.compile(r"(?P<src>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)\s+collaborat(?:ed|es|ing)?\s+with\s+(?P<tgt>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)", re.I)),
            ("acquire", re.compile(r"(?P<src>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)\s+acquir(?:ed|es|ing)\s+(?P<tgt>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)", re.I)),
            ("found", re.compile(r"(?P<src>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)\s+found(?:ed|s)?\s+(?P<tgt>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)", re.I)),
            ("use", re.compile(r"(?P<src>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)\s+use(?:d|s|ing)?\s+(?P<tgt>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)", re.I)),
            ("work_at", re.compile(r"(?P<src>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)\s+(?:work(?:s|ed|ing)?|engineer(?:s)?|employee(?:s)?)(?:\s+at|\s+for)\s+(?P<tgt>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)", re.I)),
        )
        for relation_type, pattern in relation_patterns:
            for match in pattern.finditer(sentence):
                source_entity = self._resolve_entity(match.group("src"), entity_lookup)
                target_entity = self._resolve_entity(match.group("tgt"), entity_lookup)
                if not source_entity or not target_entity or source_entity.id == target_entity.id:
                    continue
                relations.append(
                    Relation(
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        relation_type=relation_type,
                        evidence=sentence,
                        source_document_id=document.id,
                        confidence=0.75,
                        metadata={"source": "heuristic"},
                    )
                )
        return self._dedupe_relations(relations)

    def _cooccurrence_relations(self, document: Document, sentence_entities: Dict[str, List[Entity]]) -> List[Relation]:
        relations: List[Relation] = []
        for sentence, ents in sentence_entities.items():
            unique = list({entity.id: entity for entity in ents}.values())
            for i, src in enumerate(unique):
                for tgt in unique[i + 1 :]:
                    relations.append(
                        Relation(
                            source_id=src.id,
                            target_id=tgt.id,
                            relation_type="co_occurs_with",
                            evidence=sentence,
                            source_document_id=document.id,
                            confidence=0.4,
                            metadata={"source": "cooccurrence"},
                        )
                    )
        return self._dedupe_relations(relations)

    def _build_entity_lookup(self, entities: Sequence[Entity]) -> Dict[str, Entity]:
        lookup: Dict[str, Entity] = {}
        for entity in entities:
            lookup[entity.name.lower()] = entity
            for alias in entity.aliases:
                lookup[alias.lower()] = entity
        return lookup

    def _resolve_entity(self, candidate: str, lookup: Dict[str, Entity]) -> Optional[Entity]:
        candidate = self._clean_text(candidate)
        if not candidate:
            return None
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
        for key, entity in lookup.items():
            if candidate.lower() in key or key in candidate.lower():
                return entity
        return None

    def _normalize_relation_type(self, relation_type: str) -> str:
        normalized = relation_type.strip().lower().replace(" ", "_")
        return RELATION_ALIASES.get(normalized, normalized or "co_occurs_with")

    def _normalize_label(self, label: str) -> str:
        normalized = label.strip().title()
        allowed = {"Person", "Organization", "Technology", "Location", "Event", "Concept"}
        return normalized if normalized in allowed else "Concept"

    def _infer_label(self, candidate: str) -> str:
        lowered = candidate.lower()
        if lowered in {"python", "networkx", "neo4j", "azure", "github", "copilot"}:
            return "Technology"
        if any(marker in lowered for marker in ("inc", "corp", "llc", "labs", "systems", "technologies", "platform", "network", "graph", "university", "openai", "microsoft", "contoso")):
            return "Organization"
        if " " in candidate and candidate.split()[0] in {"Sam", "Alex", "Jordan", "Taylor", "Chris", "Morgan", "Riley"}:
            return "Person"
        if candidate and candidate[0].isupper():
            return "Concept"
        return "Concept"

    def _dedupe_heuristic_candidates(self, candidates: Sequence[_HeuristicCandidate]) -> List[_HeuristicCandidate]:
        seen = set()
        deduped: List[_HeuristicCandidate] = []
        for candidate in candidates:
            key = candidate.text.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _dedupe_relations(self, relations: Sequence[Relation]) -> List[Relation]:
        seen = set()
        deduped: List[Relation] = []
        for relation in relations:
            key = (relation.source_id, relation.target_id, relation.relation_type, relation.evidence)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(relation)
        return deduped

    def _unique_texts(self, texts: Sequence[str]) -> List[str]:
        seen = set()
        unique: List[str] = []
        for text in texts:
            cleaned = self._clean_text(text)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique.append(cleaned)
        return unique

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip(" \t\r\n,;:.")

    def _clamp_confidence(self, value, default: float = 0.7) -> float:
        try:
            numeric = float(value)
        except Exception:
            numeric = default
        return max(0.0, min(1.0, numeric))
