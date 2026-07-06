from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Entity:
    id: str
    name: str
    label: str
    source_document_id: str
    confidence: float = 0.5
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Relation:
    source_id: str
    target_id: str
    relation_type: str
    evidence: str
    source_document_id: str
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryIntent:
    relation_type: Optional[str] = None
    answer_role: str = "source"
    focus_mode: str = "entity"
    keywords: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractionResult:
    document: Document
    entities: List[Entity]
    relations: List[Relation]


@dataclass(slots=True)
class QueryResult:
    matched_entities: List[Entity]
    similar_entities: List[Entity]
    supporting_relations: List[Relation]
    explanation: str
    answer: str = ""
    intent: Optional[QueryIntent] = None
