from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from .graph_store import KnowledgeGraph
from .models import Document, Entity, ExtractionResult, Relation
from .utils import stable_id

try:  # pragma: no cover - optional dependency
    import javalang  # type: ignore
except Exception:  # pragma: no cover
    javalang = None


_CODE_EXTENSIONS = {
    ".py": "python",
    ".java": "java",
}


@dataclass(slots=True)
class _CodeSymbol:
    name: str
    label: str
    qualified_name: str
    source_document_id: str
    metadata: dict


@dataclass(slots=True)
class JavaSymbolRecord:
    name: str
    label: str
    qualified_name: str
    source_document_id: str
    metadata: dict = field(default_factory=dict)


class JavaProjectIndex:
    def __init__(self) -> None:
        self.by_qualified_name: dict[str, JavaSymbolRecord] = {}
        self.by_simple_name: dict[str, list[JavaSymbolRecord]] = {}

    def add(self, record: JavaSymbolRecord) -> None:
        qualified_key = record.qualified_name.lower()
        simple_key = record.name.lower()
        self.by_qualified_name[qualified_key] = record
        self.by_simple_name.setdefault(simple_key, []).append(record)

    def resolve(
        self,
        name: str,
        qualified_name: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Optional[JavaSymbolRecord]:
        candidates: list[JavaSymbolRecord] = []
        if qualified_name:
            record = self.by_qualified_name.get(qualified_name.lower())
            if record is not None:
                return record

        direct = self.by_simple_name.get(name.lower(), [])
        if direct:
            candidates.extend(direct)

        if "." in name:
            tail = name.split(".")[-1].lower()
            candidates.extend(self.by_simple_name.get(tail, []))

        if label:
            filtered = [record for record in candidates if record.label == label]
            if len(filtered) == 1:
                return filtered[0]
            if filtered:
                candidates = filtered

        unique_by_qname: dict[str, JavaSymbolRecord] = {}
        for record in candidates:
            unique_by_qname[record.qualified_name.lower()] = record
        if len(unique_by_qname) == 1:
            return next(iter(unique_by_qname.values()))
        if candidates:
            return candidates[0]
        return None


def _java_annotation_name(annotation) -> str:
    name = getattr(annotation, "name", None)
    if name is not None:
        return str(name)
    return str(annotation)


def _java_annotation_member_value(annotation, member_name: str) -> Optional[str]:
    def _clean(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            return text[1:-1]
        return text

    element = getattr(annotation, "element", None)
    if element is None:
        return None

    if hasattr(element, "pairs"):
        for pair in getattr(element, "pairs", []):
            if getattr(pair, "name", None) == member_name:
                value = getattr(pair, "value", None)
                if hasattr(value, "value"):
                    return _clean(str(getattr(value, "value")))
                return _clean(str(value)) if value is not None else None
        return None

    if hasattr(element, "value"):
        value = getattr(element, "value", None)
        if hasattr(value, "value"):
            return _clean(str(getattr(value, "value")))
        return _clean(str(value)) if value is not None else None

    return None


def _java_path_join(base_path: Optional[str], path: Optional[str]) -> str:
    def _normalize(segment: Optional[str]) -> str:
        text = str(segment or "").strip()
        if not text:
            return ""
        if not text.startswith("/"):
            text = "/" + text
        if text != "/":
            text = text.rstrip("/")
        return text

    base = _normalize(base_path)
    suffix = _normalize(path)
    if not base:
        return suffix
    if not suffix:
        return base
    if base == "/":
        return suffix or "/"
    if suffix == "/":
        return base
    return f"{base.rstrip('/')}/{suffix.lstrip('/')}"


def _java_spring_annotation_metadata(annotations: Sequence[object], base_path: Optional[str] = None) -> dict:
    names = [_java_annotation_name(annotation) for annotation in annotations]
    metadata = {
        "annotations": names,
    }
    spring: dict = {}
    stereotypes = {
        "RestController": "RestController",
        "Controller": "Controller",
    }
    stereotype = next((stereotypes[name.split(".")[-1]] for name in names if name.split(".")[-1] in stereotypes), "")
    if stereotype:
        spring["stereotype"] = stereotype
        spring["is_controller"] = True

    mapping_annotation = next(
        (name for name in names if name.endswith(("RequestMapping", "GetMapping", "PostMapping", "PutMapping", "DeleteMapping", "PatchMapping"))),
        "",
    )
    if mapping_annotation:
        http_method = {
            "GetMapping": "GET",
            "PostMapping": "POST",
            "PutMapping": "PUT",
            "DeleteMapping": "DELETE",
            "PatchMapping": "PATCH",
            "RequestMapping": "REQUEST",
        }
        annotation_name = mapping_annotation.split(".")[-1]
        annotation_object = next(annotation for annotation in annotations if _java_annotation_name(annotation).endswith(annotation_name))
        raw_path = _java_annotation_member_value(annotation_object, "value")
        endpoint = {
            "annotation": annotation_name,
            "http_method": http_method.get(annotation_name, "REQUEST"),
            "path": raw_path,
        }
        full_path = _java_path_join(base_path, raw_path)
        if full_path:
            endpoint["full_path"] = full_path
            spring["base_path"] = base_path or ""
        spring["endpoint"] = endpoint

    if spring:
        metadata["spring"] = spring
        if "endpoint" in spring:
            metadata["endpoint"] = spring["endpoint"]
    return metadata


class CodeGraphIngestor:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def load_documents(self, source: object) -> List[Document]:
        if not self.enabled:
            raise RuntimeError("Code graph ingestion is disabled.")

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                return self._load_documents_from_directory(path)
            return [self._load_document_from_file(path)]

        if isinstance(source, dict):
            source = [source]

        documents: List[Document] = []
        for index, item in enumerate(source if isinstance(source, Iterable) else []):
            if isinstance(item, Document):
                documents.append(item)
                continue
            if isinstance(item, dict):
                path_value = item.get("path") or item.get("source_file") or item.get("file")
                if not path_value:
                    raise ValueError("Code document records must include a 'path' field.")
                documents.append(self._load_document_from_file(Path(str(path_value))))
                continue
            documents.append(self._load_document_from_file(Path(str(item))))
        return documents

    def should_handle(self, document: Document) -> bool:
        return self.enabled and self._language_for_document(document) in {"python", "java"}

    def ingest_document(self, document: Document, graph: KnowledgeGraph) -> ExtractionResult:
        return self.ingest_documents([document], graph)[0]

    def ingest_documents(self, documents: Sequence[Document], graph: KnowledgeGraph) -> List[ExtractionResult]:
        if not self.enabled:
            raise RuntimeError("Code graph ingestion is disabled.")

        results: List[Optional[ExtractionResult]] = [None] * len(documents)
        java_indices: list[int] = []
        java_documents: list[Document] = []

        for index, document in enumerate(documents):
            language = self._language_for_document(document)
            if language == "java":
                graph.add_document(document)
                java_indices.append(index)
                java_documents.append(document)
            elif language == "python":
                graph.add_document(document)
                results[index] = self._ingest_python(document, graph)
            else:
                results[index] = ExtractionResult(document=document, entities=[], relations=[])

        java_index = self._collect_java_project_index(java_documents) if java_documents else JavaProjectIndex()
        for index, document in zip(java_indices, java_documents):
            results[index] = self._ingest_java(document, graph, java_index)

        return [result or ExtractionResult(document=document, entities=[], relations=[]) for result, document in zip(results, documents)]

    def export_metadata(self, source: object, output_dir: Optional[Path] = None) -> Path:
        if not self.enabled:
            raise RuntimeError("Code graph ingestion is disabled.")

        documents = self.load_documents(source)
        temp_graph = KnowledgeGraph()
        results = self.ingest_documents(documents, temp_graph)
        payload = self._build_metadata_payload(results, source)

        output_root = Path(output_dir) if output_dir is not None else Path.cwd()
        output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_root / f"code_graph_metadata_{timestamp}.json"
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    def ingest_metadata(self, source: object, graph: KnowledgeGraph) -> List[ExtractionResult]:
        if not self.enabled:
            raise RuntimeError("Code graph ingestion is disabled.")

        payload = self._load_metadata_payload(source)
        documents_data = payload.get("documents", [])
        if not isinstance(documents_data, list):
            raise ValueError("Code graph metadata must contain a 'documents' array.")

        results: List[ExtractionResult] = []
        for index, item in enumerate(documents_data):
            if not isinstance(item, dict):
                raise ValueError("Each code graph metadata document must be an object.")
            results.append(self._ingest_metadata_document(item, graph, index=index))
        return results

    def _collect_java_project_index(self, documents: Sequence[Document]) -> JavaProjectIndex:
        index = JavaProjectIndex()
        for document in documents:
            for record in self._collect_java_symbols(document):
                index.add(record)
        return index

    def _build_metadata_payload(self, results: Sequence[ExtractionResult], source: object) -> dict:
        documents: List[dict] = []
        for result in results:
            document = result.document
            entities = self._metadata_entities_for_document(document, result)
            entity_ids = {entity["id"] for entity in entities}
            relations = [
                self._serialize_metadata_relation(relation)
                for relation in sorted(
                    (
                        relation
                        for relation in result.relations
                        if self._should_export_relation(relation, entity_ids)
                    ),
                    key=lambda relation: (
                        relation.relation_type.lower(),
                        relation.source_id,
                        relation.target_id,
                        relation.evidence.lower(),
                    ),
                )
            ]
            documents.append(
                {
                    "id": document.id,
                    "source_file": str(document.metadata.get("source_file", "")),
                    "language": str(document.metadata.get("language", "")),
                    "module_name": str(document.metadata.get("module_name", "")),
                    "entities": entities,
                    "relations": relations,
                }
            )

        return {
            "kind": "code_graph_metadata",
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "source": str(source),
            "documents": documents,
        }

    def _metadata_entities_for_document(self, document: Document, result: ExtractionResult) -> List[dict]:
        if self._language_for_document(document) != "java":
            return [
                self._serialize_metadata_entity(entity)
                for entity in sorted(
                    (entity for entity in result.entities if self._should_export_entity(entity)),
                    key=lambda entity: (entity.label.lower(), entity.metadata.get("qualified_name", entity.name).lower()),
                )
            ]

        records = self._collect_java_symbols(document)
        entities: List[dict] = []
        for record in records:
            if record.label not in {"Module", "Class", "Method"}:
                continue
            entity = self._entity_from_java_record(record, KnowledgeGraph(), document, [])
            entities.append(self._serialize_metadata_entity(entity))

        entities.sort(key=lambda entity: (entity["label"].lower(), str(entity.get("qualified_name", entity["name"])).lower()))
        return entities

    def _load_metadata_payload(self, source: object) -> dict:
        if isinstance(source, (str, Path)):
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        elif isinstance(source, dict):
            payload = dict(source)
        else:
            payload = {"documents": list(source) if isinstance(source, Sequence) else []}

        if isinstance(payload, list):
            payload = {"documents": payload}
        return payload

    def _ingest_metadata_document(self, item: dict, graph: KnowledgeGraph, index: int) -> ExtractionResult:
        document_id = str(item.get("id") or item.get("source_file") or f"code-metadata-{index + 1:03d}")
        document = Document(
            id=document_id,
            text="",
            metadata={
                "kind": "code",
                "source_file": str(item.get("source_file", "")),
                "language": str(item.get("language", "")),
                "module_name": str(item.get("module_name", "")),
                "metadata_source": "code_graph_metadata",
            },
        )
        graph.add_document(document)

        entities: List[Entity] = []
        entity_lookup: dict[str, Entity] = {}
        for entity_data in item.get("entities", []):
            if not isinstance(entity_data, dict):
                continue
            entity = self._entity_from_metadata(entity_data, document)
            if entity.id not in entity_lookup:
                entity_lookup[entity.id] = entity
                entities.append(entity)
            graph.add_entity(entity)
            graph.add_mention(document.id, entity.id)

        relations: List[Relation] = []
        for relation_data in item.get("relations", []):
            if not isinstance(relation_data, dict):
                continue
            relation = self._relation_from_metadata(relation_data, document)
            if relation.source_id not in entity_lookup or relation.target_id not in entity_lookup:
                continue
            graph.add_relation(relation)
            relations.append(relation)

        return ExtractionResult(document=document, entities=entities, relations=relations)

    def _serialize_metadata_entity(self, entity: Entity) -> dict:
        qualified_name = str(entity.metadata.get("qualified_name", entity.name))
        metadata = self._export_entity_metadata(entity)
        return {
            "id": entity.id,
            "name": entity.name,
            "label": entity.label,
            "qualified_name": qualified_name,
            "source_document_id": entity.source_document_id,
            **({"metadata": metadata} if metadata else {}),
        }

    def _serialize_metadata_relation(self, relation: Relation) -> dict:
        return {
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "relation_type": relation.relation_type,
            "evidence": relation.evidence,
            "source_document_id": relation.source_document_id,
        }

    def _should_export_entity(self, entity: Entity) -> bool:
        return entity.label in {"Module", "Class", "Method", "Function"}

    def _should_export_relation(self, relation: Relation, entity_ids: set[str]) -> bool:
        return relation.relation_type in {"contains", "calls"} and relation.source_id in entity_ids and relation.target_id in entity_ids

    def _export_entity_metadata(self, entity: Entity) -> dict:
        allowed_keys = {"annotations", "endpoint", "language", "node_type", "spring"}
        metadata = {
            key: value
            for key, value in entity.metadata.items()
            if key in allowed_keys and value not in (None, [], {}, "")
        }
        return metadata

    def _entity_from_metadata(self, entity_data: dict, document: Document) -> Entity:
        label = str(entity_data.get("label", "Concept"))
        qualified_name = str(entity_data.get("qualified_name", entity_data.get("name", "")))
        entity_id = str(entity_data.get("id") or stable_id("code", f"{label}:{qualified_name}"))
        aliases = entity_data.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        metadata = {
            "documents": [document.id],
            "qualified_name": qualified_name,
            "metadata_source": "code_graph_metadata",
        }
        exported_metadata = entity_data.get("metadata", {})
        if isinstance(exported_metadata, dict):
            metadata.update(exported_metadata)
        return Entity(
            id=entity_id,
            name=str(entity_data.get("name", "")),
            label=label,
            source_document_id=str(entity_data.get("source_document_id", document.id)),
            confidence=float(entity_data.get("confidence", 0.9) or 0.9),
            aliases=[str(alias) for alias in aliases if alias is not None],
            metadata=metadata,
        )

    def _relation_from_metadata(self, relation_data: dict, document: Document) -> Relation:
        return Relation(
            source_id=str(relation_data.get("source_id", "")),
            target_id=str(relation_data.get("target_id", "")),
            relation_type=str(relation_data.get("relation_type", "contains")),
            evidence=str(relation_data.get("evidence", "")),
            source_document_id=str(relation_data.get("source_document_id", document.id)),
            confidence=float(relation_data.get("confidence", 0.9) or 0.9),
            metadata={"metadata_source": "code_graph_metadata"},
        )

    def _load_documents_from_directory(self, directory: Path) -> List[Document]:
        files = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in _CODE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No supported source code files found in directory: {directory}")
        documents: List[Document] = []
        for path in files:
            root = self._source_root_for_path(path, fallback=directory)
            documents.append(self._load_document_from_file(path, root=root))
        return documents

    def _load_document_from_file(self, path: Path, root: Optional[Path] = None) -> Document:
        suffix = path.suffix.lower()
        if suffix not in _CODE_EXTENSIONS:
            raise ValueError(f"Unsupported source code format: {path.suffix}")

        text = path.read_text(encoding="utf-8")
        language = _CODE_EXTENSIONS[suffix]
        return Document(
            id=self._document_id_from_path(path, root=root),
            text=text,
            metadata={
                "source_file": str(path),
                "format": path.suffix.lower().lstrip("."),
                "kind": "code",
                "language": language,
                "module_name": self._module_name(path, root=root),
            },
        )

    def _ingest_python(self, document: Document, graph: KnowledgeGraph) -> ExtractionResult:
        try:
            tree = ast.parse(document.text)
        except SyntaxError:
            return ExtractionResult(document=document, entities=[], relations=[])

        module_name = str(document.metadata.get("module_name") or self._module_name(Path(document.metadata.get("source_file", document.id))))
        module_symbol = self._symbol(
            name=module_name.split(".")[-1] or document.id,
            label="Module",
            qualified_name=module_name,
            document=document,
        )

        entities: List[Entity] = []
        relations: List[Relation] = []
        symbol_lookup: dict[str, Entity] = {}

        module_entity = self._register_entity(graph, document, module_symbol, entities)
        symbol_lookup[module_symbol.name.lower()] = module_entity
        symbol_lookup[module_symbol.qualified_name.lower()] = module_entity

        class_stack: list[Entity] = []
        function_stack: list[Entity] = []

        def current_parent() -> Entity:
            if function_stack:
                return function_stack[-1]
            if class_stack:
                return class_stack[-1]
            return module_entity

        def add_symbol(symbol: _CodeSymbol) -> Entity:
            entity = self._register_entity(graph, document, symbol, entities)
            symbol_lookup[symbol.name.lower()] = entity
            symbol_lookup[symbol.qualified_name.lower()] = entity
            return entity

        def add_relation(source: Entity, target: Entity, relation_type: str, evidence: str, metadata: Optional[dict] = None) -> None:
            relation = Relation(
                source_id=source.id,
                target_id=target.id,
                relation_type=relation_type,
                evidence=evidence,
                source_document_id=document.id,
                confidence=0.9,
                metadata=metadata or {"source": "code"},
            )
            graph.add_relation(relation)
            relations.append(relation)

        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
                qualified_name = ".".join([module_name, *(item.name for item in class_stack), node.name])
                class_entity = add_symbol(
                    _CodeSymbol(
                        name=node.name,
                        label="Class",
                        qualified_name=qualified_name,
                        source_document_id=document.id,
                        metadata={"kind": "code", "language": "python", "node_type": "class"},
                    )
                )
                add_relation(current_parent(), class_entity, "contains", node.name)
                for base in node.bases:
                    base_name = self._python_name(base)
                    if not base_name:
                        continue
                    base_entity = self._external_symbol(base_name, "Class", document)
                    add_relation(class_entity, base_entity, "inherits", base_name, {"source": "code", "kind": "inheritance"})

                class_stack.append(class_entity)
                self.generic_visit(node)
                class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
                self._visit_function(node)

            def _visit_function(self, node) -> None:
                label = "Method" if class_stack else "Function"
                qualified_name = ".".join([module_name, *(item.name for item in class_stack), node.name])
                function_entity = add_symbol(
                    _CodeSymbol(
                        name=node.name,
                        label=label,
                        qualified_name=qualified_name,
                        source_document_id=document.id,
                        metadata={"kind": "code", "language": "python", "node_type": label.lower()},
                    )
                )
                add_relation(current_parent(), function_entity, "contains", node.name)
                function_stack.append(function_entity)
                self.generic_visit(node)
                function_stack.pop()

            def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
                parent = current_parent()
                for alias in node.names:
                    imported_name = alias.asname or alias.name
                    imported_entity = self._external_symbol(imported_name, "Module", document)
                    add_relation(parent, imported_entity, "imports", alias.name, {"source": "code"})

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
                parent = current_parent()
                module_prefix = node.module or ""
                for alias in node.names:
                    imported_name = alias.asname or alias.name
                    qualified = f"{module_prefix}.{alias.name}" if module_prefix else alias.name
                    imported_entity = self._external_symbol(imported_name, "Symbol", document, qualified_name=qualified)
                    add_relation(parent, imported_entity, "imports", qualified, {"source": "code"})

            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                caller = current_parent()
                target_name = self._python_name(node.func)
                if target_name:
                    target_entity = self._resolve_local_or_external(target_name, document)
                    add_relation(caller, target_entity, "calls", target_name, {"source": "code"})
                self.generic_visit(node)

            def _python_name(self, node) -> str:
                if isinstance(node, ast.Name):
                    return node.id
                if isinstance(node, ast.Attribute):
                    parts: list[str] = []
                    current = node
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                    parts.reverse()
                    return ".".join(parts)
                return ""

            def _resolve_local_or_external(self, name: str, document: Document) -> Entity:
                lowered = name.lower()
                if lowered in symbol_lookup:
                    return symbol_lookup[lowered]
                if "." in lowered and lowered.split(".")[-1] in symbol_lookup:
                    return symbol_lookup[lowered.split(".")[-1]]
                return self._external_symbol(name, "Function", document)

            def _external_symbol(self, name: str, label: str, document: Document, qualified_name: Optional[str] = None) -> Entity:
                qualified = qualified_name or name
                symbol = _CodeSymbol(
                    name=name,
                    label=label,
                    qualified_name=qualified,
                    source_document_id=document.id,
                    metadata={"kind": "code", "language": "python", "external": True},
                )
                return add_symbol(symbol)

        visitor = Visitor()
        visitor.visit(tree)

        return ExtractionResult(document=document, entities=entities, relations=relations)

    def _collect_java_symbols(self, document: Document) -> List[JavaSymbolRecord]:
        if javalang is None:
            return self._collect_java_symbols_with_regex(document)

        try:
            tree = javalang.parse.parse(document.text)
        except Exception:
            return self._collect_java_symbols_with_regex(document)

        return self._collect_java_symbols_from_tree(document, tree)

    def _collect_java_symbols_from_tree(self, document: Document, tree) -> List[JavaSymbolRecord]:
        module_name = str(document.metadata.get("module_name") or self._module_name(Path(document.metadata.get("source_file", document.id))))
        records: List[JavaSymbolRecord] = [
            JavaSymbolRecord(
                name=module_name.split(".")[-1] or document.id,
                label="Module",
                qualified_name=module_name,
                source_document_id=document.id,
                metadata={"kind": "code", "language": "java", "node_type": "module"},
            )
        ]

        def walk_type(type_decl, parent_name: str, inherited_base_path: str = "") -> None:
            type_name = getattr(type_decl, "name", "")
            qualified_name = ".".join(part for part in [parent_name, type_name] if part)
            class_annotations = _java_spring_annotation_metadata(getattr(type_decl, "annotations", []) or [], base_path=inherited_base_path)
            class_base_path = str(class_annotations.get("spring", {}).get("base_path", inherited_base_path) or inherited_base_path)
            records.append(
                JavaSymbolRecord(
                    name=type_name,
                    label="Class",
                    qualified_name=qualified_name,
                    source_document_id=document.id,
                    metadata={"kind": "code", "language": "java", "node_type": "class", **class_annotations},
                )
            )

            for item in getattr(type_decl, "body", []) or []:
                item_name = type(item).__name__
                if item_name in {"ClassDeclaration", "InterfaceDeclaration", "EnumDeclaration"}:
                    walk_type(item, qualified_name, class_base_path)
                    continue

                if item_name in {"MethodDeclaration", "ConstructorDeclaration"}:
                    method_name = getattr(item, "name", "")
                    method_annotations = _java_spring_annotation_metadata(
                        getattr(item, "annotations", []) or [],
                        base_path=class_base_path,
                    )
                    records.append(
                        JavaSymbolRecord(
                            name=method_name,
                            label="Method",
                            qualified_name=".".join(part for part in [qualified_name, method_name] if part),
                            source_document_id=document.id,
                            metadata={"kind": "code", "language": "java", "node_type": "method", **method_annotations},
                        )
                    )

        for type_decl in getattr(tree, "types", []):
            walk_type(type_decl, module_name)
        return records

    def _collect_java_symbols_with_regex(self, document: Document) -> List[JavaSymbolRecord]:
        module_name = str(document.metadata.get("module_name") or self._module_name(Path(document.metadata.get("source_file", document.id))))
        records: List[JavaSymbolRecord] = [
            JavaSymbolRecord(
                name=module_name.split(".")[-1] or document.id,
                label="Module",
                qualified_name=module_name,
                source_document_id=document.id,
                metadata={"kind": "code", "language": "java", "node_type": "module"},
            )
        ]

        class_pattern = re.compile(r"\b(?:public\s+)?(?:final\s+)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_][\w]*)")
        interface_pattern = re.compile(r"\binterface\s+(?P<name>[A-Za-z_][\w]*)")
        method_pattern = re.compile(
            r"\b(?:public|private|protected|static|final|synchronized|abstract|native|strictfp|\s)+"
            r"[\w<>\[\]]+\s+(?P<name>[A-Za-z_][\w]*)\s*\("
        )

        for pattern, label in ((class_pattern, "Class"), (interface_pattern, "Interface")):
            for match in pattern.finditer(document.text):
                name = match.group("name")
                records.append(
                    JavaSymbolRecord(
                        name=name,
                        label="Class",
                        qualified_name=f"{module_name}.{name}",
                        source_document_id=document.id,
                        metadata={
                            "kind": "code",
                            "language": "java",
                            "node_type": "class",
                            "source": "regex",
                            "annotations": [],
                            "spring": {},
                        },
                    )
                )

        for match in method_pattern.finditer(document.text):
            name = match.group("name")
            records.append(
                    JavaSymbolRecord(
                        name=name,
                        label="Method",
                        qualified_name=f"{module_name}.{name}",
                        source_document_id=document.id,
                        metadata={
                            "kind": "code",
                            "language": "java",
                            "node_type": "method",
                            "source": "regex",
                            "annotations": [],
                            "spring": {},
                        },
                    )
                )

        return records

    def _ingest_java(self, document: Document, graph: KnowledgeGraph, project_index: Optional[JavaProjectIndex] = None) -> ExtractionResult:
        if javalang is None:
            return self._ingest_java_with_regex(document, graph, project_index)

        try:
            tree = javalang.parse.parse(document.text)
        except Exception:
            return self._ingest_java_with_regex(document, graph, project_index)

        module_name = str(document.metadata.get("module_name") or self._module_name(Path(document.metadata.get("source_file", document.id))))
        entities: List[Entity] = []
        relations: List[Relation] = []
        symbol_lookup: dict[str, Entity] = {}

        def add_symbol(symbol: _CodeSymbol) -> Entity:
            entity = self._register_entity(graph, document, symbol, entities)
            symbol_lookup[symbol.name.lower()] = entity
            symbol_lookup[symbol.qualified_name.lower()] = entity
            return entity

        def add_relation(source: Entity, target: Entity, relation_type: str, evidence: str, metadata: Optional[dict] = None) -> None:
            relation = Relation(
                source_id=source.id,
                target_id=target.id,
                relation_type=relation_type,
                evidence=evidence,
                source_document_id=document.id,
                confidence=0.9,
                metadata=metadata or {"source": "code"},
            )
            graph.add_relation(relation)
            relations.append(relation)

        for type_decl in getattr(tree, "types", []):
            self._ingest_java_type(
                type_decl=type_decl,
                document=document,
                graph=graph,
                entities=entities,
                relations=relations,
                symbol_lookup=symbol_lookup,
                add_symbol=add_symbol,
                add_relation=add_relation,
                parent_entity=None,
                package_name=module_name,
                project_index=project_index,
            )

        return ExtractionResult(document=document, entities=entities, relations=relations)

    def _ingest_java_type(
        self,
        type_decl,
        document: Document,
        graph: KnowledgeGraph,
        entities: List[Entity],
        relations: List[Relation],
        symbol_lookup: dict[str, Entity],
        add_symbol,
        add_relation,
        parent_entity: Optional[Entity],
        package_name: str,
        project_index: Optional[JavaProjectIndex],
    ) -> None:
        label = "Class"

        qualified_name = ".".join(part for part in [package_name, getattr(type_decl, "name", "")] if part)
        type_entity = add_symbol(
            _CodeSymbol(
                name=getattr(type_decl, "name", ""),
                label=label,
                qualified_name=qualified_name or getattr(type_decl, "name", ""),
                source_document_id=document.id,
                metadata={"kind": "code", "language": "java", "node_type": "class"},
            )
        )
        if parent_entity is not None:
            add_relation(parent_entity, type_entity, "contains", getattr(type_decl, "name", ""))

        body = getattr(type_decl, "body", []) or []
        for item in body:
            item_name = type(item).__name__
            if item_name in {"ClassDeclaration", "InterfaceDeclaration", "EnumDeclaration"}:
                self._ingest_java_type(
                    type_decl=item,
                    document=document,
                    graph=graph,
                    entities=entities,
                    relations=relations,
                    symbol_lookup=symbol_lookup,
                    add_symbol=add_symbol,
                    add_relation=add_relation,
                    parent_entity=type_entity,
                    package_name=qualified_name or package_name,
                    project_index=project_index,
                )
                continue

            if item_name in {"MethodDeclaration", "ConstructorDeclaration"}:
                method_name = getattr(item, "name", "")
                method_entity = add_symbol(
                    _CodeSymbol(
                        name=method_name,
                        label="Method",
                        qualified_name=".".join(part for part in [qualified_name, method_name] if part),
                        source_document_id=document.id,
                        metadata={"kind": "code", "language": "java", "node_type": "method"},
                    )
                )
                add_relation(type_entity, method_entity, "contains", method_name)

                for _, invocation in item.filter(getattr(javalang.tree, "MethodInvocation", object)):
                    target_name = getattr(invocation, "member", "")
                    if not target_name:
                        continue
                    target_entity = self._resolve_java_symbol(
                        target_name,
                        symbol_lookup,
                        document,
                        graph,
                        entities,
                        project_index=project_index,
                    )
                    add_relation(method_entity, target_entity, "calls", target_name, {"source": "code", "language": "java"})

    def _ingest_java_with_regex(
        self,
        document: Document,
        graph: KnowledgeGraph,
        project_index: Optional[JavaProjectIndex] = None,
    ) -> ExtractionResult:
        module_name = str(document.metadata.get("module_name") or self._module_name(Path(document.metadata.get("source_file", document.id))))
        entities: List[Entity] = []
        relations: List[Relation] = []

        class_pattern = re.compile(r"\b(?:public\s+)?(?:final\s+)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_][\w]*)")
        interface_pattern = re.compile(r"\binterface\s+(?P<name>[A-Za-z_][\w]*)")
        method_pattern = re.compile(
            r"\b(?:public|private|protected|static|final|synchronized|abstract|native|strictfp|\s)+"
            r"[\w<>\[\]]+\s+(?P<name>[A-Za-z_][\w]*)\s*\("
        )

        current_class: Optional[Entity] = None
        for pattern, label in ((class_pattern, "Class"), (interface_pattern, "Interface")):
            for match in pattern.finditer(document.text):
                symbol = self._symbol(
                    name=match.group("name"),
                    label="Class",
                    qualified_name=f"{module_name}.{match.group('name')}",
                    document=document,
                    metadata={"kind": "code", "language": "java", "source": "regex", "node_type": "class"},
                )
                entity = self._register_entity(graph, document, symbol, entities)
                current_class = entity

        for match in method_pattern.finditer(document.text):
            symbol = self._symbol(
                name=match.group("name"),
                label="Method",
                qualified_name=f"{module_name}.{match.group('name')}",
                document=document,
                metadata={"kind": "code", "language": "java", "source": "regex"},
            )
            entity = self._register_entity(graph, document, symbol, entities)
            if current_class is not None:
                relations.append(
                    Relation(
                        source_id=current_class.id,
                        target_id=entity.id,
                        relation_type="contains",
                        evidence=match.group("name"),
                        source_document_id=document.id,
                        confidence=0.6,
                        metadata={"source": "regex", "language": "java"},
                    )
                )
                graph.add_relation(relations[-1])

        return ExtractionResult(document=document, entities=entities, relations=relations)

    def _resolve_java_symbol(
        self,
        name: str,
        symbol_lookup: dict[str, Entity],
        document: Document,
        graph: KnowledgeGraph,
        entities: List[Entity],
        project_index: Optional[JavaProjectIndex] = None,
    ) -> Entity:
        lowered = name.lower()
        if lowered in symbol_lookup:
            return symbol_lookup[lowered]
        if project_index is not None:
            record = project_index.resolve(name=name, label="Method")
            if record is not None:
                return self._entity_from_java_record(record, graph, document, entities)
        return self._external_symbol(name, "Method", document, qualified_name=name, graph=graph, entities=entities)

    def _register_entity(self, graph: KnowledgeGraph, document: Document, symbol: _CodeSymbol, entities: List[Entity]) -> Entity:
        entity = Entity(
            id=stable_id("code", f"{symbol.label}:{symbol.qualified_name}"),
            name=symbol.name,
            label=symbol.label,
            source_document_id=symbol.source_document_id,
            confidence=0.9,
            aliases=[symbol.name],
            metadata={
                **symbol.metadata,
                "documents": [document.id],
                "qualified_name": symbol.qualified_name,
            },
        )
        if entity.id not in {item.id for item in entities}:
            entities.append(entity)
        graph.add_entity(entity)
        graph.add_mention(document.id, entity.id)
        return entity

    def _symbol(
        self,
        name: str,
        label: str,
        qualified_name: str,
        document: Document,
        metadata: Optional[dict] = None,
    ) -> _CodeSymbol:
        return _CodeSymbol(
            name=name,
            label=label,
            qualified_name=qualified_name,
            source_document_id=document.id,
            metadata=metadata or {"kind": "code", "language": self._language_for_document(document)},
        )

    def _external_symbol(
        self,
        name: str,
        label: str,
        document: Document,
        qualified_name: Optional[str] = None,
        graph: Optional[KnowledgeGraph] = None,
        entities: Optional[List[Entity]] = None,
    ) -> Entity:
        symbol = _CodeSymbol(
            name=name,
            label=label,
            qualified_name=qualified_name or name,
            source_document_id=document.id,
            metadata={
                "kind": "code",
                "language": self._language_for_document(document),
                "external": True,
            },
        )
        entity = Entity(
            id=stable_id("code", f"{symbol.label}:{symbol.qualified_name}"),
            name=symbol.name,
            label=symbol.label,
            source_document_id=symbol.source_document_id,
            confidence=0.5,
            aliases=[symbol.name],
            metadata={**symbol.metadata, "qualified_name": symbol.qualified_name},
        )
        if graph is not None and entities is not None:
            if entity.id not in {item.id for item in entities}:
                entities.append(entity)
            graph.add_entity(entity)
            graph.add_mention(document.id, entity.id)
        return entity

    def _entity_from_java_record(
        self,
        record: JavaSymbolRecord,
        graph: KnowledgeGraph,
        document: Document,
        entities: List[Entity],
    ) -> Entity:
        entity = Entity(
            id=stable_id("code", f"{record.label}:{record.qualified_name}"),
            name=record.name,
            label=record.label,
            source_document_id=record.source_document_id,
            confidence=0.8,
            aliases=[record.name],
            metadata={
                **record.metadata,
                "documents": [document.id],
                "qualified_name": record.qualified_name,
            },
        )
        if entity.id not in {item.id for item in entities}:
            entities.append(entity)
        graph.add_entity(entity)
        graph.add_mention(document.id, entity.id)
        return entity

    def _resolve_java_record_or_symbol(
        self,
        name: str,
        label: str,
        document: Document,
        graph: KnowledgeGraph,
        entities: List[Entity],
        qualified_name: Optional[str] = None,
        project_index: Optional[JavaProjectIndex] = None,
    ) -> Entity:
        if project_index is not None:
            record = project_index.resolve(name=name, qualified_name=qualified_name, label=label)
            if record is not None:
                return self._entity_from_java_record(record, graph, document, entities)
        return self._external_symbol(
            name,
            label,
            document,
            qualified_name=qualified_name or name,
            graph=graph,
            entities=entities,
        )

    def _language_for_document(self, document: Document) -> str:
        language = str(document.metadata.get("language", "")).lower().strip()
        if language:
            return language
        source_file = str(document.metadata.get("source_file", ""))
        suffix = Path(source_file).suffix.lower()
        return _CODE_EXTENSIONS.get(suffix, "")

    def _document_id_from_path(self, path: Path, root: Optional[Path] = None) -> str:
        if root is not None:
            try:
                relative = path.relative_to(root)
            except ValueError:
                relative = path.name
            base = str(relative).replace("\\", "/")
        else:
            base = path.name
        return base.replace("/", "__")

    def _module_name(self, path: Path, root: Optional[Path] = None) -> str:
        if root is not None:
            try:
                relative = path.relative_to(root)
            except ValueError:
                relative = path
            parts = list(relative.with_suffix("").parts)
        else:
            parts = list(path.with_suffix("").parts)
        return ".".join(part for part in parts if part) or path.stem

    def _source_root_for_path(self, path: Path, fallback: Path) -> Path:
        for ancestor in path.parents:
            if ancestor.name != "java":
                continue
            if ancestor.parent.name == "main" and ancestor.parent.parent.name == "src":
                return ancestor
        return fallback
