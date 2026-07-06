from __future__ import annotations

import re
from hashlib import sha1
from typing import Iterable, List


_STOPWORDS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "By",
    "For",
    "From",
    "In",
    "Into",
    "On",
    "Of",
    "Or",
    "The",
    "To",
    "With",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "item"


def stable_id(prefix: str, value: str) -> str:
    digest = sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_whitespace(text))
    return [part.strip() for part in parts if part.strip()]


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def is_probable_entity_token(token: str) -> bool:
    if not token:
        return False
    if token in _STOPWORDS:
        return False
    if token.isupper() and len(token) > 1:
        return True
    return token[0].isupper()

