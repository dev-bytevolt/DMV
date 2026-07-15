from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClassifiedDocument:
    id: str
    name: str
    type: str
    pages: list[int]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassifiedDocument:
        return cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            pages=sorted(int(p) for p in data["pages"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "pages": self.pages,
        }


@dataclass
class ClassificationResult:
    documents: list[ClassifiedDocument] = field(default_factory=list)
    empty_pages: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationResult:
        return cls(
            documents=[
                ClassifiedDocument.from_dict(doc) for doc in data.get("documents", [])
            ],
            empty_pages=sorted(int(p) for p in data.get("empty_pages", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "empty_pages": self.empty_pages,
        }
