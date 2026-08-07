from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OperatorSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionSpec:
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Spec:
    family: str
    family_params: dict[str, Any] = field(default_factory=dict)
    n: int = 1000
    seed: int = 0
    operators: list[OperatorSpec] = field(default_factory=list)
    projection: ProjectionSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Spec":
        d = dict(d)
        d["operators"] = [OperatorSpec(**o) for o in d.get("operators", [])]
        proj = d.get("projection")
        d["projection"] = ProjectionSpec(**proj) if proj else None
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Spec":
        return cls.from_dict(json.loads(s))

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
