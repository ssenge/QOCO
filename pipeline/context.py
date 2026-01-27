from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, MutableMapping, Optional, TypeVar

T = TypeVar("T")


@dataclass
class PipelineContext(MutableMapping[str, object]):
    data: Dict[str, object] = field(default_factory=dict)

    def __getitem__(self, key: str) -> object:
        return self.data[key]

    def __setitem__(self, key: str, value: object) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key: str, default: Optional[T] = None) -> object | T:
        return self.data.get(key, default)
