"""Load balance statistics (spec 5, 7 and 7.2).

* Std is the population one (divide by N buildings, not N - 1): the buildings we have are
  the whole population, not a sample of it.
* CV = std / mean, lower means better balanced.
* With no guests (K = 0) the basic stats are 0 and CV is skipped (cv = None) so we never divide by zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class LoadStats:
    counts: Dict[int, int] = field(default_factory=dict)  # node_id -> guest count (including buildings with 0 guests)
    total: int = 0
    buildings: int = 0
    maximum: int = 0
    minimum: int = 0
    mean: float = 0.0
    std: float = 0.0
    cv: Optional[float] = None
    note: str = ""


def compute_load_stats(counts: Mapping[int, int]) -> LoadStats:
    """Stats from the guest count per building (buildings with 0 guests must be in the mapping)."""
    items = dict(counts)
    values = list(items.values())
    total = sum(values)
    n = len(values)
    if n == 0 or total == 0:
        return LoadStats(
            counts=items,
            total=total,
            buildings=n,
            note="ไม่มีแขกในระบบ: สถิติพื้นฐานเป็น 0 และไม่คำนวณ CV",
        )
    mean = total / n
    variance = math.fsum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    return LoadStats(
        counts=items,
        total=total,
        buildings=n,
        maximum=max(values),
        minimum=min(values),
        mean=mean,
        std=std,
        cv=std / mean,
    )
