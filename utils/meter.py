from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AverageMeter:
    """
    Tracks average of a scalar metric.
    """
    name: str
    sum: float = 0.0
    cnt: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * int(n)
        self.cnt += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.cnt)


@dataclass
class MetricBook:
    """
    A small container for multiple AverageMeters.
    """
    meters: Dict[str, AverageMeter] = field(default_factory=dict)

    def update(self, name: str, val: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter(name=name)
        self.meters[name].update(val, n)

    def as_dict(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}
