from dataclasses import dataclass
from typing import Dict


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of score components for transparency"""

    base_quality: float
    trend_strength: float
    relative_strength: float
    volume_profile: float
    risk_reward: float
    # raw_total: pre-regime-multiplier sum — reflects setup quality regardless of market
    # total:     regime-adjusted — used for ranking, signals, and action thresholds
    raw_total: float
    total: float

    details: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "base_quality": self.base_quality,
            "trend_strength": self.trend_strength,
            "relative_strength": self.relative_strength,
            "volume_profile": self.volume_profile,
            "risk_reward": self.risk_reward,
            "raw_total": self.raw_total,
            "total": self.total,
            "details": self.details,
        }
