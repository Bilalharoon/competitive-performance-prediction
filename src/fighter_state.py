from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass
class FighterState:
    fighter_id: int

    rating: float
    tournament_date: str  # 'YYYY-MM'

    matches_fought: int = 0
    wins: int = 0
    losses: int = 0
    last_match_date: Optional[date] = None

    def days_since_last_fight(self, current_date: date) -> Optional[int]:
        if self.last_match_date is None:
            return None
        return (date.fromisoformat(self.tournament_date) - self.last_match_date).days

    def record_match(self, won: bool, match_date: date) -> None:
        self.matches_fought += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.last_match_date = match_date