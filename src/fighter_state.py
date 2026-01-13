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
    prev_match_date: Optional[date] = None

    def days_since_last_fight(self, current_date: date) -> Optional[int]:
        target_date = self.last_match_date
        
        # If the fighter already fought today, look at the previous match date
        # to get the "time since last event" rather than 0
        if target_date == current_date:
            target_date = self.prev_match_date
            
        if target_date is None:
            return None
        return (current_date - target_date).days

    def record_match(self, won: bool, match_date: date) -> None:
        self.matches_fought += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        
        # Only update previous date if we moved to a new date
        if self.last_match_date is not None and self.last_match_date != match_date:
            self.prev_match_date = self.last_match_date
            
        self.last_match_date = match_date