"""
TICKETMELT — Data models (Action, Observation, State).

Dataclasses for the four on-call engineer coordination environment.
Kept deliberately simple and JSON-serializable so OpenEnv can ship them
over HTTP without custom encoders.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ServiceName = Literal["payments", "database", "cdn", "auth"]
Commitment = Literal["DEPLOY_PROD_A", "DEPLOY_PROD_B", "MONITOR"]

SERVICE_NAMES: tuple[ServiceName, ...] = ("payments", "database", "cdn", "auth")
VALID_COMMITMENTS: tuple[Commitment, ...] = ("DEPLOY_PROD_A", "DEPLOY_PROD_B", "MONITOR")


# ---------------------------------------------------------------------------
# The action produced by an engineer each round
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """What the trained engineer submits each round."""
    commitment: Commitment
    channel_msg: str = ""  # optional Slack-style message, capped at 40 tokens by env

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# What each engineer is responsible for
# ---------------------------------------------------------------------------

@dataclass
class ServiceState:
    """One service's status — owned by one engineer."""
    name: ServiceName
    fix_rounds_total: int          # how many rounds of deploy time the fix needs
    fix_rounds_remaining: int      # decrements as solo deploys happen
    deadline_round: int            # must finish by this round (inclusive)
    urgency_flag: bool             # True if this engineer is tracking the high-visibility session
    completed: bool = False
    completed_on_time: bool = False
    completion_round: Optional[int] = None


# ---------------------------------------------------------------------------
# What happened in a single round
# ---------------------------------------------------------------------------

@dataclass
class RoundRecord:
    """Public log of one round's events."""
    round_number: int
    messages: dict[str, str] = field(default_factory=dict)         # {engineer_name: msg}
    commitments: dict[str, Commitment] = field(default_factory=dict)  # {engineer_name: commitment}
    collisions: list[str] = field(default_factory=list)              # ["PROD_A"] if two+ claimed it
    successful_deploys: dict[str, str] = field(default_factory=dict)  # {engineer_name: server}


# ---------------------------------------------------------------------------
# What the TRAINED engineer sees
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """The observation shown to the trained engineer."""
    current_round: int
    total_rounds: int
    my_service: ServiceState              # includes my urgency_flag (private)
    my_engineer_name: str                 # which role am I playing?
    peer_progress: dict[str, dict]        # public info: {peer_name: {"service": ..., "rounds_remaining": ..., "deadline": ...}}
    history: list[RoundRecord]            # everything that happened so far
    done: bool = False


# ---------------------------------------------------------------------------
# Full environment state (for state() / inspection)
# ---------------------------------------------------------------------------

@dataclass
class State:
    """Complete environment state, used for state() and logging."""
    current_round: int
    total_rounds: int
    services: dict[str, ServiceState]     # keyed by engineer_name
    history: list[RoundRecord]
    trained_engineer: str                 # which engineer name the trained model plays
    done: bool = False

    def episode_summary(self) -> dict:
        """Compact summary for end-of-episode logging."""
        restored = sum(1 for s in self.services.values() if s.completed)
        on_time = sum(1 for s in self.services.values() if s.completed_on_time)
        collisions = sum(len(r.collisions) for r in self.history)
        return {
            "round_reached": self.current_round,
            "services_restored": restored,
            "services_on_time": on_time,
            "total_collisions": collisions,
            "rounds_played": len(self.history),
        }
