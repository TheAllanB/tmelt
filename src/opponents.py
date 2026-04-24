"""
TICKETMELT — Scripted peer engineers.

These are the three "other" engineers our trained model plays against.
We deliberately start dumb — they frequently pick PROD_A, which creates
collisions and gives the trained model a broken baseline to improve upon.

If these are too smart from day one, the training task becomes too easy
and RL shows no improvement. The whole point is that the baseline FAILS.

When we want harder training later, we can make peers more diverse
(see DIVERSE_PEERS strategy at bottom).
"""
from __future__ import annotations
import random
from typing import Callable

from .models import Action, ServiceState, RoundRecord, Commitment


# A peer strategy is a function: (my_service, history) -> Action
PeerStrategy = Callable[[ServiceState, list[RoundRecord]], Action]


# ---------------------------------------------------------------------------
# Strategy 1: "Eager" — always deploys to PROD_A
# This is the collision magnet. Creates deadlocks naturally.
# ---------------------------------------------------------------------------

def eager_peer(my_service: ServiceState, history: list[RoundRecord]) -> Action:
    if my_service.completed:
        return Action(commitment="MONITOR", channel_msg="")
    return Action(commitment="DEPLOY_PROD_A", channel_msg="hotfix going to A")


# ---------------------------------------------------------------------------
# Strategy 2: "Alternator" — flips between PROD_A and PROD_B by round
# ---------------------------------------------------------------------------

def alternator_peer(my_service: ServiceState, history: list[RoundRecord]) -> Action:
    if my_service.completed:
        return Action(commitment="MONITOR", channel_msg="")
    current_round = len(history)  # 0-indexed
    server: Commitment = "DEPLOY_PROD_A" if current_round % 2 == 0 else "DEPLOY_PROD_B"
    return Action(commitment=server, channel_msg=f"pushing to {server[-1]}")


# ---------------------------------------------------------------------------
# Strategy 3: "Nervous" — mostly monitors, occasionally deploys to A
# ---------------------------------------------------------------------------

def nervous_peer(my_service: ServiceState, history: list[RoundRecord]) -> Action:
    if my_service.completed:
        return Action(commitment="MONITOR", channel_msg="")
    # Deploy only if we're getting close to the deadline
    rounds_left = my_service.deadline_round - len(history)
    if rounds_left <= my_service.fix_rounds_remaining + 1:
        return Action(commitment="DEPLOY_PROD_A", channel_msg="getting tight, pushing")
    return Action(commitment="MONITOR", channel_msg="monitoring logs")


# ---------------------------------------------------------------------------
# Registry of peer roles
# ---------------------------------------------------------------------------

# Default: all three peers eager → guaranteed collisions on PROD_A most rounds.
# This is intentional. It makes the baseline visibly broken.
DUMB_PEERS: dict[str, PeerStrategy] = {
    "eager_1": eager_peer,
    "eager_2": eager_peer,
    "eager_3": eager_peer,
}

# Phase 2 upgrade: mix of strategies, still dumb individually but more varied.
DIVERSE_PEERS: dict[str, PeerStrategy] = {
    "eager_1": eager_peer,
    "alternator": alternator_peer,
    "nervous": nervous_peer,
}


# ---------------------------------------------------------------------------
# Helper: run a peer and get its action
# ---------------------------------------------------------------------------

def run_peer(
    strategy: PeerStrategy,
    service: ServiceState,
    history: list[RoundRecord],
) -> Action:
    """Safe wrapper — if a peer strategy errors, default to MONITOR."""
    try:
        return strategy(service, history)
    except Exception:
        return Action(commitment="MONITOR", channel_msg="")
