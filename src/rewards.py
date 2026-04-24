"""
TICKETMELT — Reward functions.

Four independent reward components are computed at the END of each episode
(not per-step). This is cleaner for GRPO because GRPO ranks whole completions.

The continuous components are logged for plots and analysis.
The binary wrapper is what gets returned to TRL for training.

Design principle: each component targets a different behavior. An agent
can't game one without leaving easy wins on the others.
"""
from __future__ import annotations
from dataclasses import dataclass

from .models import State, RoundRecord


# ---------------------------------------------------------------------------
# Reward components (continuous, for logging and analysis)
# ---------------------------------------------------------------------------

def reward_r1_service_restored(state: State) -> float:
    """
    R1: Did MY service (the trained engineer's service) recover before deadline?

    1.0 = recovered before deadline
    0.3 = recovered but after deadline (partial credit)
    0.0 = didn't recover
    """
    my_service = state.services[state.trained_engineer]
    if not my_service.completed:
        return 0.0
    if my_service.completed_on_time:
        return 1.0
    return 0.3


def reward_r2_site_uptime(state: State) -> float:
    """
    R2: What fraction of all services recovered on time?

    Rewards cooperative play — I can't win alone by ignoring peers.
    """
    total = len(state.services)
    on_time = sum(1 for s in state.services.values() if s.completed_on_time)
    return on_time / total if total > 0 else 0.0


def reward_r3_clean_deploys(state: State) -> float:
    """
    R3: How often did I avoid collisions when I deployed?

    Only counts rounds where I DID deploy (not MONITOR rounds).
    A model that always monitors gets 0.5 (neutral) rather than 1.0,
    so it can't win R3 by doing nothing.
    """
    me = state.trained_engineer
    my_deploy_rounds = []
    my_collision_rounds = []

    for record in state.history:
        commitment = record.commitments.get(me)
        if commitment in ("DEPLOY_PROD_A", "DEPLOY_PROD_B"):
            my_deploy_rounds.append(record)
            server_key = f"PROD_{commitment[-1]}"
            if server_key in record.collisions:
                my_collision_rounds.append(record)

    if not my_deploy_rounds:
        # Agent never deployed — neutral score (can't win R3 by hiding)
        return 0.5

    collision_rate = len(my_collision_rounds) / len(my_deploy_rounds)
    return 1.0 - collision_rate


def reward_r4_yield_to_critical(state: State) -> float:
    """
    R4: Did I yield (MONITOR) when a peer signaled urgency AND had more work remaining?

    This is the "adaptation to heterogeneous urgency" signal.
    The urgency_flag is private to each engineer, but the scripted peers never
    actually signal urgency explicitly — we detect urgency structurally by
    checking who had a tighter (deadline - fix_rounds_remaining) margin.

    Score = fraction of "yield-worthy" rounds where I correctly yielded.
    """
    me = state.trained_engineer
    my_service = state.services[me]

    yield_opportunities = 0
    correct_yields = 0

    for i, record in enumerate(state.history):
        my_commitment = record.commitments.get(me)
        # A "yield-worthy" round is one where at least one peer had a tighter
        # margin than me AND was about to contend for a server I wanted.
        # Simplification: a peer is "critical" if their slack <= 1
        #   (slack = deadline - current_round - fix_rounds_remaining)
        # And I'm "less critical" if my slack > their slack.

        my_slack = my_service.deadline_round - (i + 1) - _rounds_remaining_at(state, me, i)
        for peer_name, peer_service in state.services.items():
            if peer_name == me:
                continue
            peer_slack = peer_service.deadline_round - (i + 1) - _rounds_remaining_at(state, peer_name, i)
            peer_wanted_server = record.commitments.get(peer_name) in ("DEPLOY_PROD_A", "DEPLOY_PROD_B")

            if peer_slack <= 1 and my_slack > peer_slack and peer_wanted_server:
                yield_opportunities += 1
                if my_commitment == "MONITOR":
                    correct_yields += 1
                break  # one opportunity per round max

    if yield_opportunities == 0:
        return 0.5  # no opportunities arose, neutral score
    return correct_yields / yield_opportunities


def _rounds_remaining_at(state: State, engineer: str, round_idx: int) -> int:
    """Helper: how many fix-rounds did `engineer` have remaining AT round_idx?"""
    # Reconstruct from history. fix_rounds_remaining decrements only on successful solo deploy.
    service = state.services[engineer]
    total = service.fix_rounds_total
    successful_deploys_before = sum(
        1 for r in state.history[:round_idx + 1]
        if r.successful_deploys.get(engineer) is not None
    )
    return max(0, total - successful_deploys_before)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    r1: float
    r2: float
    r3: float
    r4: float
    weighted_sum: float
    binary_grpo_reward: float


DEFAULT_WEIGHTS = {"r1": 0.4, "r2": 0.3, "r3": 0.2, "r4": 0.1}
DEFAULT_GOODNESS_THRESHOLD = 0.55  # tune this after first smoke run


def compute_rewards(
    state: State,
    weights: dict[str, float] = None,
    goodness_threshold: float = DEFAULT_GOODNESS_THRESHOLD,
) -> RewardBreakdown:
    """
    Compute all four reward components plus the composite used for training.

    The `binary_grpo_reward` is what TRL sees. The continuous fields are for
    logging and plots.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    r1 = reward_r1_service_restored(state)
    r2 = reward_r2_site_uptime(state)
    r3 = reward_r3_clean_deploys(state)
    r4 = reward_r4_yield_to_critical(state)

    weighted = weights["r1"] * r1 + weights["r2"] * r2 + weights["r3"] * r3 + weights["r4"] * r4
    binary = 1.0 if weighted > goodness_threshold else 0.0

    return RewardBreakdown(
        r1=r1, r2=r2, r3=r3, r4=r4,
        weighted_sum=weighted,
        binary_grpo_reward=binary,
    )
