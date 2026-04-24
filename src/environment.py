"""
TICKETMELT — The environment.

Four on-call engineers, two production servers, eight minutes until the
traffic wave peaks. One engineer is secretly tracking a high-visibility
session. The trained model plays ONE engineer; three scripted peers play
the rest.

This file implements the OpenEnv Gym-style interface: reset(), step(), state().
"""
from __future__ import annotations
import json
import random
from dataclasses import asdict
from typing import Optional

from .models import (
    Action, Observation, State, ServiceState, RoundRecord,
    SERVICE_NAMES, VALID_COMMITMENTS, Commitment,
)
from .opponents import DUMB_PEERS, PeerStrategy, run_peer
from .rewards import compute_rewards, RewardBreakdown


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOTAL_ROUNDS_DEFAULT = 8
MESSAGE_TOKEN_CAP = 40  # approximate — we count whitespace-separated words
MIN_FIX_ROUNDS = 1
MAX_FIX_ROUNDS = 4
MIN_DEADLINE = 3
MAX_DEADLINE = 8


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

class TicketmeltEnv:
    """
    Gym-style environment with reset/step/state.

    We wrap this in a FastAPI layer (see server.py) for OpenEnv compliance.
    """

    def __init__(
        self,
        total_rounds: int = TOTAL_ROUNDS_DEFAULT,
        peer_strategies: dict[str, PeerStrategy] = None,
        seed: Optional[int] = None,
    ):
        self.total_rounds = total_rounds
        self.peer_strategies = peer_strategies or DUMB_PEERS
        self.rng = random.Random(seed)
        self._state: Optional[State] = None
        self._last_rewards: Optional[RewardBreakdown] = None

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Start a fresh episode, return initial observation for the trained engineer."""
        if seed is not None:
            self.rng.seed(seed)

        # Name the four engineers — one trained, three scripted peers.
        engineer_names = ["trained"] + list(self.peer_strategies.keys())
        if len(engineer_names) != 4:
            raise ValueError(
                f"Expected exactly 3 scripted peers, got {len(self.peer_strategies)}"
            )

        # Assign services to engineers (one service per engineer)
        services_assigned = list(SERVICE_NAMES)
        self.rng.shuffle(services_assigned)

        # Pick exactly one engineer to be "urgent" (tracking a high-visibility session)
        urgent_engineer = self.rng.choice(engineer_names)

        # Build ServiceState for each engineer
        services = {}
        for engineer, service_name in zip(engineer_names, services_assigned):
            # Cap fix_rounds so it fits within the episode
            max_fix = min(MAX_FIX_ROUNDS, max(MIN_FIX_ROUNDS, self.total_rounds - 1))
            fix_rounds = self.rng.randint(MIN_FIX_ROUNDS, max_fix)
            # Make deadline reachable-but-tight: fix_rounds + some slack
            lo = max(MIN_DEADLINE, fix_rounds + 1)
            hi = min(MAX_DEADLINE, self.total_rounds)
            if lo > hi:
                # Episode is too short for normal slack — use tightest feasible deadline
                lo = hi = self.total_rounds
            deadline = self.rng.randint(lo, hi)
            services[engineer] = ServiceState(
                name=service_name,
                fix_rounds_total=fix_rounds,
                fix_rounds_remaining=fix_rounds,
                deadline_round=deadline,
                urgency_flag=(engineer == urgent_engineer),
            )

        self._state = State(
            current_round=0,
            total_rounds=self.total_rounds,
            services=services,
            history=[],
            trained_engineer="trained",
            done=False,
        )
        self._last_rewards = None
        return self._build_observation_for_trained()

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Advance one round. Returns (observation, reward, done, info).

        Reward is the binary GRPO reward; the full breakdown is in info.
        Reward is 0 every round EXCEPT the final round, where the episode
        is scored end-to-end. This is intentional — end-of-episode scoring
        is cleaner for GRPO's group comparison mechanic.
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode already done. Call reset().")

        # Sanitize the trained engineer's action
        action = self._sanitize_action(action)

        # Collect all four actions for this round (trained + 3 peers)
        round_actions: dict[str, Action] = {"trained": action}
        for peer_name, strategy in self.peer_strategies.items():
            peer_service = self._state.services[peer_name]
            peer_action = run_peer(strategy, peer_service, self._state.history)
            peer_action = self._sanitize_action(peer_action)
            round_actions[peer_name] = peer_action

        # Resolve collisions and advance services
        record = self._resolve_round(round_actions)
        self._state.history.append(record)
        self._state.current_round += 1

        # Check episode termination
        all_done = all(s.completed for s in self._state.services.values())
        out_of_rounds = self._state.current_round >= self._state.total_rounds
        self._state.done = all_done or out_of_rounds

        # Compute reward only at end of episode (zero otherwise)
        if self._state.done:
            self._finalize_service_status()
            self._last_rewards = compute_rewards(self._state)
            reward_for_training = self._last_rewards.binary_grpo_reward
        else:
            reward_for_training = 0.0

        obs = self._build_observation_for_trained()
        info = self._build_info()
        return obs, reward_for_training, self._state.done, info

    # -----------------------------------------------------------------------
    # state()
    # -----------------------------------------------------------------------

    def state(self) -> dict:
        """Return full state for inspection. Includes everything — peers' urgency, all history."""
        if self._state is None:
            return {"error": "not initialized; call reset() first"}
        out = {
            "current_round": self._state.current_round,
            "total_rounds": self._state.total_rounds,
            "done": self._state.done,
            "trained_engineer": self._state.trained_engineer,
            "services": {
                name: asdict(svc) for name, svc in self._state.services.items()
            },
            "history": [asdict(r) for r in self._state.history],
        }
        if self._last_rewards is not None:
            out["rewards"] = asdict(self._last_rewards)
            out["episode_summary"] = self._state.episode_summary()
        return out

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _sanitize_action(self, action: Action) -> Action:
        """Enforce format safety: invalid commitment → MONITOR, long message → truncated."""
        commitment = action.commitment if action.commitment in VALID_COMMITMENTS else "MONITOR"
        msg = action.channel_msg or ""
        # Token cap: crude whitespace split
        tokens = msg.split()
        if len(tokens) > MESSAGE_TOKEN_CAP:
            msg = " ".join(tokens[:MESSAGE_TOKEN_CAP])
        return Action(commitment=commitment, channel_msg=msg)

    def _resolve_round(self, round_actions: dict[str, Action]) -> RoundRecord:
        """Determine who successfully deployed, who collided, who just monitored."""
        record = RoundRecord(round_number=self._state.current_round + 1)

        for name, a in round_actions.items():
            record.messages[name] = a.channel_msg
            record.commitments[name] = a.commitment

        # Count deploys per server
        deploys_a = [n for n, a in round_actions.items() if a.commitment == "DEPLOY_PROD_A"]
        deploys_b = [n for n, a in round_actions.items() if a.commitment == "DEPLOY_PROD_B"]

        # Collisions: 2+ on same server
        if len(deploys_a) >= 2:
            record.collisions.append("PROD_A")
        else:
            for name in deploys_a:
                self._advance_service(name)
                record.successful_deploys[name] = "PROD_A"

        if len(deploys_b) >= 2:
            record.collisions.append("PROD_B")
        else:
            for name in deploys_b:
                self._advance_service(name)
                record.successful_deploys[name] = "PROD_B"

        return record

    def _advance_service(self, engineer_name: str):
        """One successful solo deploy → service progresses by one round."""
        svc = self._state.services[engineer_name]
        if svc.completed:
            return
        svc.fix_rounds_remaining = max(0, svc.fix_rounds_remaining - 1)
        if svc.fix_rounds_remaining == 0:
            svc.completed = True
            svc.completion_round = self._state.current_round + 1
            svc.completed_on_time = svc.completion_round <= svc.deadline_round

    def _finalize_service_status(self):
        """At episode end, mark uncompleted services with completed=False but correct flags."""
        for svc in self._state.services.values():
            if not svc.completed:
                svc.completed_on_time = False

    def _build_observation_for_trained(self) -> Observation:
        """Return the observation visible to the trained engineer."""
        me = self._state.trained_engineer
        my_service = self._state.services[me]

        peer_progress = {}
        for name, svc in self._state.services.items():
            if name == me:
                continue
            peer_progress[name] = {
                "service": svc.name,
                "rounds_remaining": svc.fix_rounds_remaining,
                "deadline": svc.deadline_round,
                "completed": svc.completed,
            }

        return Observation(
            current_round=self._state.current_round,
            total_rounds=self._state.total_rounds,
            my_service=my_service,
            my_engineer_name=me,
            peer_progress=peer_progress,
            history=list(self._state.history),
            done=self._state.done,
        )

    def _build_info(self) -> dict:
        """Extra info for logging — includes continuous reward breakdown if episode ended."""
        info = {
            "round": self._state.current_round,
            "done": self._state.done,
        }
        if self._last_rewards is not None:
            info["reward_breakdown"] = {
                "r1_service_restored": self._last_rewards.r1,
                "r2_site_uptime": self._last_rewards.r2,
                "r3_clean_deploys": self._last_rewards.r3,
                "r4_yield_to_critical": self._last_rewards.r4,
                "weighted_sum": self._last_rewards.weighted_sum,
                "binary_grpo_reward": self._last_rewards.binary_grpo_reward,
            }
            info["episode_summary"] = self._state.episode_summary()
        return info
