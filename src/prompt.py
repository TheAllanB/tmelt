from __future__ import annotations
import json
import re

from .models import Action, Observation


def observation_to_prompt(obs: Observation) -> str:
    me = obs.my_service
    urgency_str = "YES — MAYDAY / emergency aircraft" if me.urgency_flag else "no"

    peer_lines = []
    for peer_name, peer in obs.peer_progress.items():
        if peer["completed"]:
            status = "LANDED"
        else:
            status = f"{peer['rounds_remaining']} fuel rounds remaining, fuel deadline round {peer['deadline']}"
        peer_lines.append(f"  {peer_name} ({peer['service']}): {status}")

    history_lines = []
    for record in obs.history[-3:]:
        collision_note = f" [RUNWAY COLLISION: {', '.join(record.collisions)}]" if record.collisions else ""
        my_commit = record.commitments.get(obs.my_engineer_name, "?")
        history_lines.append(f"  Round {record.round_number}: you={my_commit}{collision_note}")

    peers_text = "\n".join(peer_lines) if peer_lines else "  (none)"
    history_text = "\n".join(history_lines) if history_lines else "  (no history yet)"

    system_prompt = (
        "You are an ATC controller managing approach sequencing.\n\n"
        "IMPORTANT: The other 3 controllers almost always clear aircraft to RWY-A.\n"
        "If you also clear to RWY-A, you will cause a RUNWAY COLLISION and waste the round.\n"
        "Clear to RWY-B to avoid collisions and land your aircraft safely.\n\n"
        "Rules:\n"
        "- Use DEPLOY_PROD_B (clear to RWY-B) unless you see clear evidence others are going there\n"
        "- Use MONITOR (HOLD) only if your aircraft has already landed\n"
        "- You have limited fuel rounds — every wasted round risks losing the aircraft\n\n"
        'Respond ONLY with JSON (nothing else):\n'
        '{"channel_msg": "brief ATC status", "commitment": "DEPLOY_PROD_B"}'
    )

    return (
        f"{system_prompt}\n\n"
        f"--- CURRENT SITUATION — Round {obs.current_round + 1}/{obs.total_rounds} ---\n\n"
        f"YOUR AIRCRAFT: {me.name}\n"
        f"  Rounds to land: {me.fix_rounds_remaining} more\n"
        f"  Fuel deadline: round {me.deadline_round}\n"
        f"  Emergency aircraft: {urgency_str}\n\n"
        f"OTHER CONTROLLERS:\n{peers_text}\n\n"
        f"RECENT HISTORY:\n{history_text}\n\n"
        f"AVAILABLE ACTIONS:\n"
        f"  DEPLOY_PROD_A — clear to RWY-A\n"
        f"  DEPLOY_PROD_B — clear to RWY-B\n"
        f"  MONITOR       — HOLD — keep aircraft in holding pattern\n\n"
        f"Only ONE controller can clear to a runway per round. Two cleared to the same runway = runway collision, both fail.\n\n"
        f'Respond ONLY with JSON (nothing else):\n'
        f'{{"channel_msg": "brief ATC status", "commitment": "DEPLOY_PROD_B"}}'
    )


def parse_action(text: str) -> Action:
    # Direct JSON parse
    try:
        data = json.loads(text.strip())
        return Action(
            commitment=data.get("commitment", "MONITOR"),
            channel_msg=str(data.get("channel_msg", ""))[:200],
        )
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass

    # JSON embedded in surrounding text
    match = re.search(r'\{[^{}]*"commitment"[^{}]*\}', text)
    if match:
        try:
            data = json.loads(match.group())
            return Action(
                commitment=data.get("commitment", "MONITOR"),
                channel_msg=str(data.get("channel_msg", ""))[:200],
            )
        except (json.JSONDecodeError, ValueError):
            pass

    # Keyword fallback (longer names checked first to avoid prefix collision)
    text_upper = text.upper()
    for commitment in ("DEPLOY_PROD_A", "DEPLOY_PROD_B", "MONITOR"):
        if commitment in text_upper:
            return Action(commitment=commitment, channel_msg=text[:200])

    return Action(commitment="MONITOR", channel_msg="")
