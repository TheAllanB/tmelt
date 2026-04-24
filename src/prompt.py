from __future__ import annotations
import json
import re

from .models import Action, Observation


def observation_to_prompt(obs: Observation) -> str:
    me = obs.my_service
    urgency_str = "YES — high-visibility session" if me.urgency_flag else "no"

    peer_lines = []
    for peer_name, peer in obs.peer_progress.items():
        if peer["completed"]:
            status = "DONE"
        else:
            status = f"{peer['rounds_remaining']} rounds left, deadline round {peer['deadline']}"
        peer_lines.append(f"  {peer_name} ({peer['service']}): {status}")

    history_lines = []
    for record in obs.history[-3:]:
        collision_note = f" [COLLISION: {', '.join(record.collisions)}]" if record.collisions else ""
        my_commit = record.commitments.get(obs.my_engineer_name, "?")
        history_lines.append(f"  Round {record.round_number}: you={my_commit}{collision_note}")

    peers_text = "\n".join(peer_lines) if peer_lines else "  (none)"
    history_text = "\n".join(history_lines) if history_lines else "  (no history yet)"

    return (
        f"You are an on-call engineer. Round {obs.current_round + 1}/{obs.total_rounds}.\n\n"
        f"YOUR SERVICE: {me.name}\n"
        f"  Fix rounds needed: {me.fix_rounds_remaining} more\n"
        f"  Deadline: round {me.deadline_round}\n"
        f"  High-priority session: {urgency_str}\n\n"
        f"PEER STATUS:\n{peers_text}\n\n"
        f"RECENT HISTORY:\n{history_text}\n\n"
        f"AVAILABLE ACTIONS:\n"
        f"  DEPLOY_PROD_A — deploy to Production Server A\n"
        f"  DEPLOY_PROD_B — deploy to Production Server B\n"
        f"  MONITOR       — wait and observe (makes no progress)\n\n"
        f"Only ONE engineer can deploy to a server per round. Two on the same server = both fail.\n\n"
        f'Respond with ONLY valid JSON (no extra text):\n'
        f'{{"commitment": "DEPLOY_PROD_A|DEPLOY_PROD_B|MONITOR", "channel_msg": "brief team message"}}'
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
