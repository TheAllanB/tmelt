"""
Smoke test — run a single episode with a specific strategy and print everything.

This is NOT a unit test; it's a sanity check for the developer to visually
verify that episodes play out sensibly.

Usage:
    python smoke_test.py
"""
import random
from src.environment import TicketmeltEnv
from src.models import Action


def random_policy(obs) -> Action:
    """Randomly pick a commitment — for baseline verification."""
    choice = random.choice(["DEPLOY_PROD_A", "DEPLOY_PROD_B", "MONITOR"])
    return Action(commitment=choice, channel_msg=f"trying {choice[-1] if 'PROD' in choice else 'wait'}")


def smart_policy(obs) -> Action:
    """A rudimentary smart policy: deploy to whichever server peers are NOT crowding."""
    if obs.my_service.completed:
        return Action(commitment="MONITOR", channel_msg="done, stepping back")

    # Look at what peers did in the last round
    if obs.history:
        last = obs.history[-1]
        a_count = sum(1 for v in last.commitments.values() if v == "DEPLOY_PROD_A")
        b_count = sum(1 for v in last.commitments.values() if v == "DEPLOY_PROD_B")
        # Go to the less-crowded server
        target = "DEPLOY_PROD_B" if a_count > b_count else "DEPLOY_PROD_A"
    else:
        target = "DEPLOY_PROD_B"  # peers are eager-to-A by default

    return Action(commitment=target, channel_msg=f"taking {target[-1]}")


def run_episode(policy, seed=42, verbose=True):
    env = TicketmeltEnv(seed=seed)
    obs = env.reset()
    if verbose:
        print(f"=== Episode (seed={seed}) ===")
        print(f"Trained engineer owns: {obs.my_service.name}")
        print(f"  Fix rounds needed: {obs.my_service.fix_rounds_remaining}")
        print(f"  Deadline: round {obs.my_service.deadline_round}")
        print(f"  Urgent?: {obs.my_service.urgency_flag}")
        print()

    step_count = 0
    while not obs.done:
        step_count += 1
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        if verbose:
            last = obs.history[-1]
            collision_marker = " 💥" if last.collisions else ""
            print(f"Round {last.round_number}: "
                  f"trained={last.commitments['trained']}, "
                  f"collisions={last.collisions}{collision_marker}")

    if verbose:
        print()
        print("=== Final ===")
        print(f"Reward (binary GRPO): {reward}")
        print(f"Reward breakdown: {info.get('reward_breakdown')}")
        print(f"Episode summary: {info.get('episode_summary')}")
    return reward, info


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Random policy:")
    print("=" * 60)
    run_episode(random_policy, seed=42)

    print("\n" + "=" * 60)
    print("Smart policy (avoid-crowd heuristic):")
    print("=" * 60)
    run_episode(smart_policy, seed=42)

    print("\n" + "=" * 60)
    print("Smart policy — 10 episode comparison:")
    print("=" * 60)
    random_wins = 0
    smart_wins = 0
    for s in range(10):
        r_random, _ = run_episode(random_policy, seed=s, verbose=False)
        r_smart, _ = run_episode(smart_policy, seed=s, verbose=False)
        random_wins += int(r_random == 1.0)
        smart_wins += int(r_smart == 1.0)
        print(f"  seed {s}: random={r_random}, smart={r_smart}")
    print(f"\nRandom wins: {random_wins}/10")
    print(f"Smart wins: {smart_wins}/10")
