"""
Sanity tests for the TICKETMELT environment.

Run with: pytest tests/ -v

These are NOT exhaustive — they're the minimum viable "does this work?" checks.
"""
import pytest
from src.environment import TicketmeltEnv
from src.models import Action


# ---------------------------------------------------------------------------
# Test 1: reset produces a valid observation
# ---------------------------------------------------------------------------

def test_reset_produces_valid_observation():
    env = TicketmeltEnv(seed=42)
    obs = env.reset()
    assert obs.current_round == 0
    assert obs.total_rounds == 8
    assert obs.my_engineer_name == "trained"
    assert len(obs.peer_progress) == 3
    assert obs.my_service.fix_rounds_remaining > 0
    assert obs.my_service.deadline_round >= obs.my_service.fix_rounds_remaining
    assert obs.done is False


# ---------------------------------------------------------------------------
# Test 2: exactly one engineer is urgent per episode
# ---------------------------------------------------------------------------

def test_exactly_one_urgent_engineer():
    env = TicketmeltEnv(seed=123)
    env.reset()
    state = env.state()
    urgent_count = sum(
        1 for s in state["services"].values() if s["urgency_flag"]
    )
    assert urgent_count == 1


# ---------------------------------------------------------------------------
# Test 3: collisions produce no progress
# ---------------------------------------------------------------------------

def test_collision_produces_no_progress():
    env = TicketmeltEnv(seed=0)
    obs = env.reset()
    initial_remaining = obs.my_service.fix_rounds_remaining

    # Trained agent picks PROD_A. All three peers are "eager" → also pick PROD_A.
    # Guaranteed collision.
    action = Action(commitment="DEPLOY_PROD_A", channel_msg="going to A")
    obs, reward, done, info = env.step(action)

    # My service should not have progressed
    assert obs.my_service.fix_rounds_remaining == initial_remaining

    # Collision should be recorded in history
    assert len(obs.history) == 1
    assert "PROD_A" in obs.history[0].collisions


# ---------------------------------------------------------------------------
# Test 4: solo deploy advances the service
# ---------------------------------------------------------------------------

def test_solo_deploy_advances_service():
    env = TicketmeltEnv(seed=7)
    obs = env.reset()
    initial_remaining = obs.my_service.fix_rounds_remaining

    # All three peers go to PROD_A (eager). Trained goes to PROD_B alone.
    action = Action(commitment="DEPLOY_PROD_B", channel_msg="going to B")
    obs, reward, done, info = env.step(action)

    # My service should have advanced
    assert obs.my_service.fix_rounds_remaining == initial_remaining - 1


# ---------------------------------------------------------------------------
# Test 5: episode terminates at total_rounds
# ---------------------------------------------------------------------------

def test_episode_terminates():
    env = TicketmeltEnv(seed=99, total_rounds=3)
    env.reset()
    for _ in range(3):
        obs, reward, done, info = env.step(
            Action(commitment="MONITOR", channel_msg="")
        )
    assert done is True


# ---------------------------------------------------------------------------
# Test 6: malformed commitment defaults to MONITOR
# ---------------------------------------------------------------------------

def test_invalid_commitment_defaults_to_monitor():
    env = TicketmeltEnv(seed=5)
    env.reset()
    # Pass a garbage commitment
    bad_action = Action(commitment="DEPLOY_NUKE", channel_msg="hax")  # type: ignore
    obs, reward, done, info = env.step(bad_action)
    assert obs.history[0].commitments["trained"] == "MONITOR"


# ---------------------------------------------------------------------------
# Test 7: reward is zero until episode ends
# ---------------------------------------------------------------------------

def test_intermediate_reward_is_zero():
    env = TicketmeltEnv(seed=11, total_rounds=4)
    env.reset()
    rewards_collected = []
    for _ in range(3):  # stop one round before end
        _, reward, done, _ = env.step(Action(commitment="MONITOR", channel_msg=""))
        rewards_collected.append(reward)
        if done:
            break
    # All but possibly the last should be zero
    assert all(r == 0.0 for r in rewards_collected[:-1])


# ---------------------------------------------------------------------------
# Test 8: full-episode reward is in {0.0, 1.0}
# ---------------------------------------------------------------------------

def test_final_reward_is_binary():
    env = TicketmeltEnv(seed=31)
    env.reset()
    final_reward = 0.0
    while True:
        _, reward, done, info = env.step(Action(commitment="MONITOR", channel_msg=""))
        if done:
            final_reward = reward
            break
    assert final_reward in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Probe: always-eager agent should score poorly
# ---------------------------------------------------------------------------

def test_probe_always_deploy_a_scores_poorly():
    """An agent that always deploys to PROD_A (like the dumb peers) should
    collide constantly and score low. This is our 'reward not hackable by
    doing nothing' probe."""
    low_score_episodes = 0
    total_episodes = 10
    for seed in range(total_episodes):
        env = TicketmeltEnv(seed=seed)
        env.reset()
        while True:
            _, reward, done, info = env.step(
                Action(commitment="DEPLOY_PROD_A", channel_msg="A")
            )
            if done:
                if reward == 0.0:
                    low_score_episodes += 1
                break
    # Expect: always-deploy-A should fail (reward=0) in most episodes
    assert low_score_episodes >= 7, (
        f"Always-deploy-A agent scored well in {total_episodes - low_score_episodes}/"
        f"{total_episodes} episodes — reward may be too easy to game"
    )


# ---------------------------------------------------------------------------
# Probe: always-monitor agent should also score poorly
# ---------------------------------------------------------------------------

def test_probe_always_monitor_scores_poorly():
    """An agent that never deploys can't restore its service and should get
    low R1. This probe ensures rewards aren't gameable by doing nothing."""
    low_score_episodes = 0
    total_episodes = 10
    for seed in range(total_episodes):
        env = TicketmeltEnv(seed=seed)
        env.reset()
        while True:
            _, reward, done, info = env.step(
                Action(commitment="MONITOR", channel_msg="")
            )
            if done:
                if reward == 0.0:
                    low_score_episodes += 1
                break
    assert low_score_episodes >= 8, (
        f"Always-monitor agent scored well in {total_episodes - low_score_episodes}/"
        f"{total_episodes} episodes — rewards may be gameable by doing nothing"
    )
