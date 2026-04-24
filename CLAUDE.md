# TICKETMELT

OpenEnv-compliant multi-agent RL environment. Four on-call engineers, two production servers, eight rounds. Trains LLMs to break convergent reasoning and adapt to heterogeneous urgency in simultaneous coordination. Hackathon submission (April 2026).

## Why

Training data gap: DPBench (Feb 2026) showed frontier LLMs deadlock 95%+ on simultaneous coordination. NeurIPS 2024 Concordia Contest showed LLMs struggle with urgency adaptation. This env trains both capabilities in one scenario.

## Tech Stack

- **Python 3.10+**, stdlib-only for the core env (no heavy deps)
- **OpenEnv 0.1** (spec from Meta + Hugging Face) for env interface + HF Spaces deployment
- **FastAPI** + **uvicorn** for the server wrapper (to be added)
- **TRL** (Hugging Face) + **GRPO** for training
- **Unsloth** for efficient LoRA training on Colab T4
- **Qwen-2.5-3B-Instruct** as base model
- **pytest** for tests

## Project Structure

```
ticketmelt/
├── src/                      # Core env (pure Python, no network deps)
│   ├── __init__.py           # Public API — what the client gets
│   ├── models.py             # Action, Observation, State dataclasses
│   ├── environment.py        # reset/step/state + collision resolution
│   ├── opponents.py          # Scripted peer engineer strategies
│   └── rewards.py            # R1-R4 + binary-for-GRPO wrapper
├── tests/
│   └── test_env.py           # 10 sanity + probe tests (all passing)
├── smoke_test.py             # Visual episode runner for dev inspection
├── .claude/docs/             # Deeper docs for specific topics
└── CLAUDE.md                 # You are here
```

## Key Files (With Line Anchors)

- **src/environment.py:64** — `reset()` generates episodes. Deadline logic has edge case around small `total_rounds` — see the `lo > hi` guard at line 92.
- **src/environment.py:119** — `step()` returns `(obs, reward, done, info)`. Reward is `0.0` every round EXCEPT termination. Continuous breakdown is in `info["reward_breakdown"]`.
- **src/environment.py:193** — `_sanitize_action()` is the format-safety gate. Invalid commitment → `MONITOR`.
- **src/environment.py:203** — `_resolve_round()` is where collision logic lives. 2+ on one server → both fail. Solo → advance.
- **src/rewards.py:151** — `compute_rewards()` returns `RewardBreakdown` with both continuous components and binary GRPO reward.
- **src/rewards.py:148** — `DEFAULT_GOODNESS_THRESHOLD = 0.55`. Tune this after first smoke training run.
- **src/rewards.py:147** — `DEFAULT_WEIGHTS = {"r1": 0.4, "r2": 0.3, "r3": 0.2, "r4": 0.1}`.
- **src/opponents.py:68** — `DUMB_PEERS` (default). All three peers always pick PROD_A. Deliberately creates collisions.
- **src/opponents.py:75** — `DIVERSE_PEERS` (Phase 2 upgrade). Mix of strategies for harder training.
- **src/models.py:20** — `VALID_COMMITMENTS = ("DEPLOY_PROD_A", "DEPLOY_PROD_B", "MONITOR")`. Anything else → MONITOR default.

## Essential Commands

```bash
# Run all tests (should show 10 passed)
python -m pytest tests/ -v

# Watch a full episode unfold visually
python smoke_test.py

# Run a single test
python -m pytest tests/test_env.py::test_collision_produces_no_progress -v

# Run just the probe tests (verify rewards aren't gameable)
python -m pytest tests/test_env.py -v -k "probe"
```

## What's Built vs What's Next

**Built and tested:**
- Core environment (reset/step/state)
- Collision resolution
- Scripted peers (dumb default)
- All 4 reward components + binary wrapper
- 10-test suite (sanity + anti-hacking probes)

**Not yet built (next steps in order):**
1. FastAPI server wrapper (`src/server.py`) — exposes env over HTTP
2. `Dockerfile` + `openenv.yaml` manifest for HF Spaces deployment
3. Prompt template — converts `Observation` → text for LLM
4. Rollout function — glues model + env for one full episode
5. GRPO training loop in Colab notebook
6. Baseline eval script
7. Plotting code for reward curves

## Critical Design Decisions

- **Solo developer, 3-day hackathon.** Cut scope ruthlessly. Favor working over elegant.
- **Reward is BINARY to GRPO, CONTINUOUS in logs.** See `.claude/docs/architectural_patterns.md#reward-hybrid-pattern`.
- **Peers start dumb on purpose.** If baseline is too good, RL shows no improvement. See `.claude/docs/architectural_patterns.md#goldilocks-difficulty`.
- **End-of-episode scoring only.** Intermediate rewards all 0.0. Cleaner for GRPO group comparison.
- **Format gate before reward gate.** Invalid action → silent MONITOR default + zero reward. See `src/environment.py:193`.

## Additional Documentation

Check these when working on specific topics:

- **`.claude/docs/architectural_patterns.md`** — Design patterns that recur across files: the reward hybrid pattern, Goldilocks difficulty, format-safety defaults, dataclass-as-wire-format, peer strategy registry.
- **`../TICKETMELT_README.md`** — User-facing project pitch, judging narrative, results section. Update results after training.
- **`../TICKETMELT_PLAN.md`** — Original phase-wise execution plan (partially superseded by solo path).

## Hackathon Judging Weights (Memorize)

- 40% Environment innovation — already strong (research-grounded)
- 30% Storytelling — README + 2-min video
- 20% Showing improvement — reward curve + before/after rollouts
- 10% Reward/pipeline quality — multi-component + anti-hacking probes

Ship > perfect. If behind schedule, cut: full self-play, ablations, probe expansion, demo UI polish. Keep: deployment, one training run, plots, video, README.
