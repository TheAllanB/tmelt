# TICKETMELT Phase 2 — Build Progress

## Milestones

- [x] Core environment (`src/environment.py`, `src/models.py`, `src/rewards.py`, `src/opponents.py`)
- [x] 10-test suite for core env (`tests/test_env.py`) — all passing
- [x] Smoke test script (`smoke_test.py`)
- [x] `requirements.txt` (fastapi, uvicorn, httpx, pytest, matplotlib)
- [x] Prompt template + action parser (`src/prompt.py`) — 9 tests passing
- [x] FastAPI server wrapper (`src/server.py`) — 6 tests passing
- [x] Rollout function (`src/rollout.py`) + `tests/test_rollout.py` — 6 tests passing
- [x] `Dockerfile` + `openenv.yaml` (HF Spaces deployment)
- [x] Baseline eval script (`training/baseline_eval.py`)
- [x] Plot script (`plot_rewards.py`) — outputs `plots/reward_curve.png` + `plots/component_rewards.png`
- [x] Training notebook (`training/ticketmelt_training.ipynb`) — GRPO on Colab T4
- [x] `TICKETMELT_README.md` (copied from Downloads, paths updated)
- [x] Full test suite green — 31 tests passing

## Test count: 31/31 passing

---

## Resume Prompt

Paste this at the start of the next session to resume exactly where we left off:

```
We're building TICKETMELT, a multi-agent RL environment for a hackathon (April 2026).
The project is at C:\Users\allan\Desktop\tmelt.

Phase 1 (core env) is complete. Phase 2 is partially done — see progress.md for the checklist.

Completed so far:
- requirements.txt
- src/prompt.py + tests/test_prompt.py (9 tests)
- src/server.py + tests/test_server.py (6 tests)

Next task to implement (TDD, in this order):
1. src/rollout.py + tests/test_rollout.py — run_episode() + parse_action_from_model() with mocked model (6 tests)
2. Dockerfile + openenv.yaml
3. training/baseline_eval.py (eval script writing JSON logs to compare before/after training)
4. plot_rewards.py — reads two JSON files, writes plots/reward_curve.png + plots/component_rewards.png
5. training/ticketmelt_training.ipynb — GRPO Colab notebook using Unsloth + TRL + Qwen-2.5-3B-Instruct
6. TICKETMELT_README.md — copy C:\Users\allan\Downloads\TICKETMELT_README.md to project root, update file paths to match actual structure (training/ not notebooks/, etc.)
7. Run full pytest suite — must hit 31 tests passing

The full implementation plan is at docs/superpowers/plans/2026-04-23-ticketmelt-phase2.md.
Use the executing-plans skill and continue from Task 4 (rollout).
The README from Downloads is the polished user-facing version — use it instead of writing a new one.
```
