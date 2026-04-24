# Architectural Patterns

Patterns that recur across TICKETMELT's files. If you see one of these while editing, follow it. If you're introducing code that conflicts with one, pause and reconsider.

---

## Reward Hybrid Pattern

**What:** Rewards are computed continuously for analysis but emitted as binary for training.

**Where it appears:**
- `src/rewards.py:151` — `compute_rewards()` returns `RewardBreakdown` with both `weighted_sum` (continuous) and `binary_grpo_reward` (0.0 or 1.0)
- `src/environment.py:155` — `step()` returns the binary reward to the caller, stashes the full breakdown in `info["reward_breakdown"]`

**Why:** GRPO ranks completions within a group. Binary rewards give cleaner gradients than nuanced continuous ones (per TRL team's documented recommendation). But continuous values are needed for plots, per-component analysis, and tuning.

**Rule:** When adding a new reward component, compute it continuously, then have it feed into the weighted sum. Let the `goodness_threshold` decide binary. Don't create separate "training reward" and "analysis reward" code paths — one source of truth.

---

## Goldilocks Difficulty

**What:** Task difficulty is deliberately tuned so baseline fails often but success is possible.

**Where it appears:**
- `src/opponents.py:68` — `DUMB_PEERS` has all three peers always pick PROD_A (deliberate)
- `src/environment.py:30` — `MIN_FIX_ROUNDS=1`, `MAX_FIX_ROUNDS=4`, `TOTAL_ROUNDS_DEFAULT=8` (tuned for ~60% random baseline)

**Why:** RL needs non-zero reward probability to learn. Too easy → trained model looks no different from base. Too hard → model never sees success, never learns. Verified via `smoke_test.py` — random wins ~6/10, smart wins ~10/10.

**Rule:** If baseline shifts (e.g. you change peer strategies), verify `smoke_test.py` still shows ~4-7/10 random wins. If it hits 10/10 or 0/10, retune fix_rounds bounds or swap peer strategies.

---

## Format-Safety Defaults

**What:** Any malformed input silently degrades to a safe default. Rewards handle the "did they comply" judgment.

**Where it appears:**
- `src/environment.py:193` — `_sanitize_action()` coerces bad commitments to `"MONITOR"` and truncates oversized messages
- `src/opponents.py:86` — `run_peer()` wraps peer strategies in try/except, falls back to MONITOR
- Invalid commitment in reward accounting = treated as MONITOR for R3 scoring

**Why:** The env must never crash on bad agent output during training. An LLM will produce gibberish sometimes; the env should keep running so GRPO can assign low reward and move on. Crashing kills the rollout batch.

**Rule:** Any new input path (new action field, new API endpoint) needs a safe default. Never raise on agent-provided data. Raise only on env configuration errors (e.g. wrong number of peers in `__init__`).

---

## Dataclass-As-Wire-Format

**What:** All types that cross the env↔client boundary are plain dataclasses with only JSON-safe fields.

**Where it appears:**
- `src/models.py` — every class is `@dataclass`, only uses `str`, `int`, `bool`, `list`, `dict`, `Literal`
- `src/models.py:33` — `Action.to_dict()` uses stdlib `asdict()`, no custom serializer
- `src/environment.py:178` — `state()` returns pre-built dict via `asdict(svc)`, not raw `ServiceState` objects

**Why:** OpenEnv ships types over HTTP. Dataclass + JSON primitives = free serialization. Custom classes or non-JSON types (datetime, enum) break the client/server boundary.

**Rule:** Before adding a field to `Action`, `Observation`, `ServiceState`, or `RoundRecord`, check it's JSON-serializable. Use `Literal[...]` for enums, never `enum.Enum`. Use `int` for timestamps, not `datetime`.

---

## Peer Strategy Registry

**What:** Scripted opponents are function-valued dict entries, swappable by passing a different registry to the env constructor.

**Where it appears:**
- `src/opponents.py:22` — `PeerStrategy = Callable[[ServiceState, list[RoundRecord]], Action]`
- `src/opponents.py:68` — `DUMB_PEERS` and `DIVERSE_PEERS` are dicts of name→strategy
- `src/environment.py:51` — `TicketmeltEnv(peer_strategies=...)` accepts any registry

**Why:** Curriculum learning. Start with `DUMB_PEERS` (collisions everywhere). Once training plateaus, swap to `DIVERSE_PEERS` (harder) without touching env code. No need to subclass the env or add conditionals.

**Rule:** New opponent behaviors go in `opponents.py` as functions matching `PeerStrategy`. Group them into named dicts with a semantic name (e.g. `ADVERSARIAL_PEERS`). Never inline peer logic into `environment.py`.

---

## End-Of-Episode Scoring

**What:** `step()` returns reward 0.0 every round except the terminal round, where the full episode is scored.

**Where it appears:**
- `src/environment.py:155` — The `if self._state.done` check gates the reward computation
- `src/rewards.py` — All four reward functions take the final `State`, not per-round deltas

**Why:** GRPO cares about whole-trajectory quality vs other trajectories in the group. Intermediate rewards add variance without signal. Also avoids the "model learns to maximize round-1 reward at the cost of round-8" pathology.

**Rule:** If you're tempted to add a per-round reward (e.g. "small bonus for a good message"), don't. Add it as a new component of the end-of-episode score instead. Per-round rewards and GRPO fight each other.

---

## Seeded Determinism

**What:** Every random decision in the env flows through `self.rng` (a seeded `random.Random` instance), never the global `random` module.

**Where it appears:**
- `src/environment.py:56` — `self.rng = random.Random(seed)` in `__init__`
- `src/environment.py:67` — optional seed reset in `reset()`
- All calls like `self.rng.randint(...)`, `self.rng.choice(...)`, `self.rng.shuffle(...)` throughout `reset()`

**Why:** Reproducibility. Baseline eval and trained eval must run on the same seeds for valid comparison. Also enables exact reproduction of reward-hacking bugs during debugging.

**Rule:** Never use `random.random()` or `random.choice()` directly. Always go through `self.rng`. If adding a new source of randomness (e.g. stochastic peer behavior), thread the RNG through — don't seed a new one.

---

## Trained Engineer Is Named "trained"

**What:** The trained agent's slot is always keyed `"trained"` throughout the env.

**Where it appears:**
- `src/environment.py:70` — `engineer_names = ["trained"] + list(...)`
- `src/environment.py:133` — `round_actions: dict = {"trained": action}`
- `src/rewards.py:31` — `state.services[state.trained_engineer]` where `trained_engineer = "trained"`

**Why:** Hardcoding one name keeps rewards, observation construction, and the prompt template aligned without passing an engineer-id parameter through every call. The scripted peers take the other three slots.

**Rule:** Don't rename `"trained"` to something else. If we ever add multi-agent self-play, that will be a bigger refactor (multiple trained slots) — not just a rename.
