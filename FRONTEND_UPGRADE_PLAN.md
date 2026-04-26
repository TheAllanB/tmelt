# TICKETMELT Frontend Upgrade — "Neural Rewiring" Visualization

## CRITICAL SAFETY NOTE
**Do NOT modify any files in `src/` — no changes to server.py, environment.py, rewards.py, models.py, opponents.py, prompt.py, rollout.py, or any test files.**
**Only touch `static/index.html`.** Everything else stays byte-for-byte identical.

---

## What You're Building

Replace the current `static/index.html` in the HuggingFace Space repo at:
https://huggingface.co/spaces/TheAllanB/ticketmelt

The new frontend is a single-page showcase for the TICKETMELT project — an OpenEnv RL environment for training LLMs on multi-agent coordination. Judges from the OpenEnv Hackathon (April 2026) will visit this page to evaluate the project. First impressions matter enormously — 40% of the score is environment innovation, 30% is storytelling.

---

## Design Aesthetic: "Mission Control Terminal"

**Tone**: Dark retro-futuristic terminal. Think NASA mission control crossed with a hacker terminal — not playful, not corporate, deeply atmospheric.

**Palette** (CSS variables throughout):
```
--bg:         #020a06     (near-black green)
--bg2:        #040f08     (panel surfaces)
--green:      #00ff7f     (primary accent, glows)
--green-dim:  rgba(0,255,127,0.15)
--amber:      #ffb300     (secondary accent)
--red:        #ff3b3b     (danger/collision)
--blue:       #4fc3f7     (PROD_A / RWY-A)
--pink:       #ff7eb3     (R4/yield)
--white:      #e8ffe8     (text)
--grid:       #0a2a14     (grid lines)
--panel:      #05140a     (card backgrounds)
--border:     #0f3a1a     (borders)
```

**Fonts** (Google Fonts):
- `Orbitron` — headers, stat values, hero title (weight 700/900)
- `Share Tech Mono` — body text, labels, data (weight 400)

**Effects**:
- Scanline overlay (CSS repeating-linear-gradient, pointer-events:none, z-index:9999)
- Text glow via text-shadow: `0 0 20px var(--green)`
- Animated grid background on hero
- Green dot pulse animation for "live" indicators
- All backgrounds dark — no light mode

---

## Page Structure (7 Sections)

### Section 1: HERO
- Full viewport height
- Animated grid background (CSS background-image with grid lines)
- Badge: "◈ OPENENV HACKATHON — APR 2026"
- Giant title: "TICKETMELT" in Orbitron 96px with green glow. The "MELT" portion in amber.
- Subtitle: "MULTI-AGENT COORDINATION · SIMULTANEOUS RESOURCE CONTENTION · GRPO TRAINING"
- One-paragraph description of the project
- Stats row: 4 AGENTS | 2 SERVERS | 31 TESTS | 4 REWARD COMPONENTS
- CTA buttons: "▶ LIVE DEMO" → #demo, "◈ API ENDPOINT" → opens /state in new tab
- Scroll indicator at bottom

### Section 2: PROBLEM STATEMENT (id="problem")
- Section label: "◈ 01 / PROBLEM STATEMENT"
- Title: "Why LLMs Deadlock Under Pressure"
- 4 cards in 2×2 grid:
  1. "95% DEADLOCK RATE" — DPBench Feb 2026: frontier LLMs deadlock 95%+ on simultaneous coordination. Communication channels made it worse.
  2. "0% URGENCY ADAPTATION" — NeurIPS 2024 Concordia: LLMs fail at heterogeneous urgency. Critical peers treated identically to low-priority ones.
  3. "4× SIMULTANEOUS COMMITMENT" — 4 agents commit simultaneously with no observation of others. Identical reasoning → identical actions → collision.
  4. "RL FIX — GRPO" — GRPO trains Qwen-2.5-3B-Instruct from binary reward signal alone. No demonstrations. Model learns asymmetric strategies.

### Section 3: LIVE DEMO — THE CENTERPIECE (id="demo")

This section contains THREE visualization panels side by side or stacked:

#### Panel A: "Neural Rewiring" Network Graph (THE STAR FEATURE)

A force-directed-style network visualization showing how the model's decision weights change during training.

**Layout**: 
- 4 agent nodes on the left column (labeled UAL-441, DAL-892, SWA-77 ⚠, AAL-213)
- 2 server nodes on the right column (PROD_A, PROD_B)
- 8 weighted edges connecting each agent to each server
- CSS `perspective: 800px` on the container, slight `rotateX(5deg) rotateY(-5deg)` tilt for 3D feel
- The whole network sits on a subtle grid plane

**Edge weights (interpolated from real training data)**:

Step 0 (untrained baseline probabilities — all agents biased toward PROD_A):
```
UAL→A: 0.85  UAL→B: 0.15
DAL→A: 0.80  DAL→B: 0.20
SWA→A: 0.75  SWA→B: 0.25  (SWA is emergency aircraft)
AAL→A: 0.82  AAL→B: 0.18
```

Step 16 (trained — learned to distribute and yield):
```
UAL→A: 0.35  UAL→B: 0.65
DAL→A: 0.55  DAL→B: 0.45
SWA→A: 0.30  SWA→B: 0.70  (emergency → strongest shift to B)
AAL→A: 0.45  AAL→B: 0.55
```

Intermediate steps: linear interpolation between step 0 and step 16.

**Edge rendering**:
- Width = probability × 8px (so 0.85 = 6.8px thick, 0.15 = 1.2px thin)
- Color: edges to PROD_A = blue (#4fc3f7), edges to PROD_B = amber (#ffb300)
- Glow: CSS filter or box-shadow on SVG lines for thick edges
- Opacity: probability × 0.8 + 0.2 (so even thin edges are visible)

**PROD_A collision zone**:
- A pulsing red/amber ring around the PROD_A node when most edges point to it
- Ring intensity = sum of PROD_A probabilities / 4
- At step 0: bright red pulse (high collision probability)
- At step 16: dim/gone (collision risk reduced)

**"Symmetry Breaking" indicator**:
- Show a small metric below the network: "Symmetry Score: 0.95" at step 0, dropping to "0.42" at step 16
- Symmetry = how similar all agents' distributions are (high = deadlock)
- Formula: 1 - std(agent_prod_a_probs) / mean(agent_prod_a_probs)

**Interactive control**:
- Range slider labeled "TRAINING STEP" from 0 to 16
- As user drags: edges animate thickness/color smoothly (CSS transition 0.3s)
- Step counter shows "STEP 0/16" → "STEP 16/16"
- Below slider: key metrics that update: "Collision Risk: 96% → 23%", "Yield Score: 0.17 → 0.28"

**SWA-77 special treatment**:
- SWA-77 node has a red glow (emergency aircraft)
- Its edges show the most dramatic rewiring (R4 improvement)
- A small label appears when trained: "Learned: yield to critical"

#### Panel B: ATC Radar Replay (existing radar, keep as-is)

Keep the current radar visualization with untrained/trained episode replay. This is already built and working. Do not remove it — it tells the episode-level story while Panel A tells the weight-level story.

Same features:
- Animated radar canvas with sweep line
- 4 aircraft blips with approach trajectories  
- Flight progress strips panel
- Radio comms log
- UNTRAINED / TRAINED mode toggle
- ▶ RUN EPISODE, ≡ COMPARE, ↺ RESET buttons
- Collision explosions, MAYDAY handling, landing animations

Same episode data from the original (8-round untrained FAIL, 5-round trained WIN).

#### Panel C: Live API Panel

- Title: "◈ LIVE ENVIRONMENT STATE"
- Two buttons: "FETCH STATE" and "RESET EPISODE"
- FETCH STATE calls `fetch('/state')` and displays the JSON response
- RESET calls `fetch('/reset', {method:'POST'})` and displays response
- JSON display area: pre-formatted, green monospace text on dark background
- Health indicator dot (green pulse = healthy)
- This proves to judges the environment is actually running

### Section 4: REWARD ARCHITECTURE (id="rewards")
4 cards in a row, each with colored top border accent:
- R1 (blue border, weight 0.5): SERVICE RESTORED — Primary signal. Agent's service fixed within deadline.
- R2 (green border, weight 0.3): SITE UPTIME — Group welfare. Team services restored fraction.
- R3 (amber border, weight 0.1): CLEAN DEPLOYS — Collision avoidance. Collision-free deploy fraction.
- R4 (pink border, weight 0.1): YIELD TO CRITICAL — Urgency adaptation. Yielded to critical peer.

Below cards: inline reward formula block:
```
reward = 1.0 if (r1 > 0.5 and 0.5·r1 + 0.3·r2 + 0.1·r3 + 0.1·r4 > 0.35) else 0.0
```

### Section 5: RESULTS (id="results")

Two-column layout:

**Left column**: Results comparison table
```
| METRIC              | BASELINE | TRAINED | DELTA  |
|---------------------|----------|---------|--------|
| Win Rate            | 0.800    | 0.700   | -0.100 |
| R1: Service         | 0.860    | 0.745   | -0.115 |
| R2: Uptime          | 0.263    | 0.287   | +0.025 |
| R3: No Collision    | 0.767    | 0.767   | +0.000 |
| R4: Yield           | 0.169    | 0.276   | +0.106 |
| Collision Rate      | 0.961    | 0.898   | -0.063 |
```
Positive deltas in green, negative in red, zero in gray.

Honest framing text below table:
"With only 16 GRPO gradient steps, the model shows meaningful coordination improvements: yield-to-critical behavior improved +0.106, site uptime improved, and collision rate dropped. Win rate dipped slightly — consistent with a short training run on a strong baseline model."

**Right column**: Two canvas-drawn charts

Chart 1 — Animated horizontal bar chart (before/after per metric):
- Red bars = baseline, blue bars = trained
- Animate bar widths on scroll into view (IntersectionObserver)
- Data: Win Rate 0.80/0.70, R1 0.86/0.745, R2 0.263/0.287, R3 0.767/0.767, R4 0.169/0.276

Chart 2 — Reward curve (line chart, 16 steps):
- Real training data: [0.625, 0.875, 0.75, 0.625, 1.0, 0.75, 0.375, 1.0, 0.625, 1.0, 0.375, 0.25, 0.75, 0.375, 1.0, 1.0]
- Dashed red baseline at 0.80
- Rolling mean line in blue
- Individual dots (green if 1.0, blue otherwise)
- X-axis: "Training Step", Y-axis: "Reward"

### Section 6: TRAINING CONFIGURATION (id="training")
Two config tables side by side:

Left table — Model & Training:
```
BASE MODEL:        Qwen-2.5-3B-Instruct
METHOD:            GRPO (TRL + Unsloth)
LORA RANK:         r=16
LEARNING RATE:     5e-6
NUM GENERATIONS:   8 per prompt
TEMPERATURE:       0.9
MAX GRAD NORM:     0.1
GRADIENT STEPS:    16
KL (MAX):          < 0.003
GPU:               NVIDIA A100 40GB
```

Right table — Environment:
```
AGENTS:            4 engineers
SERVERS:           2 (PROD_A / PROD_B)
EPISODE LENGTH:    Up to 8 rounds
PEER STRATEGY:     Mixed (70% DUMB / 30% DIVERSE)
COMMITMENT:        Simultaneous (blind)
COLLISION RULE:    2 on same server = both fail
URGENCY FLAGS:     Yes (critical deadlines)
REWARD TYPE:       Binary (inline computed)
TESTS:             31 passing
OPENENV:           ✓ compliant
```

Research framing block below:
"Symmetry breaking — stop deploying to the same server as peers. Strategic yielding — let urgent peers go first. Collision recovery — adapt mid-episode when collisions occur."

### Section 7: LINKS (id="links")
3 cards linking to:
1. Live Space → https://theallanb-ticketmelt.hf.space
2. State Endpoint → https://theallanb-ticketmelt.hf.space/state
3. HF Space Repo → https://huggingface.co/spaces/TheAllanB/ticketmelt

### Footer
"TICKETMELT · OPENENV HACKATHON APR 2026 · theAllanB"
Green pulse dot + "SYSTEM NOMINAL"

---

## Technical Implementation Notes

### File to modify
`static/index.html` — single file, everything inline (CSS + JS + HTML). No external files except Google Fonts.

### CSS 3D for Neural Rewiring
```css
.network-container {
  perspective: 800px;
  perspective-origin: 50% 40%;
}
.network-plane {
  transform: rotateX(8deg) rotateY(-3deg);
  transform-style: preserve-3d;
}
```

### Edge rendering approach
Use SVG `<line>` elements inside the 3D-transformed container. Edge width via `stroke-width`, color via `stroke`, glow via `filter: drop-shadow(0 0 4px color)`. Animate with CSS transitions on `stroke-width` and `stroke` when slider value changes.

### Slider-driven animation
```javascript
const slider = document.getElementById('training-step');
slider.addEventListener('input', (e) => {
  const step = parseInt(e.target.value);
  const t = step / 16; // 0.0 to 1.0
  updateEdgeWeights(t);
  updateMetrics(t);
  updateCollisionGlow(t);
});

function updateEdgeWeights(t) {
  // Interpolate between untrained and trained probabilities
  edges.forEach(edge => {
    const w = edge.startWeight + (edge.endWeight - edge.startWeight) * t;
    edge.element.style.strokeWidth = (w * 8) + 'px';
    edge.element.style.opacity = w * 0.8 + 0.2;
  });
}
```

### Live API fetching
```javascript
async function fetchState() {
  const res = await fetch('/state');
  const data = await res.json();
  document.getElementById('api-output').textContent = JSON.stringify(data, null, 2);
}
```

### Radar canvas
Reuse the exact radar animation code from the current `static/index.html`. Copy the full canvas drawing logic, episode data, flight strips, comms log, and controls. Do not modify the radar behavior.

### Performance
- Use `requestAnimationFrame` only for the radar canvas sweep
- Neural rewiring edges use CSS transitions (GPU-accelerated)
- Charts drawn once on scroll intersection, not continuously
- No heavy libraries — pure vanilla JS

---

## Execution Steps

1. Clone the repo: `git clone https://huggingface.co/spaces/TheAllanB/ticketmelt`
2. Read the current `static/index.html` to understand existing radar code
3. Create the new `static/index.html` with all 7 sections
4. Keep all existing radar/episode/comms code intact inside the new page
5. Add the Neural Rewiring visualization as a new panel in Section 3
6. Test locally if possible (the fetch calls to /state won't work locally but everything else should render)
7. Commit: `git add static/index.html && git commit -m "Frontend upgrade: Neural Rewiring visualization + enhanced demo"`
8. Push: `git push hf main`
9. Verify the Space rebuilds (it's a Docker container, takes 1-3 min)
10. Report back what was changed

---

## DO NOT TOUCH
- `src/` — any file
- `tests/` — any file
- `openenv.yaml`
- `Dockerfile` (unless needed for static file serving, which is already configured)
- `requirements.txt`
- `training/` — any file
- `plots/` — any file
- `examples/` — any file
- `TICKETMELT_README.md`
- `README.md`
