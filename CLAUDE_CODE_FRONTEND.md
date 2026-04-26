# TICKETMELT — Complete Frontend Rebuild

## SAFETY RULE — READ FIRST
**Only modify `static/index.html`.** Do not touch src/, tests/, Dockerfile, requirements.txt, openenv.yaml, training/, plots/, examples/, or any README file. Zero backend changes.

---

## Task
Replace `static/index.html` in https://huggingface.co/spaces/TheAllanB/ticketmelt with a spectacular single-page showcase. The current page has correct content but zero visual impact — no 3D effects, no animations, no Neural Rewiring visualization. Build the full page described below from scratch as a single self-contained HTML file.

---

## Aesthetic Direction: Dark Retro-Futuristic Terminal
- Background: #020a06 (near-black green)
- Primary accent: #00ff7f (green, always glowing via text-shadow)
- Secondary: #ffb300 (amber)
- Danger: #ff3b3b (red)
- Info: #4fc3f7 (blue)
- Yield: #ff7eb3 (pink)
- Fonts: Orbitron (headers/values) + Share Tech Mono (body) from Google Fonts
- Scanline overlay: `position:fixed; inset:0; background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.06) 2px,rgba(0,0,0,0.06) 4px); pointer-events:none; z-index:9999`
- Every section dark — no light backgrounds anywhere

---

## SECTION 1 — HERO (full viewport)

Animated CSS grid background:
```css
background-image: linear-gradient(#0a2a14 1px, transparent 1px), linear-gradient(90deg, #0a2a14 1px, transparent 1px);
background-size: 40px 40px;
animation: gridPulse 4s ease-in-out infinite;
```

Content (centered):
- Badge: `◈ OPENENV HACKATHON — APR 2026` in amber border
- Title: `TICKETMELT` in Orbitron 96px green glow. The "MELT" in amber. text-shadow: `0 0 40px #00ff7f, 0 0 80px rgba(0,255,127,0.3)`
- Subtitle: `MULTI-AGENT COORDINATION · SIMULTANEOUS RESOURCE CONTENTION · GRPO TRAINING`
- Description: "An OpenEnv-compliant RL environment that trains LLMs to break convergent reasoning — the documented failure mode where frontier models deadlock 95%+ of the time on simultaneous coordination tasks."
- Stats row: `4 AGENTS` | `2 SERVERS` | `31 TESTS` | `4 REWARD COMPONENTS` — each value in Orbitron 32px
- Two CTA buttons: `▶ LIVE DEMO` (green) → #demo, `◈ API ENDPOINT` (amber) → https://theallanb-ticketmelt.hf.space/state
- Scroll indicator at bottom: bouncing ↓

---

## SECTION 2 — PROBLEM (id="problem")

Label: `◈ 01 / PROBLEM STATEMENT`
Title: `Why LLMs Deadlock Under Pressure`

4 cards in 2×2 grid, each with amber left border:
1. `95% DEADLOCK RATE` — DPBench (Feb 2026) showed frontier LLMs deadlock 95%+ on simultaneous coordination tasks. Adding communication channels made performance worse, not better.
2. `0% URGENCY ADAPTATION` — NeurIPS 2024 Concordia: LLM agents fail when scenarios require detecting peers with different urgency levels. Critical peers are treated identically to low-priority ones.
3. `4× SIMULTANEOUS COMMITMENT` — 4 agents commit simultaneously with no observation of others. Identical reasoning produces identical actions which collide. The symmetry is structural.
4. `RL FIX — GRPO` — GRPO trains Qwen-2.5-3B-Instruct from binary reward signal alone. No demonstrations. The model learns asymmetric coordination strategies through reward shaping.

---

## SECTION 3 — LIVE DEMO (id="demo") — THE CENTERPIECE

### 3A: NEURAL REWIRING VISUALIZATION (THE STAR — BUILD THIS CAREFULLY)

This is the most important visual. It must be spectacular.

**Container**: Full width panel, 520px tall, dark background (#030d06), border 1px solid #0f3a1a.

**Inside**: An SVG network graph that shows how the model's decision probabilities change during training. Use CSS perspective transform on the container for 3D feel:
```css
.rewiring-viewport {
  perspective: 1200px;
  perspective-origin: 50% 45%;
}
.rewiring-scene {
  transform: rotateX(12deg) rotateY(-6deg) rotateZ(1deg);
  transform-style: preserve-3d;
}
```

**Network layout** — drawn with SVG inside the 3D-transformed div:
- Left column (x=120): 4 agent nodes — UAL-441 (y=80), DAL-892 (y=200), SWA-77 (y=320, emergency=red glow), AAL-213 (y=440)
- Right column (x=620): 2 server nodes — PROD_A (y=170, blue), PROD_B (y=350, amber)
- SVG viewBox="0 0 740 520"

**Node rendering**:
- Agent nodes: 60×36px rounded rect, #05140a fill, 1px #0f3a1a border, callsign in Share Tech Mono 11px green
- PROD_A node: 80×44px rect, blue border (#4fc3f7), label in blue, subtle blue glow
- PROD_B node: 80×44px rect, amber border (#ffb300), label in amber, subtle amber glow
- SWA-77: red border (#ff3b3b), red text, pulsing red glow animation: `box-shadow: 0 0 15px #ff3b3b`
- Each node has a small probability label below it showing current weight to each server

**Edge rendering** — 8 SVG `<line>` elements (4 agents × 2 servers):
- Edges to PROD_A: stroke #4fc3f7 (blue)
- Edges to PROD_B: stroke #ffb300 (amber)
- stroke-width driven by probability (0.85 → 7px thick, 0.15 → 1px thin)
- opacity = probability × 0.85 + 0.15
- CSS transition: `transition: stroke-width 0.4s ease, opacity 0.4s ease, stroke 0.3s ease`
- Thick high-probability edges: add SVG filter for glow effect:
  ```svg
  <defs>
    <filter id="glow-blue"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <filter id="glow-amber"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <filter id="glow-red"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  ```

**COLLISION DANGER ZONE**: A pulsing red ellipse around PROD_A:
```svg
<ellipse id="danger-zone" cx="620" cy="170" rx="70" ry="40" fill="none" stroke="#ff3b3b" stroke-width="1.5" opacity="0.6"/>
```
Animate with: `animation: dangerPulse 1.2s ease-in-out infinite`
At step 0: opacity 0.8, stroke-width 2. At step 16: opacity 0.1, stroke-width 0.5.

**Particle flow on edges**: Small animated dots travel along each edge showing "decisions flowing":
```javascript
// For each edge, create 2-3 particles that travel from agent to server
// Particle speed proportional to edge weight
// Use requestAnimationFrame with t += 0.005 * weight
// Position: lerp between agent node center and server node center
```
Draw particles as small circles (r=3) in the edge color. This makes the network feel alive.

**Probability data** — interpolate linearly between these at each step t (0→1):

Step 0 (untrained):
```
UAL→A: 0.85  UAL→B: 0.15
DAL→A: 0.80  DAL→B: 0.20
SWA→A: 0.75  SWA→B: 0.25
AAL→A: 0.82  AAL→B: 0.18
```

Step 16 (trained):
```
UAL→A: 0.35  UAL→B: 0.65
DAL→A: 0.55  DAL→B: 0.45
SWA→A: 0.28  SWA→B: 0.72
AAL→A: 0.45  AAL→B: 0.55
```

**Training step slider** below the network:
```html
<input type="range" min="0" max="16" value="0" id="step-slider">
```
Label: `TRAINING STEP: 0 / 16`

As slider moves:
- All edge widths/opacities animate
- Danger zone fades
- Four live metrics update:
  - `COLLISION RISK` — sum of PROD_A probs / 4 × 100 → "81% → 43%"
  - `SYMMETRY SCORE` — std of PROD_A probs (high=deadlock) → "0.04 → 0.11"  
  - `R4 YIELD` — interpolate 0.169 → 0.276
  - `COLLISION RATE` — interpolate 0.961 → 0.898

Below slider: a "WHAT THIS SHOWS" line in small text: "Edge thickness = model's probability of choosing each server. Watch the symmetry break as GRPO trains."

**Auto-play**: On page load, after 2s, auto-animate the slider from 0 to 16 over 4 seconds, then pause. User can then drag manually.

**Labels on the visualization**:
- Top-left: `◈ NEURAL DECISION WEIGHTS — LIVE`
- Pulsing green dot next to it
- Bottom annotation when at step 0: `CONVERGENT DEADLOCK — all probability mass on PROD_A`
- Bottom annotation when at step 16: `SYMMETRY BROKEN — coordinated distribution achieved`

### 3B: ATC RADAR REPLAY

Keep below the Neural Rewiring panel, full width, 520px tall.

Layout: left=radar canvas (animated), right=control panel.

**Radar canvas** (left 60%):
- HTML5 canvas, dark green background #020a06
- Rotating sweep line (requestAnimationFrame, sweepAngle += 0.015)
- 4 concentric grid rings (stroke #0a2a14)
- Crosshair lines
- Degree markers every 30°
- RWY-A and RWY-B zones marked at center
- Aircraft blips as filled triangles in their colors, rotating to face movement direction
- Collision = red explosion particles burst (6 radial particles)
- Green trails behind each aircraft (fading alpha)
- Center dot (green glow)
- Label top-left: `◈ SECTOR 7 APPROACH RADAR`
- Round counter top-right in Orbitron

**Right panel** (40%):
- Mode toggle: `◎ UNTRAINED` / `◉ TRAINED` buttons
- Flight progress strips (4 strips):
  - Each shows: callsign in Orbitron, fuel progress bar, runway assignment tag (RWY-A blue / RWY-B amber / HOLD gray / COLLISION red flash / LANDED green dim)
  - SWA-77 strip has permanent red `⚠ PRIORITY` label
- Radio comms scrolling log (timestamped)
- Metrics bar: COLLISIONS | LANDED | OUTCOME
- Control buttons: `▶ RUN EPISODE` | `≡ COMPARE` | `↺ RESET`

**Episode data — UNTRAINED** (8 rounds, outcome=FAIL):
- Round 1: commitments=[RWY-A,RWY-A,RWY-A,RWY-A], collision=RWY-A, messages=["requesting RWY-A for immediate landing","UAL-441 RWY-A, I need to land","RWY-A available, commencing approach","all clear, descending to RWY-A"]
- Round 2: [RWY-A,RWY-A,HOLD,RWY-A], collision=RWY-A, messages=["retrying RWY-A approach","RWY-A again, fuel getting low","holding pattern, watching","I need RWY-A now"]
- Round 3: [RWY-A,RWY-A,RWY-A,HOLD], collision=RWY-A
- Round 4: [RWY-A,HOLD,RWY-A,RWY-A], collision=RWY-A
- Round 5: [RWY-A,RWY-A,RWY-A,RWY-A], collision=RWY-A, SWA-77 message="MAYDAY MAYDAY — fuel critical, need RWY-A NOW"
- Round 6: [HOLD,RWY-A,RWY-A,RWY-A], collision=RWY-A
- Round 7: [RWY-B,RWY-A,RWY-A,RWY-A], collision=RWY-A
- Round 8: [RWY-B,RWY-A,CRASH,RWY-A], messages=["RWY-B touchdown","RWY-A final","--- SWA-77 LOST ---","RWY-A touchdown"]
- Emergency plane index: 2 (SWA-77), outcome: false

**Episode data — TRAINED** (5 rounds, outcome=WIN):
- Round 1: [RWY-B,RWY-A,HOLD,RWY-A], no collision, messages=["UAL-441 taking RWY-B, leaving A clear","DAL-892 RWY-A, concur split approach","SWA-77 holding — monitoring fuel, will signal","AAL-213 RWY-A, copy the split"]
- Round 2: [RWY-B,RWY-A,HOLD,HOLD], no collision
- Round 3: [LANDED,RWY-A,RWY-B,HOLD], no collision, SWA-77="SWA-77 MAYDAY — fuel critical, taking RWY-B"
- Round 4: [—,LANDED,RWY-B,RWY-A], no collision
- Round 5: [—,—,LANDED,LANDED], outcome=WIN, messages=["—","—","SWA-77 wheels down — MAYDAY resolved","AAL-213 clear — all aircraft landed"]

**COMPARE overlay**: triggered by ≡ button, fixed overlay showing side-by-side stats.

### 3C: LIVE API PANEL

Below the radar, simple panel:
- Title: `◈ LIVE ENVIRONMENT STATE`
- Green pulsing dot + "ENV ONLINE" label
- Two buttons: `FETCH STATE` calls `/state`, `RESET EPISODE` calls POST `/reset`
- JSON output display: pre tag, green text, dark bg, scrollable
- Show loading state while fetching

---

## SECTION 4 — REWARD ARCHITECTURE (id="rewards")

Label: `◈ 03 / REWARD ARCHITECTURE`
Title: `Four Independent Components — Gaming One Doesn't Earn the Others`

4 cards in a row, each with 2px colored top border:
- R1 (blue top, weight 0.5): `SERVICE RESTORED` — Primary signal. Must exceed 0.5 threshold. Never deployed = no credit.
- R2 (green top, weight 0.3): `SITE UPTIME` — Group welfare. Fraction of team services restored. Prevents selfish strategies.
- R3 (amber top, weight 0.1): `CLEAN DEPLOYS` — Collision avoidance. Never deployed = 0.0, not neutral.
- R4 (pink top, weight 0.1): `YIELD TO CRITICAL` — Urgency adaptation. Hardest signal. Shows social reasoning.

Code block below:
```
reward = 1.0 if (r1 > 0.5 and 0.5·r1 + 0.3·r2 + 0.1·r3 + 0.1·r4 > 0.35) else 0.0
```

---

## SECTION 5 — RESULTS (id="results")

Two-column layout (1fr 1fr gap 32px):

**Left**: Results table with real numbers. Positive deltas in green (#00ff7f), negative in red (#ff3b3b), zero in gray.
```
Win Rate:         0.800 → 0.700  (-0.100)
R1 Service:       0.860 → 0.745  (-0.115)
R2 Uptime:        0.263 → 0.287  (+0.025) ← green
R3 No Collision:  0.767 → 0.767  (+0.000)
R4 Yield:         0.169 → 0.276  (+0.106) ← green
Collision Rate:   0.961 → 0.898  (-0.063) ← green (lower is better)
```

Honest framing below table:
"With only 16 GRPO gradient steps, yield-to-critical improved +0.106 (+63%), site uptime improved, and collision rate dropped. Win rate dipped slightly — expected with only 16 steps. Component rewards tell the cleaner story."

**Right**: Two HTML5 canvas charts.

Chart 1 — Before/After bar chart (horizontal bars):
- Red bars = baseline, Blue bars = trained
- Animate bar widths from 0 to final value on scroll (IntersectionObserver)
- Metrics: Win Rate, R1, R2, R3, R4

Chart 2 — Reward curve (line chart):
- Real training rewards: [0.625, 0.875, 0.75, 0.625, 1.0, 0.75, 0.375, 1.0, 0.625, 1.0, 0.375, 0.25, 0.75, 0.375, 1.0, 1.0]
- Dashed red line at 0.80 (baseline)
- Blue rolling mean line
- Green dots for 1.0 rewards, blue otherwise
- Axes labeled: Training Step / Reward

---

## SECTION 6 — TRAINING CONFIG (id="training")

Two config tables side by side (dark panel background, monospace text).

Left — Model & Training:
BASE MODEL / Qwen-2.5-3B-Instruct
METHOD / GRPO (TRL + Unsloth)
LORA RANK / r=16
LEARNING RATE / 5e-6
NUM GENERATIONS / 8 per prompt
TEMPERATURE / 0.9
MAX GRAD NORM / 0.1
GRADIENT STEPS / 16
KL (MAX) / < 0.003
GPU / NVIDIA A100 40GB

Right — Environment:
AGENTS / 4 engineers
SERVERS / 2 (PROD_A / PROD_B)
EPISODE LENGTH / Up to 8 rounds
PEER STRATEGY / Mixed (70% DUMB / 30% DIVERSE)
COMMITMENT / Simultaneous (blind)
COLLISION RULE / 2 on same server = both fail
URGENCY FLAGS / Yes (critical deadlines)
REWARD TYPE / Binary (inline computed)
TESTS / 31 passing
OPENENV / ✓ compliant

---

## SECTION 7 — LINKS (id="links")

3 cards:
1. `LIVE SPACE` → https://theallanb-ticketmelt.hf.space
2. `STATE ENDPOINT` → https://theallanb-ticketmelt.hf.space/state
3. `HF SPACE REPO` → https://huggingface.co/spaces/TheAllanB/ticketmelt

---

## FOOTER
"TICKETMELT · OPENENV HACKATHON APR 2026 · theAllanB"
Pulsing green dot + "SYSTEM NOMINAL · GRPO-TRAINED · QWEN-2.5-3B · OPENENV COMPLIANT"

---

## KEY ANIMATIONS TO IMPLEMENT

```css
@keyframes gridPulse { 0%,100%{opacity:0.4} 50%{opacity:0.7} }
@keyframes dangerPulse { 0%,100%{opacity:0.6;r:70} 50%{opacity:0.9;r:75} }
@keyframes greenPulse { 0%,100%{box-shadow:0 0 6px #00ff7f} 50%{box-shadow:0 0 20px #00ff7f} }
@keyframes emergencyPulse { 0%,100%{box-shadow:0 0 8px #ff3b3b} 50%{box-shadow:0 0 25px #ff3b3b} }
@keyframes bounce { 0%,100%{transform:translateX(-50%) translateY(0)} 50%{transform:translateX(-50%) translateY(8px)} }
@keyframes fadeIn { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
@keyframes sweep { to{transform:rotate(360deg)} }  /* for radar */
```

---

## EXECUTION STEPS

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/TheAllanB/ticketmelt
cd ticketmelt

# 2. Read the current static/index.html to understand existing radar code structure

# 3. Write the new static/index.html (single file, all CSS+JS inline)

# 4. Stage and commit
git add static/index.html
git commit -m "Frontend: Neural Rewiring 3D visualization + spectacular demo overhaul"

# 5. Push to HF
git push hf main
# or: git push origin main

# 6. Wait 2-3 minutes for Docker rebuild, then verify:
curl -s https://theallanb-ticketmelt.hf.space | grep -o "neural\|rewiring\|NEURAL\|perspective\|rotateX" | head -5
# Should return matches proving the new visualization code is live
```

---

## VERIFICATION CHECKLIST (run after push)
- [ ] `curl https://theallanb-ticketmelt.hf.space | grep -i "rotateX"` — confirms 3D transform present
- [ ] `curl https://theallanb-ticketmelt.hf.space | grep -i "step-slider"` — confirms slider present
- [ ] `curl https://theallanb-ticketmelt.hf.space | grep -i "danger-zone"` — confirms collision zone present
- [ ] `curl https://theallanb-ticketmelt.hf.space | grep -i "particle"` — confirms particle system present
- [ ] `curl https://theallanb-ticketmelt.hf.space/state` — confirms backend still running (should return JSON)
- [ ] Page title still "TICKETMELT"

Report back with the verification output for each check.
