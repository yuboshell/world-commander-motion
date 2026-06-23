# world-commander-motion

The **v0 de-risk** for the *real-time crowd-motion command* research plan
(`world-commander` repo → `plan/research-plan.md`). It tests the two riskiest
assumptions before any GPU training:

1. a **synthetic-data pipeline** can produce usable (free-form command → coordinated
   crowd motion) triples, and
2. a **controllable motion model** can follow a command in real time.

## The pipeline (three layers)

```
canonical command ──[L1 sim]──▶ coordinated agent trajectories
                   ──[L2 LLM]──▶ free-form / abstract command text
                   ──[L3 model]─▶ full-body motion
            ──▶ (command_text, init_state, targets, trajectory, motion) triples
```

- **L1 (`mca/world.py`, `mca/commands.py`)** — a GPU-free 2D crowd sim (move-to-target
  + separation) renders canonical commands (`go_to`, `form_line`, `regroup`, `disperse`)
  into coordinated trajectories. **Real, runs anywhere.**
- **L2 (`mca/language.py`)** — paraphrase a canonical command into free-form/abstract
  text. `MockParaphraser` is templated (offline); `RealParaphraser` calls a served LLM.
- **L3 (`mca/render.py`)** — trajectory → full-body motion. `StubRenderer` passes the
  path through (offline); `RealRenderer` wraps a pretrained controllable model
  (TLControl / CAMDM) — **fill in on amax** (needs the GPU + weights).

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_v0.py            # offline: mock LLM + stub renderer; validates the pipeline
pytest -q                           # sim / pipeline / metric tests, no GPU
```

Offline output reports, per command, the **grounding** (did the crowd reach the
commanded target) and **collision rate** of the generated data, plus an example
paraphrase — confirming the pipeline structure end to end.

## Next, on amax (per the plan)

1. swap `MockParaphraser` → `RealParaphraser` (the served Qwen) for real free-form variety;
2. swap `StubRenderer` → `RealRenderer` (load TLControl/CAMDM) for full-body motion;
3. add the **model-in-the-loop grounding test**: condition a controllable model on
   `command_text` and measure whether it reproduces the commanded behaviour, plus FPS /
   latency / VRAM (the budget frontier).

amax is the primary experiment box; yubopc (RTX 4060) is the consumer-GPU real-time claim.
