# world-commander-motion — agent notes

The **v0 de-risk** for the real-time crowd-motion-command research plan. Goal: prove the
synthetic-data pipeline yields usable (free-form command -> coordinated motion) triples and
that a controllable motion model can follow a command in real time — before any GPU training.

## What's here
- `mca/world.py`, `mca/commands.py` — **L1**: GPU-free 2D crowd sim; canonical commands
  (`go_to`, `form_line`, `regroup`, `disperse`) -> coordinated trajectories. **Real.**
- `mca/language.py` — **L2**: canonical -> free-form text. `MockParaphraser` (offline,
  templated) and `RealParaphraser` (served LLM — fill in on amax).
- `mca/render.py` — **L3**: trajectory -> full-body motion. `StubRenderer` (offline
  passthrough) and `RealRenderer` (TLControl / CAMDM — **fill in on amax**, needs GPU).
- `mca/generate.py` — orchestrates L1->L2->L3 into triples.
- `mca/metrics.py` — grounding, collision rate (the measurement layer).
- `scripts/run_v0.py` — offline smoke run; `tests/` — no-GPU tests.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_v0.py
pytest -q
```

## The contract
- The **core (L1 + metrics) stays GPU-free** so it runs and tests anywhere. Motion-model
  (torch) deps live only behind `RealRenderer` and install on the GPU box — never add them
  to `requirements.txt`.
- **amax41 is the primary experiment box**; yubopc (RTX 4060) is the consumer-GPU real-time
  claim. The served LLM on amax backs `RealParaphraser`.
- The plan of record is `plan/research-plan.md` in the **world-commander** repo (data
  layers, model, training stages, eval, risks). Keep this repo's structure matching it.

## Next steps (the v0 → v1 path)
1. on amax: `RealParaphraser` (served Qwen) for genuine free-form / abstract commands;
2. on amax: `RealRenderer` (load TLControl/CAMDM) -> full-body motion;
3. the **model-in-the-loop grounding test** — condition a controllable model on
   `command_text`, measure whether it reproduces the behaviour, plus FPS / latency / VRAM;
4. then v1: the coordination layer (rule-based -> RL/MARL) and the budget frontier.

## Status
- Proposal/prototype stage. Offline pipeline only; `RealRenderer` not yet implemented.
- Remote: GitLab (GitHub account suspended). No build beyond the smoke run + tests.
- **v0 experiment session (2026-06-23, on amax41) — see `experiments/REPORT.md`.** Done:
  `RealParaphraser` fixed + run on the served Qwen (L2); the interpreter [A] built
  (`mca/interpret.py`) with a coordinate guard; the model-in-the-loop grounding test run in
  its GPU-free form (text→command→sim) — concrete grounding 0.88→0.96, abstract 0.18→0.74
  with the guard; added the `flank` command + `formation_error`/`completion_time` metrics; E1–E6
  + viz + an end-to-end `demo.py`. Still open: `RealRenderer` (no TLControl/CAMDM weights on
  box) and the coordination layer [B]. Changes are uncommitted (in the working tree).
