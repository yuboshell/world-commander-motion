#!/usr/bin/env python
"""Assemble the v0 report as ONE self-contained report.html — same convention as
world-commander-bench (scripts/build_report.py + arena/viz.py): embedded images, an
interactive replay viewer, sections of intro -> figure -> table -> caption, metric
definitions, and a members-only/noindex private page. Numbers are read from the JSONs
the experiments wrote (experiments/results/*.json); figures are reused from figures/.

    python experiments/build_report.py            # writes experiments/report.html
    python experiments/build_report.py --publish  # also copy -> ../world-commander-bench/motion.html

The report is published as "Crowd Motion (E4)" on the shared world-commander-bench Pages hub:
--publish copies it to that repo's motion.html; commit + push the bench repo (gitlab remote) to
deploy. For a local preview use serve_report.sh (127.0.0.1 + SSH tunnel) — never a public host
(public Pages publishing is retired; it caused a GitHub suspension)."""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np

from common import CANON, COLORS, ORDER, RESULTS, FIGS, ROOT, simulate, plt

OUT = ROOT / "report.html"
TITLE = "World Commander — Crowd Motion (v0 de-risk)"
META = {
    "Machine": "amax41 (3× RTX 2080 Ti)",
    "Served model": "Qwen/Qwen3-14B-AWQ via vLLM (OpenAI-compatible, localhost:8000/v1)",
    "Pipeline": "L1 NumPy crowd sim (real) · L2 served-Qwen paraphraser · [A] served-Qwen "
                "interpreter · L3 OmniControl (trajectory → full-body motion, GPU 2)",
    "Note": "the served Qwen is shared, so latency carries contention noise; L1/metrics are "
            "GPU-free and deterministic.",
}


def J(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def uri(path: Path) -> str:
    mime = "image/gif" if path.suffix == ".gif" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


# ----------------------------------------------------------------------------- replay frames
def crowd_frames(kind, n=10, steps=80, seed=7, stride=2) -> list[str]:
    """Render the L1 rollout for one command to a list of PNG data: URIs (the replay)."""
    tr = simulate(CANON[kind], n=n, steps=steps, seed=seed)
    traj, tgt = tr["trajectory"], tr["targets"]
    cmap = plt.cm.viridis(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(4.3, 4.5))
    uris = []
    for t in range(0, len(traj), stride):
        ax.clear()
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
        ax.grid(True, color="0.9"); ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(0, 11, 2))
        ax.scatter(tgt[:, 0], tgt[:, 1], marker="x", color="crimson", s=45, alpha=0.7, zorder=2)
        lo = max(0, t - 20)
        for i in range(n):
            ax.plot(traj[lo:t + 1, i, 0], traj[lo:t + 1, i, 1], "-", color=cmap[i], lw=1.0, alpha=0.5)
        ax.scatter(traj[t, :, 0], traj[t, :, 1], color=cmap, s=70, edgecolors="black", lw=0.5, zorder=3)
        ax.set_title(f'{kind}  ·  step {t}/{len(traj) - 1}', fontsize=10)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=88, bbox_inches="tight")
        uris.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    plt.close(fig)
    return uris


# ----------------------------------------------------------------------------- tables
def _table(headers, rows):
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>\n<tr>{h}</tr>\n{body}</table>"


def _gallery_html(items):
    """items = [(Path, label), ...] -> a responsive grid of embedded images (e.g. motion GIFs)."""
    cells = "".join(f'<figure><img src="{uri(p)}"><figcaption>{lab}</figcaption></figure>'
                    for p, lab in items)
    return f'<div class="gallery">{cells}</div>'


def t_coverage():
    d = J("exp1_coverage")
    rows = [[k, f"{d[k]['grounding'][0]:.2f}", f"{d[k]['collision'][0]:.2f}",
             f"{d[k]['formation_error'][0]:.2f}", f"{d[k]['completion_sec'][0]:.2f} s"] for k in ORDER]
    return _table(["command", "grounding", "collision rate", "formation err", "completion"], rows)


def t_scaling():
    d = J("exp3_scaling")
    want = [5, 12, 30, 50, 100, 200]
    idx = {n: d["N"].index(n) for n in want if n in d["N"]}
    rows = [[n, f"{d['grounding'][i]:.2f}", f"{d['collision'][i]:.2f}", f"{d['steps_per_s'][i]:,.0f} steps/s"]
            for n, i in idx.items()]
    return _table(["N agents", "grounding", "collision rate", "sim throughput"], rows)


def t_frontier():
    d = J("exp4_frontier")
    ss, sr, c, g = d["closest_to_ideal"]
    df = d["default"]
    rows = [[f"closest to ideal (sep_strength={ss:.2f}, sep_radius={sr:.2f})", f"{c:.2f}", f"{g:.2f}",
             f"{d['gap_to_ideal']:.2f}"],
            [f"repo default (sep_strength={df['params'][0]:.2f}, sep_radius={df['params'][1]:.2f})",
             f"{df['collision']:.2f}", f"{df['grounding']:.2f}", "—"]]
    return _table(["operating point", "collision rate", "grounding", "gap to ideal"], rows)


def t_paraphrase():
    d = J("exp5_paraphrase"); m, r = d["mock"], d["real"]
    rows = [["unique-paraphrase ratio", f"{m['unique_ratio']:.2f}", f"{r['unique_ratio']:.2f}"],
            ["MATTR (length-robust)", f"{m['mattr']:.2f}", f"{r['mattr']:.2f}"],
            ["vocabulary (unique words)", f"{m['vocab']:.0f}", f"{r['vocab']:.0f}"],
            ["mean length (words)", f"{m['mean_len']:.1f}", f"{r['mean_len']:.1f}"]]
    return _table(["metric", "Mock (templated)", "Real (Qwen)"], rows)


def t_interpreter():
    d = J("exp6_interpreter")
    rows = []
    for cond in ("concrete", "abstract"):
        for name in ("LLM", "LLM-guarded", "keyword", "none"):
            r = d[cond][name]
            pe = (f"{r['param_err_mean']:.2f}" if r["param_err_mean"] is not None else "—")
            label = name if name != "LLM-guarded" else "<b>LLM-guarded</b>"
            rows.append([cond, label, f"{r['kind_acc']:.2f}", f"{r['behavioural_grounding']:.2f}", pe])
    return _table(["paraphrase", "interpreter", "kind accuracy", "behavioural grounding", "coord err"], rows)


def t_l3():
    d = J("l3_motion"); pc = d["per_command"]
    rows = [[k, pc[k]["frames"], f"{pc[k]['gen_s']:.0f} s", f"{pc[k]['throughput_fps']:.1f}",
             f"{pc[k]['peak_vram_gb']:.2f}", f"{pc[k]['realtime_factor']:.0f}×"] for k in ORDER]
    return _table(["command", "frames", "gen latency", "throughput (fps)", "peak VRAM (GB)",
                   "slower than real-time"], rows)


# ----------------------------------------------------------------------------- sections
def sections():
    return [
        {"title": "E1 — Coverage: every canonical command grounds", "png": FIGS / "exp1_coverage.png",
         "intro": "<p>All five canonical commands (<code>go_to · form_line · regroup · disperse · "
         "flank</code> — the last added this run to match the plan) are pushed through the L1 sim "
         "and scored on the full v0 measurement layer, over 24 random initialisations each.</p>",
         "table": t_coverage(),
         "caption": "Grounding is ≈<b>1.0</b> everywhere and commands complete in <b>0.6–2.1 s</b>. "
         "The one wart is <b>regroup's collision rate (0.62)</b>: tight gathering packs agents — an "
         "early signal of the coordination tension E3/E4 quantify."},

        {"title": "E2 — Convergence: how fast a command is satisfied", "png": FIGS / "exp2_convergence.png",
         "intro": "<p>One long rollout per command, grounding/formation-error read off at many "
         "intermediate horizons (16 inits). The knee of each curve is the effective completion time — "
         "a coordination/real-time-budget quantity.</p>",
         "table": None,
         "caption": "Every command reaches 90% grounding within <b>0.7–1.8 s</b> of sim time "
         "(disperse fastest, go_to/flank slowest). Formation error decays to near zero; the L1 "
         "dynamics are fast and stable."},

        {"title": "E3 — Scaling: density and the O(N²) ceiling", "png": FIGS / "exp3_scaling.png",
         "intro": "<p>Hold the 10×10 world fixed and grow the crowd (averaged over all commands). "
         "Coordination metrics and the sim's own throughput are both tracked.</p>",
         "table": t_scaling(),
         "caption": "Two ceilings appear: <b>density</b> — collisions saturate to ~1.0 by ~50 agents "
         "and grounding decays as the world crowds; and <b>compute</b> — throughput falls ~O(N²) (the "
         "all-pairs separation step), so the naive sim needs spatial hashing before large crowds."},

        {"title": "E4 — Coordination frontier: collision vs grounding", "png": FIGS / "exp4_frontier.png",
         "intro": "<p>Sweep the two separation parameters at a fixed density and map collision rate "
         "and grounding. Weak separation grounds but collides; strong separation avoids collisions but "
         "pushes agents off-target.</p>",
         "table": t_frontier(),
         "caption": "There is <b>no setting that is both collision-free and grounded</b> — a genuine "
         "Pareto tradeoff, closest approach to the ideal corner still <b>0.21 away</b>. This gap is "
         "exactly what the v1 coordination layer (RVO/ORCA → RL/MARL) must close; it is not a grounding "
         "problem."},

        {"title": "E5 — L2 text freedom: served Qwen vs templates", "png": FIGS / "exp5_paraphrase.png",
         "intro": "<p>The plan's premise is that the LLM is where real free-form / abstract command "
         "variety comes from. We compare the templated <code>MockParaphraser</code> against the served "
         "Qwen on length-robust diversity (20 paraphrases × 5 commands).</p>",
         "table": t_paraphrase(),
         "caption": "Qwen is far freer — <b>3× more unique phrasings</b>, 2× vocabulary, 3.4→11 words — "
         "and produces genuinely abstract orders (<i>“take up firing positions on your own initiative”</i>, "
         "<i>“converge at the breach point”</i>). The templated mock is capped at its 20-template ceiling."},

        {"title": "E6 — Closed-loop grounding: the v0 centrepiece", "png": FIGS / "exp6_interpreter.png",
         "intro": "<p>The GPU-free version of the plan's model-in-the-loop grounding test, run end to "
         "end with <b>real amax components</b>: canonical → <b>Qwen paraphrase</b> (L2) → free-form text "
         "→ <b>Qwen interpreter</b> [A] → sim (L1) → grounded against the original intent? Two paraphrase "
         "regimes (concrete = coords allowed; abstract = no numbers, pure intent) × four interpreters "
         "(LLM, LLM-guarded, keyword baseline, no-interpreter anchor).</p>",
         "table": t_interpreter(),
         "caption": "Command <b>type</b> recovers from free-form text (0.80–0.97, vs 0.20 chance) — the "
         "data layer is usable. The failure is <b>spatial</b>: on abstract orders the raw LLM invents "
         "coordinates for landmarks it cannot see (coord err <b>3.17</b>), collapsing grounding to "
         "<b>0.18</b>. A coordinate guard — accept coordinates only when the text specifies one — "
         "restores abstract grounding to <b>0.74</b> and even lifts concrete to <b>0.96</b>. So the "
         "abstract-intent bottleneck is grounding the <i>location</i>, not reading the <i>intent</i> "
         "(plan risk #2, now with a working v0 mitigation)."},

        {"title": "L3 — real full-body motion (OmniControl): the trajectory becomes a person",
         "png": FIGS / "motion_montage.png",
         "intro": "<p>The final layer, now real (not the stub). Feed each command's per-agent path into "
         "a pretrained, trajectory-controllable motion model (<b>OmniControl</b>, HumanML3D) as the "
         "pelvis spatial-control signal, and it generates <b>full-body motion that walks the path</b> — "
         "on one RTX 2080 Ti (GPU 2). The L1 dots stay as the honest coordination view; this is the same "
         "trajectories rendered as people (green = the commanded pelvis path).</p>",
         "gallery": [(FIGS / f"motion_{k}.gif", k) for k in ORDER],
         "table": t_l3(),
         "caption": "<b>Assumption (b), half-validated.</b> A controllable model does turn the commanded "
         "trajectory into plausible full-body motion, and it is <b>VRAM-cheap (~0.57 GB)</b> — it fits a "
         "consumer GPU with room to spare. But generation is diffusion + spatial guidance: <b>~153 s for "
         "a 9.8 s clip, ~16× slower than real time</b> (1.3 fps). So the real-time bottleneck is "
         "<b>latency, not memory</b> — exactly what a distilled / fast model (MotionLCM ~30 ms, the "
         "efficiency line) must close. This is the budget-frontier datapoint the plan calls for."},

        {"title": "Capstone — the whole pipeline end to end", "png": FIGS / "demo_montage.png",
         "intro": "<p><code>experiments/demo.py</code> runs a typed free-form order through the entire "
         "stack with the real LLM: text → guarded Qwen interpreter → canonical command → sim → motion. "
         "Try your own: <code>python experiments/demo.py \"hold the left edge\"</code>.</p>",
         "table": None,
         "caption": "Five orders (one concrete, four abstract) all interpret correctly and execute at "
         "grounding 1.00, 0.4–1.0 s each — including <i>“fall back and regroup at the centre”</i> "
         "(the guard defers the coordinate; the canonical centre is right) and <i>“swing wide and "
         "envelop them from both flanks”</i> → <code>flank</code>."},
    ]


# ----------------------------------------------------------------------------- assemble
def main():
    ap = argparse.ArgumentParser(description="build the self-contained crowd-motion report.html")
    ap.add_argument("--publish", action="store_true",
                    help="also copy report.html into the bench repo as motion.html (the shared hub)")
    publish = ap.parse_args().publish
    generated = time.strftime("%Y-%m-%d, %I:%M %p %Z")
    replays = {k: crowd_frames(k) for k in ORDER}
    print(f"  rendered replay frames for {len(replays)} commands")
    replays_js = "{\n" + ",\n".join(
        f'"{k}": [' + ",".join(f'"{u}"' for u in frs) + "]" for k, frs in replays.items()) + "\n}"

    meta_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in META.items())
    e6 = J("exp6_interpreter"); lat = e6["latency"]
    summary = (
        f"5 canonical commands · grounding ≈<b>1.0</b> (E1) · closed-loop command recovery "
        f"<b>{e6['concrete']['LLM']['kind_acc']:.2f}</b> and end-to-end grounding "
        f"<b>{e6['concrete']['LLM-guarded']['behavioural_grounding']:.2f}</b> through the real Qwen "
        f"(E6) · interpreter latency p50 <b>{lat['isolated_p50'] * 1000:.0f} ms</b> · L3 real full-body "
        f"motion via OmniControl (VRAM <b>{J('l3_motion')['aggregate']['peak_vram_gb']:.2f} GB</b>, "
        f"~{J('l3_motion')['aggregate']['mean_realtime_factor']:.0f}× slower than real-time)")

    sec_html = ""
    for s in sections():
        img = f'<img class="metrics" src="{uri(s["png"])}" alt="{s["title"]}">\n' if s.get("png") else ""
        gal = _gallery_html(s["gallery"]) + "\n" if s.get("gallery") else ""
        sec_html += (f"<h2>{s['title']}</h2>\n{s.get('intro','')}\n{gal}{img}"
                     f"{s.get('table') or ''}\n<p class=\"hint\">{s['caption']}</p>\n")

    defs = [
        ("Grounding accuracy", "fraction of agents within a tolerance of their commanded target at the "
         "final frame (does the crowd reach what was ordered)."),
        ("Collision rate", "fraction of timesteps with at least one agent-pair closer than a radius "
         "(coordination quality)."),
        ("Formation error", "RMS distance of agents from their assigned targets at the final frame "
         "(how tight the realised formation is)."),
        ("Completion time", "first frame at which ≥90% of agents are grounded, in steps (÷30 fps → "
         "seconds)."),
        ("Command recovery / kind accuracy", "the interpreter returns the correct canonical command "
         "type from free-form text (the core decodability test)."),
        ("Behavioural grounding", "execute the <i>recovered</i> command and measure the fraction of "
         "agents that end within tolerance of the <i>original</i> intent's nearest target — the true "
         "end-to-end score."),
        ("Interpreter latency", "wall-clock per served-LLM interpret call (a real budget-frontier "
         "datapoint for component [A])."),
        ("L3 generation latency / VRAM", "wall-clock to generate one 196-frame (9.8 s) full-body clip "
         "from a trajectory, and peak GPU memory — the motion-model budget; 'slower than real-time' = "
         "generation time ÷ clip length."),
    ]
    defs_html = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in defs)
    opts = "".join(f'<option value="{k}">{k}</option>' for k in ORDER)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="robots" content="noindex, nofollow">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 920px;
         color: #1a1a1a; line-height: 1.55; padding: 0 1rem; }}
  h1 {{ font-size: 1.5rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.12rem; margin-top: 2.4rem; border-bottom: 1px solid #eee; padding-bottom: .2rem; }}
  .summary {{ background:#f3f5f7; border-radius:8px; padding:.7rem 1rem; font-size:1rem; }}
  table {{ border-collapse: collapse; margin:.6rem 0; font-size:.92rem; }}
  th, td {{ border:1px solid #e3e3e3; padding:.35rem .65rem; text-align:left; vertical-align:top; }}
  th {{ background:#f7f8fa; }}
  dl {{ margin:.4rem 0; }} dt {{ font-weight:600; margin-top:.5rem; }} dd {{ margin:0 0 .2rem 1rem; }}
  img.metrics {{ width:100%; border:1px solid #e2e2e2; border-radius:6px; }}
  .gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(165px,1fr)); gap:12px; margin:12px 0; }}
  .gallery figure {{ margin:0; }} .gallery img {{ width:100%; border:1px solid #e2e2e2; border-radius:6px; }}
  .gallery figcaption {{ text-align:center; font-size:.82rem; color:#666; margin-top:2px; }}
  .viewer {{ text-align:center; }}
  #frame {{ width:430px; max-width:100%; border:1px solid #e2e2e2; border-radius:6px; }}
  .controls {{ display:flex; gap:.5rem; align-items:center; justify-content:center;
               flex-wrap:wrap; margin:.7rem 0; }}
  button, select {{ font-size:1rem; padding:.3rem .6rem; cursor:pointer; }}
  input[type=range] {{ width:60%; }}
  .hint {{ color:#666; font-size:.86rem; }}
  footer {{ margin-top:3rem; color:#777; font-size:.82rem; border-top:1px solid #eee; padding-top:.6rem; }}
</style></head>
<body>
<div style="font-size:.9rem;color:#666;border-bottom:1px solid #eee;padding-bottom:.6rem;margin-bottom:1rem">Reports: <a href="index.html">Grid Arena (E1)</a> &middot; <a href="sc2.html">StarCraft II (E2)</a> &middot; <a href="embodiment.html">Embodiment (E3)</a> &middot; <b>Crowd Motion (E4)</b></div>
<h1>{TITLE}</h1>
<p class="hint">Free-form natural-language command → coordinated crowd motion, synthetic-data pipeline.
Tests the two riskiest assumptions before any GPU training.</p>
<p class="hint"><b>Updated:</b> {generated} &middot; members-only</p>

<table>{meta_rows}</table>

<p>Below is the L1 crowd executing each canonical command (target = red ×). <b>Pick a command and
press Play</b> (or drag the slider), then read on for what is measured.</p>

<h2>Replay <span class="hint">(choose a command; drag the slider or press Play)</span></h2>
<div class="viewer">
  <img id="frame" alt="crowd frame">
  <div class="controls">
    <label>command <select id="cmd">{opts}</select></label>
    <button id="prev">◀ Prev</button>
    <button id="play">▶ Play</button>
    <button id="next">Next ▶</button>
    <label>speed
      <select id="speed">
        <option value="120">slow</option>
        <option value="60" selected>1×</option>
        <option value="30">fast</option>
      </select>
    </label>
  </div>
  <input id="slider" type="range" min="0" value="0">
  <div class="hint">frame <span id="idx">0</span> / <span id="total">0</span></div>
</div>

<h2>What this is</h2>
<p>The <b>v0 de-risk</b> for real-time crowd-motion command. A canonical command drives a GPU-free 2D
crowd sim (<b>L1</b>) into coordinated trajectories; a served LLM paraphrases it into free-form text
(<b>L2</b>); an LLM interpreter (<b>[A]</b>) maps text back to a command; metrics score the loop. It
tests the two riskiest assumptions (research-plan §7): <b>(a)</b> the synthetic pipeline yields usable
data, and <b>(b)</b> a controllable model follows a command in real time. This run attacks <b>(a)</b>
end to end with the real served Qwen, and gets <b>(b)</b> half-way: real full-body motion now follows
each trajectory (L3, OmniControl on a 2080 Ti) — VRAM-cheap but not yet real-time (see L3 below).</p>

<h2>Results at a glance</h2>
<div class="summary">{summary}</div>

<h2>Metrics — what each number means</h2>
<dl>{defs_html}</dl>

{sec_html}

<h2>De-risked vs. remaining</h2>
<p><b>De-risked:</b> L1 grounds all 5 commands; the served-Qwen L2 paraphraser works (after a real
bugfix for reasoning-model output) and adds genuine variety; the data layer is usable for concrete
commands end to end through real LLM components; the coordinate guard mitigates abstract
hallucination (0.18→0.74); and <b>L3 now turns each trajectory into real full-body motion</b>
(OmniControl) at ~0.57 GB VRAM. <b>Remaining:</b> <b>real-time</b> motion — the generator is ~16×
too slow, so distill / swap for a fast model (MotionLCM) to finish assumption (b); principled abstract
spatial grounding (<code>resolve(landmark, world_state)</code> / reward-from-language); the
coordination layer [B] (RVO/ORCA → MARL) to close the E4 gap; spatial hashing for the O(N²) sim.</p>

<script>
const REPLAYS = {replays_js};
const order = {json.dumps(ORDER)};
const img = document.getElementById('frame'), slider = document.getElementById('slider');
const idx = document.getElementById('idx'), total = document.getElementById('total');
const playBtn = document.getElementById('play'), speed = document.getElementById('speed');
const cmdSel = document.getElementById('cmd');
let frames = REPLAYS[order[0]], i = 0, timer = null;
function load(name) {{ stop(); frames = REPLAYS[name]; slider.max = frames.length - 1;
  total.textContent = frames.length - 1; show(0); }}
function show(n) {{ i = (n + frames.length) % frames.length; img.src = frames[i];
  slider.value = i; idx.textContent = i; }}
function stop() {{ if (timer) {{ clearInterval(timer); timer = null; playBtn.textContent = '▶ Play'; }} }}
function play() {{ stop(); timer = setInterval(() => {{ if (i >= frames.length - 1) {{ stop(); return; }}
  show(i + 1); }}, parseInt(speed.value, 10)); playBtn.textContent = '⏸ Pause'; }}
document.getElementById('prev').onclick = () => {{ stop(); show(i - 1); }};
document.getElementById('next').onclick = () => {{ stop(); show(i + 1); }};
playBtn.onclick = () => timer ? stop() : play();
slider.oninput = () => {{ stop(); show(parseInt(slider.value, 10)); }};
cmdSel.onchange = () => load(cmdSel.value);
speed.onchange = () => {{ if (timer) play(); }};
load(order[0]);
</script>

<footer>
World Commander — Crowd Motion track, v0 de-risk. L1/metrics are real GPU-free measurements; L2/[A]
run on the served Qwen. Self-contained report (images + replay frames embedded); regenerate with
<code>python experiments/build_report.py</code>. Full prose + reproduction in
<code>experiments/REPORT.md</code>.
</footer>
</body></html>"""
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT.parent)}  ({len(html) / 1e6:.1f} MB)")
    if publish:
        dest = ROOT.parent.parent / "world-commander-bench" / "motion.html"
        if dest.parent.exists():
            dest.write_bytes(OUT.read_bytes())
            print(f"published -> {dest}\n  next: (cd {dest.parent} && git add motion.html && "
                  "git commit -m 'update motion report' && git push gitlab main)")
        else:
            print(f"--publish: bench repo not found at {dest.parent} (skipped)")


if __name__ == "__main__":
    main()
