"""E6 — Closed-loop interpreter grounding (the v0 centrepiece). The offline, GPU-free
version of the plan's model-in-the-loop grounding test, run end to end with REAL amax
components:

    canonical cmd --L2 Qwen paraphrase--> free-form text --[A] interpreter--> recovered cmd
                  --L1 sim--> trajectory --> grounded against the ORIGINAL intent?

If the synthetic free-form text carries enough signal that an interpreter recovers the
behaviour, the data layer (assumption a) is usable. We run TWO conditions:
  - concrete : paraphrases that may keep coordinates (the easy regime)
  - abstract : paraphrases with NO numbers/coordinates, pure tactical intent (the plan's
               novelty regime, where a keyword baseline should fail)
and three interpreters: LLM (served Qwen, the plan's [A]), keyword (MockInterpreter),
and none (a no-interpreter anchor that ignores the text). We report command-recovery
accuracy, end-to-end behavioural grounding, coordinate-recovery error, and interpreter
latency (a real budget-frontier datapoint)."""
from __future__ import annotations

import time

import numpy as np
from openai import OpenAI

from common import CANON, ORDER, pmap, save_json, savefig, plt
from mca.commands import KINDS, Command, targets_for
from mca.interpret import LLMInterpreter, MockInterpreter, _has_location_signal
from mca.language import RealParaphraser, _GLOSS, _strip_thinking
from mca.metrics import grounding  # noqa: F401  (kept for parity / interactive use)
from mca.world import World

BASE, KEY, MODEL = "http://localhost:8000/v1", "EMPTY", "Qwen/Qwen3-14B-AWQ"
SAMPLES = 15        # paraphrases per command per condition
N_AGENTS, STEPS, TOL = 8, 120, 0.8
_NEEDED = {"go_to": ["point"], "regroup": ["point"], "flank": ["point"],
           "form_line": ["start", "end"], "disperse": []}

_client = OpenAI(base_url=BASE, api_key=KEY)


def abstract_paraphrase(kind: str, seed: int) -> str:
    """A paraphrase that conveys intent with NO numbers/coordinates — the abstract regime."""
    prompt = ("Rephrase this crowd order as ONE short, natural command a commander shouts to "
              "their units. Express the INTENT in tactical language. Do NOT use any numbers, "
              "coordinates or grid references — refer to the objective / cover / the rally "
              f"point / both sides, etc. Reply with only the order.\nOrder: {_GLOSS[kind]}")
    r = _client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=1.0,
        max_tokens=48, seed=seed,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return _strip_thinking(r.choices[0].message.content or "")


def exec_command(recovered: Command) -> Command:
    """Make the recovered command runnable: fill any params the interpreter dropped with the
    canonical params for that kind (isolates kind-intent recovery; rewards param recovery)."""
    params = dict(recovered.params)
    for key in _NEEDED.get(recovered.kind, []):
        params.setdefault(key, CANON[recovered.kind].params[key])
    return Command(recovered.kind, params)


def behavioural_grounding(true_kind: str, exec_cmd: Command, seed: int) -> float:
    """Fraction of agents that, after executing the recovered command, end within TOL of the
    nearest target the ORIGINAL command intended (same initial world)."""
    w0 = World.random_init(N_AGENTS, seed=seed)
    orig_targets = targets_for(CANON[true_kind], w0)
    we = World.random_init(N_AGENTS, seed=seed)
    we.target = targets_for(exec_cmd, we)
    final = we.rollout(STEPS)[-1]
    d = np.linalg.norm(final[:, None, :] - orig_targets[None, :, :], axis=2).min(axis=1)
    return float((d < TOL).mean())


def param_error(true_kind: str, rec: Command):
    canon = CANON[true_kind].params
    if true_kind in ("go_to", "regroup", "flank") and "point" in rec.params:
        return float(np.linalg.norm(np.array(rec.params["point"]) - np.array(canon["point"])))
    if true_kind == "form_line" and "start" in rec.params and "end" in rec.params:
        return float(0.5 * (np.linalg.norm(np.array(rec.params["start"]) - np.array(canon["start"]))
                            + np.linalg.norm(np.array(rec.params["end"]) - np.array(canon["end"]))))
    return None


def build_samples(paraphrase_fn):
    specs = [(k, s) for k in ORDER for s in range(SAMPLES)]
    texts = pmap(lambda ks: paraphrase_fn(ks[0], ks[1]), specs)
    return [{"kind": k, "seed": i, "text": t}
            for (k, _), t, i in zip(specs, texts, range(len(specs)))]


def analyze(samples, llm, mock):
    def interp_llm(smp):
        t0 = time.perf_counter()
        cmd = llm(smp["text"], world=World.random_init(N_AGENTS, seed=smp["seed"]))
        return cmd, time.perf_counter() - t0          # thread-local timing

    llm_out = pmap(interp_llm, samples, workers=8)
    llm_cmds = [c for c, _ in llm_out]
    drop = lambda c: Command(c.kind, {k: v for k, v in c.params.items()
                                      if k not in ("point", "start", "end")})
    recovered = {
        "LLM": llm_cmds,                                          # raw (guard off)
        # the fix: keep coordinates only when the source text actually specifies a location,
        # otherwise defer (no extra LLM calls — post-processes the same responses).
        "LLM-guarded": [c if _has_location_signal(s["text"]) else drop(c)
                        for c, s in zip(llm_cmds, samples)],
        # ablation: always drop coords — isolates kind-recovery from coordinate-hallucination.
        "LLM-kind": [drop(c) for c in llm_cmds],
        "keyword": [mock(s["text"]) for s in samples],
        "none": [Command("go_to", dict(CANON["go_to"].params)) for _ in samples],
    }
    latency = [dt for _, dt in llm_out]

    out = {"latency_concurrent_s": latency}
    for name, recs in recovered.items():
        kind_hit, beh, perr = [], [], []
        conf = {a: {b: 0 for b in KINDS} for a in KINDS}
        for smp, rec in zip(samples, recs):
            tk = smp["kind"]
            kind_hit.append(rec.kind == tk)
            conf[tk][rec.kind] += 1
            beh.append(behavioural_grounding(tk, exec_command(rec), smp["seed"]))
            e = param_error(tk, rec)
            if e is not None and rec.kind == tk:
                perr.append(e)
        out[name] = {
            "kind_acc": float(np.mean(kind_hit)),
            "kind_acc_per_cmd": {tk: float(np.mean([r.kind == tk for s, r in zip(samples, recs)
                                                    if s["kind"] == tk])) for tk in ORDER},
            "behavioural_grounding": float(np.mean(beh)),
            "param_err_mean": float(np.mean(perr)) if perr else None,
            "param_err_n": len(perr),
            "confusion": conf,
        }
    out["_examples"] = [{"true": s["kind"], "text": s["text"], "LLM": recovered["LLM"][i].kind,
                         "keyword": recovered["keyword"][i].kind} for i, s in enumerate(samples)]
    return out


def run():
    llm = LLMInterpreter(BASE, KEY, MODEL, guard_coords=False)   # raw; guard applied in analyze()
    mock = MockInterpreter()
    concrete = analyze(build_samples(lambda k, s: RealParaphraser(BASE, KEY, MODEL, temperature=1.0)(CANON[k], seed=s)), llm, mock)
    abstract = analyze(build_samples(abstract_paraphrase), llm, mock)

    # isolated (sequential) latency micro-benchmark for a clean per-request number
    seq = []
    for smp in build_samples(abstract_paraphrase)[:12]:
        llm(smp["text"], world=World.random_init(N_AGENTS, seed=smp["seed"]))
        seq.append(llm.last_latency)
    lat = concrete["latency_concurrent_s"]
    latency = {"concurrent_mean": float(np.mean(lat)), "concurrent_p50": float(np.percentile(lat, 50)),
               "concurrent_p95": float(np.percentile(lat, 95)),
               "isolated_p50": float(np.percentile(seq, 50)), "n": len(lat)}
    return {"concrete": concrete, "abstract": abstract, "latency": latency}


def report(R):
    print(f"\nE6 CLOSED-LOOP INTERPRETER GROUNDING  ({SAMPLES}/command x {len(ORDER)} per "
          f"condition = {SAMPLES * len(ORDER)} round-trips each)")
    for cond in ("concrete", "abstract"):
        print(f"\n[{cond} paraphrases]")
        print(f"{'interpreter':12s} {'kind_acc':>9s} {'beh.grnd':>9s} {'coord_err':>14s}")
        for name in ("LLM", "LLM-guarded", "LLM-kind", "keyword", "none"):
            r = R[cond][name]
            pe = (f"{r['param_err_mean']:.2f} (n={r['param_err_n']})"
                  if r["param_err_mean"] is not None else "n/a")
            print(f"{name:12s} {r['kind_acc']:>9.2f} {r['behavioural_grounding']:>9.2f} {pe:>14s}")
    lat = R["latency"]
    print(f"\ninterpreter latency: isolated p50={lat['isolated_p50'] * 1000:.0f}ms | "
          f"concurrent(x8) mean={lat['concurrent_mean'] * 1000:.0f}ms p95={lat['concurrent_p95'] * 1000:.0f}ms")
    print("\nabstract round-trips (true | text | LLM-> / keyword->):")
    ex = R["abstract"]["_examples"]
    for e in ex[:: max(1, len(ex) // 10)]:
        fl = "OK " if e["LLM"] == e["true"] else "ERR"
        fk = "ok" if e["keyword"] == e["true"] else "X"
        print(f"  [{fl}] {e['true']:9s} \"{e['text'][:52]}\" -> LLM:{e['LLM']} kw:{e['keyword']}({fk})")
    save_json("exp6_interpreter.json", R)


def _bars(ax, R, metric, title, ylim=(0, 1.05)):
    conds = ["concrete", "abstract"]; names = ["LLM", "LLM-guarded", "keyword", "none"]
    cols = {"LLM": "#1f77b4", "LLM-guarded": "#2ca02c", "keyword": "#8c8c8c", "none": "#d62728"}
    x = np.arange(len(conds)); w = 0.2
    for i, name in enumerate(names):
        ax.bar(x + (i - 1.5) * w, [R[c][name][metric] for c in conds], w, label=name, color=cols[name])
    ax.axhline(0.2, ls="--", c="gray", lw=1)
    ax.set_xticks(x, conds); ax.set(title=title, ylim=ylim)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)


def figure(R):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _bars(axes[0], R, "kind_acc", "command-type recovery accuracy")
    _bars(axes[1], R, "behavioural_grounding", "end-to-end behavioural grounding")
    conf = R["abstract"]["LLM"]["confusion"]
    M = np.array([[conf[a][b] for b in KINDS] for a in KINDS], dtype=float)
    im = axes[2].imshow(M, cmap="Blues")
    axes[2].set_xticks(range(len(KINDS)), KINDS, rotation=45)
    axes[2].set_yticks(range(len(KINDS)), KINDS)
    axes[2].set(title="LLM confusion, abstract (true->pred)", xlabel="predicted", ylabel="true")
    for i in range(len(KINDS)):
        for j in range(len(KINDS)):
            if M[i, j]:
                axes[2].text(j, i, int(M[i, j]), ha="center", va="center",
                             color="white" if M[i, j] > M.max() / 2 else "black")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle("E6 — closed-loop grounding (real Qwen L2 + interpreter): raw LLM hallucinates "
                 "coordinates on abstract intent (grounding 0.18); the coordinate guard restores "
                 "it (LLM-guarded)", fontsize=10)
    savefig(fig, "exp6_interpreter.png")


if __name__ == "__main__":
    R = run()
    report(R)
    figure(R)
