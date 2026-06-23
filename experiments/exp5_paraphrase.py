"""E5 — L2 text freedom. Compare the offline templated MockParaphraser against the served
Qwen RealParaphraser on lexical diversity. The plan's premise is that the LLM is where real
free-form / abstract command variety (拉高文本自由度) comes from; this quantifies it
(distinct-n, type-token ratio, unique-paraphrase ratio, length) and surfaces examples."""
from __future__ import annotations

import re

from common import CANON, ORDER, pmap, save_json, savefig, plt
from mca.language import MockParaphraser, RealParaphraser

BASE, KEY, MODEL = "http://localhost:8000/v1", "EMPTY", "Qwen/Qwen3-14B-AWQ"
K = 20            # paraphrases per command per source


def _toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _mattr(tokens, window=10):
    """Moving-average type-token ratio: lexical diversity that, unlike raw TTR/distinct-n,
    is robust to corpus length (critical here — Qwen text is ~4x longer than templates)."""
    if len(tokens) <= window:
        return len(set(tokens)) / max(len(tokens), 1)
    ttrs = [len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]
    return sum(ttrs) / len(ttrs)


def diversity(corpus):
    toks = [t for s in corpus for t in _toks(s)]
    return {
        "n_samples": len(corpus),
        "n_unique": len(set(corpus)),
        "unique_ratio": len(set(corpus)) / len(corpus),
        "mattr": _mattr(toks, 10),
        "vocab": len(set(toks)),
        "mean_len": sum(len(_toks(s)) for s in corpus) / len(corpus),
    }


def gen():
    mock = MockParaphraser()
    real = RealParaphraser(BASE, KEY, MODEL, temperature=1.0)
    mock_corpus, real_corpus, examples = {}, {}, {}
    for kind in ORDER:
        cmd = CANON[kind]
        mock_corpus[kind] = [mock(cmd) for _ in range(K)]
        real_corpus[kind] = pmap(lambda s: real(cmd, seed=s), list(range(K)))
        examples[kind] = {"mock": sorted(set(mock_corpus[kind])),
                          "real": real_corpus[kind][:8]}
        print(f"  {kind}: mock {len(set(mock_corpus[kind]))} uniq / real "
              f"{len(set(real_corpus[kind]))} uniq")
    return mock_corpus, real_corpus, examples


def report(mock_corpus, real_corpus, examples):
    flat = lambda d: [s for kind in ORDER for s in d[kind]]
    dm, dr = diversity(flat(mock_corpus)), diversity(flat(real_corpus))
    print(f"\nE5 PARAPHRASE DIVERSITY  ({K}/command x {len(ORDER)} commands)")
    print(f"{'metric':16s} {'Mock(tmpl)':>12s} {'Real(Qwen)':>12s}")
    for k in ["n_unique", "unique_ratio", "mattr", "vocab", "mean_len"]:
        print(f"{k:16s} {dm[k]:>12.2f} {dr[k]:>12.2f}")
    print(f"(note: Mock draws from {len(ORDER) * 4} fixed templates -> unique_ratio ceiling "
          f"~{4 / K:.2f}; MATTR is length-robust, unlike raw TTR/distinct-n.)")
    print("\nexample Qwen paraphrases (free-form / abstract):")
    for kind in ORDER:
        for ex in examples[kind]["real"][:3]:
            print(f"  [{kind:9s}] {ex}")
    save_json("exp5_paraphrase.json", {"mock": dm, "real": dr, "examples": examples})
    return dm, dr


def figure(dm, dr):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    keys = ["unique_ratio", "mattr"]
    x = range(len(keys)); w = 0.38
    axes[0].bar([i - w / 2 for i in x], [dm[k] for k in keys], w, label="Mock (templated)", color="#8c8c8c")
    axes[0].bar([i + w / 2 for i in x], [dr[k] for k in keys], w, label="Real (Qwen)", color="#1f77b4")
    axes[0].set_xticks(list(x), ["unique\nparaphrase ratio", "MATTR\n(length-robust)"])
    axes[0].set(title="diversity (higher = freer text)", ylim=(0, 1.05))
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
    for ax, key, title in [(axes[1], "vocab", "vocabulary size (unique words)"),
                           (axes[2], "mean_len", "mean command length (words)")]:
        vals = [dm[key], dr[key]]
        ax.bar(["Mock", "Real"], vals, color=["#8c8c8c", "#1f77b4"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}" if key == "vocab" else f"{v:.1f}", ha="center", va="bottom")
        ax.set(title=title); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"E5 — L2 paraphrase diversity: templated vs served Qwen ({K}/command)", fontsize=11)
    savefig(fig, "exp5_paraphrase.png")


if __name__ == "__main__":
    mc, rc, ex = gen()
    dm, dr = report(mc, rc, ex)
    figure(dm, dr)
