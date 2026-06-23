"""Paraphrase a canonical command into free-form / abstract natural language (data layer L2).
MockParaphraser is offline + templated (for pipeline validation); RealParaphraser calls a
served LLM (fill in on amax) for genuine free-form / abstract variety (拉高文本自由度)."""
from __future__ import annotations

import random

TEMPLATES = {
    "go_to":     ["move to {pt}", "everyone head to {pt}", "push to {pt}", "rally at {pt}"],
    "form_line": ["form a line", "line up", "get into a row", "fall into a line formation"],
    "regroup":   ["regroup", "pull back together", "everyone group up", "consolidate on me"],
    "disperse":  ["spread out", "disperse", "fan out", "scatter and spread"],
}


class MockParaphraser:
    """Templated paraphrases — deterministic-ish, offline, no dependencies."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random(0)

    def __call__(self, cmd) -> str:
        opts = TEMPLATES.get(cmd.kind, [cmd.kind])
        pt = cmd.params.get("point")
        pt_str = tuple(round(float(x), 1) for x in pt) if pt is not None else ""
        return self.rng.choice(opts).format(pt=pt_str)


class RealParaphraser:
    """LLM paraphraser over an OpenAI-compatible endpoint (e.g. amax's served Qwen).

    Fill in on amax: point base_url at the served model. This is where real free-form
    and abstract command variety comes from — the L2 layer of the plan."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.9):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def __call__(self, cmd) -> str:
        prompt = (
            "Rephrase this crowd command as one short, natural order a player might shout "
            "to their units. Vary the wording; abstract phrasing is welcome. "
            f"Reply with only the order.\nCommand: {cmd.kind} {cmd.params}"
        )
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=32,
        )
        return (r.choices[0].message.content or "").strip()
