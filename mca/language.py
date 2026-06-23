"""Paraphrase a canonical command into free-form / abstract natural language (data layer L2).
MockParaphraser is offline + templated (for pipeline validation); RealParaphraser calls a
served LLM (fill in on amax) for genuine free-form / abstract variety (拉高文本自由度)."""
from __future__ import annotations

import random
import re

TEMPLATES = {
    "go_to":     ["move to {pt}", "everyone head to {pt}", "push to {pt}", "rally at {pt}"],
    "form_line": ["form a line", "line up", "get into a row", "fall into a line formation"],
    "regroup":   ["regroup", "pull back together", "everyone group up", "consolidate on me"],
    "disperse":  ["spread out", "disperse", "fan out", "scatter and spread"],
    "flank":     ["flank them", "hit both sides", "envelop the target", "split and flank"],
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


_GLOSS = {
    "go_to":     "move/advance the whole group to a location",
    "form_line": "line up in a row between two points",
    "regroup":   "fall back and gather tightly at a rally point",
    "disperse":  "spread out / scatter from the current position",
    "flank":     "split and envelop a target from both sides",
}


def _strip_thinking(text: str) -> str:
    """Remove a Qwen-style <think>...</think> trace and surrounding quotes/markup."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^<think>.*", "", text, flags=re.DOTALL).strip()   # unclosed (truncated) trace
    return text.strip().strip('"').strip("'").strip()


class RealParaphraser:
    """LLM paraphraser over an OpenAI-compatible endpoint (e.g. amax's served Qwen) — L2 of
    the plan, where real free-form / abstract command variety (拉高文本自由度) comes from.

    Reasoning models (Qwen3) emit a <think> trace; we disable it via chat_template_kwargs
    and also strip it defensively, so the reply is just the order."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.9,
                 enable_thinking: bool = False, abstract: bool = True):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.abstract = abstract
        self._calls = 0          # per-call seed -> distinct draws even at one temperature

    def __call__(self, cmd, seed: int | None = None) -> str:
        gloss = _GLOSS.get(cmd.kind, cmd.kind)
        style = ("Sometimes be concrete, sometimes abstract/tactical (e.g. 'we're "
                 "overextended, pull back to cover'). " if self.abstract else "")
        prompt = (
            "Rephrase this crowd order as ONE short, natural command a commander might "
            f"shout to their units. {style}Vary the wording. Reply with only the order, "
            f"no quotes.\nOrder: {gloss}\nDetails: {cmd.params}"
        )
        if seed is None:
            self._calls += 1
            seed = self._calls
        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=64,
            seed=seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        )
        try:
            r = self.client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("extra_body", None)
            r = self.client.chat.completions.create(**kwargs)
        return _strip_thinking(r.choices[0].message.content or "")
