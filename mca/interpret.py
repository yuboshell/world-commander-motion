"""Command interpreter (system component [A] of the research plan): free-form / abstract
command text + world state -> a canonical Command (kind + params).

This is the *inverse* of the L2 paraphraser and the thing the v0 grounding test actually
exercises: if the synthetic free-form text carries enough signal that an interpreter can
recover the behaviour, the data layer is usable. `MockInterpreter` is a GPU-free keyword
baseline (and the offline fallback); `LLMInterpreter` is the v0 "frozen LLM + prompt"
interpreter over amax's served Qwen (OpenAI-compatible)."""
from __future__ import annotations

import json
import re
import time

from .commands import KINDS, Command

# keyword -> kind, checked in priority order (first match wins). Ordered so that the more
# specific intents (flank, disperse, regroup) are tested before the generic go_to.
_KEYWORDS = [
    ("flank",     ("flank", "envelop", "encircle", "pincer", "surround", "both sides", "wings", "either side")),
    ("disperse",  ("disperse", "spread", "scatter", "fan out", "break up", "spread out", "split up")),
    ("regroup",   ("regroup", "group up", "consolidate", "pull back", "fall back", "gather", "on me", "rally back", "close up", "form up on")),
    ("form_line", ("line", "row", "rank", "abreast", "single file", "in a line")),
    ("go_to",     ("go to", "move to", "head to", "head for", "push to", "advance", "rally", "march", "proceed", "get to", "move out", "go", "move")),
]


def _floats(text: str):
    return [float(x) for x in re.findall(r"[-+]?\d+\.?\d*", text)]


_NUMWORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
             "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
             "sixty", "seventy", "eighty", "ninety", "hundred"}


def _has_location_signal(text: str) -> bool:
    """Does the operator's text actually contain a coordinate/quantity (digits or number-words)?
    If not, any coordinates an LLM returns are hallucinated and must not be trusted."""
    t = text.lower()
    if re.search(r"\d", t):
        return True
    return bool(set(re.findall(r"[a-z]+", t)) & _NUMWORDS)


def _kind_from_text(text: str) -> str:
    t = text.lower()
    for kind, kws in _KEYWORDS:
        if any(k in t for k in kws):
            return kind
    return "go_to"


class MockInterpreter:
    """Keyword + regex interpreter — deterministic, offline, no dependencies. A genuine
    baseline for the LLM interpreter: it handles literal phrasing but is brittle to the
    abstract orders the L2 LLM produces (which is exactly what the comparison reveals)."""

    def __init__(self):
        self.last_latency = 0.0
        self.last_raw = ""

    def __call__(self, text: str, world=None) -> Command:
        t0 = time.perf_counter()
        kind = _kind_from_text(text)
        nums = _floats(text)
        params: dict = {}
        if kind == "form_line":
            if len(nums) >= 4:
                params = {"start": nums[0:2], "end": nums[2:4]}
        elif kind in ("go_to", "regroup", "flank"):
            if len(nums) >= 2:
                params = {"point": nums[0:2]}
        # disperse takes no params
        self.last_raw = kind
        self.last_latency = time.perf_counter() - t0
        return Command(kind, params)


_SYSTEM = (
    "You are the command interpreter for a real-time crowd controller. Translate the "
    "operator's free-form order into exactly ONE canonical command, as strict JSON.\n"
    "Allowed commands:\n"
    '  {"kind":"go_to","params":{"point":[x,y]}}                  move/advance/push the whole group to one location\n'
    '  {"kind":"form_line","params":{"start":[x,y],"end":[x,y]}}  line up in a row between two points\n'
    '  {"kind":"regroup","params":{"point":[x,y]}}                fall back and gather tightly at a rally point\n'
    '  {"kind":"disperse","params":{}}                            spread out / scatter from the current position\n'
    '  {"kind":"flank","params":{"point":[x,y]}}                  split and envelop a target from BOTH sides at once\n'
    "Coordinates are in a square world from [0,0] to [B,B]. If the order gives no explicit "
    "coordinates, omit the point/start/end fields. Reply with ONLY the JSON object."
)


def _extract_json(s: str):
    """Pull the first complete top-level JSON object out of a model reply (tolerates
    <think> traces and prose around it)."""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except Exception:
                    return None
    return None


class LLMInterpreter:
    """v0 interpreter [A]: a frozen served LLM (amax's Qwen) + prompt. Maps free-form text
    -> canonical Command. Defaults target the OpenAI-compatible vLLM endpoint on amax."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY",
                 model: str = "Qwen/Qwen3-14B-AWQ", temperature: float = 0.0,
                 enable_thinking: bool = False, guard_coords: bool = True):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        # guard_coords: drop coordinates the source text never specified, instead of trusting an
        # LLM hallucination. Defers spatial grounding to the world/default (see E6). Safe default.
        self.guard_coords = guard_coords
        self.last_latency = 0.0
        self.last_raw = ""

    def __call__(self, text: str, world=None) -> Command:
        ctx = ""
        if world is not None:
            c = world.pos.mean(axis=0)
            ctx = (f"World: {world.n} agents, bounds 0..{world.bounds:.0f}, "
                   f"crowd centroid ~({c[0]:.1f}, {c[1]:.1f}). ")
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": f'{ctx}Order: "{text}"'}],
            temperature=self.temperature,
            max_tokens=256,
        )
        # vLLM/Qwen3: disable the reasoning trace so we get JSON directly.
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        t0 = time.perf_counter()
        try:
            r = self.client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("extra_body", None)          # endpoint may reject the kwarg
            r = self.client.chat.completions.create(**kwargs)
        self.last_latency = time.perf_counter() - t0
        self.last_raw = (r.choices[0].message.content or "").strip()
        cmd = self._parse(self.last_raw)
        if self.guard_coords and not _has_location_signal(text):
            cmd = Command(cmd.kind, {k: v for k, v in cmd.params.items()
                                     if k not in ("point", "start", "end")})
        return cmd

    @staticmethod
    def _parse(raw: str) -> Command:
        obj = _extract_json(raw) or {}
        kind = obj.get("kind", "go_to")
        if kind not in KINDS:
            kind = "go_to"
        params = obj.get("params", {}) or {}
        clean: dict = {}
        for key in ("point", "start", "end"):
            v = params.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    clean[key] = [float(v[0]), float(v[1])]
                except (TypeError, ValueError):
                    pass
        for key in ("dist", "spread"):
            if key in params:
                try:
                    clean[key] = float(params[key])
                except (TypeError, ValueError):
                    pass
        return Command(kind, clean)
