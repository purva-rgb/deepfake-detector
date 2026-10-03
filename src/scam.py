"""Deterministic YAML scam-rule matcher with a context guard (SPEC 6a). No ML.

Guard: a hit is suppressed if a negation / warning word occurs within `window_tokens` tokens
BEFORE the match start inside the same sentence, or if a rule-level `suppress_if` regex
matches the sentence.  S_scam = min(1, S_max + 0.10 * (N_categories - 1)).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import ROOT

RULES_PATH = ROOT / "rules.yaml"
CATEGORY_BONUS = 0.10
_SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+(?=\s|$)|$)", re.S)
_TOKEN_RE = re.compile(r"[\w']+")


@dataclass
class Hit:
    rule_id: str
    category: str
    strength: float
    start: int          # char offsets in the ORIGINAL text
    end: int
    text: str
    suppressed: bool = False
    reason: str = ""


@dataclass
class Rules:
    window_tokens: int
    negation: re.Pattern
    categories: dict
    rules: list = field(default_factory=list)   # dicts with compiled patterns


def load_rules(path: Path | str = RULES_PATH) -> Rules:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    g = cfg["guard"]
    rules = []
    for r in cfg["rules"]:
        assert r.get("language") == "en", f"rule {r['id']} must declare language: en"
        rules.append({
            "id": r["id"], "category": r["category"], "strength": float(r["strength"]),
            "patterns": [re.compile(p, re.I) for p in r["patterns"]],
            "suppress_if": [re.compile(p, re.I) for p in r.get("suppress_if", [])],
        })
    return Rules(int(g["window_tokens"]), re.compile(g["negation_regex"], re.I), cfg["categories"], rules)


def _normalise(text: str) -> str:
    # same-length replacements so offsets stay valid for highlighting
    return text.replace("\u2019", "'").replace("\u2018", "'").replace("\u00a0", " ")


def scan(text: str, rules: Rules, include_suppressed: bool = False) -> list[Hit]:
    text = _normalise(text)
    hits: list[Hit] = []
    for sm in _SENT_RE.finditer(text):
        sent = sm.group(0)
        if not sent.strip():
            continue
        off = sm.start()
        for rule in rules.rules:
            for pat in rule["patterns"]:
                for m in pat.finditer(sent):
                    h = Hit(rule["id"], rule["category"], rule["strength"], off + m.start(), off + m.end(),
                            m.group(0))
                    before = _TOKEN_RE.findall(sent[: m.start()].lower())[-rules.window_tokens:]
                    neg = rules.negation.search(" ".join(before)) if before else None
                    if neg:
                        h.suppressed, h.reason = True, f"negation/warning word '{neg.group(0)}' before match"
                    elif any(s.search(sent) for s in rule["suppress_if"]):
                        h.suppressed, h.reason = True, "legitimate-support context"
                    if include_suppressed or not h.suppressed:
                        hits.append(h)
    # de-duplicate identical spans from the same rule (several patterns may hit the same words)
    uniq, seen = [], set()
    for h in sorted(hits, key=lambda h: (h.start, h.end)):
        k = (h.rule_id, h.start, h.end)
        if k not in seen:
            seen.add(k)
            uniq.append(h)
    return uniq


def score(hits: list[Hit]) -> dict:
    """S_scam from active (unsuppressed) hits. No hits -> S = 0.0 (text WAS evaluated, nothing found)."""
    act = [h for h in hits if not h.suppressed]
    if not act:
        return {"S": 0.0, "S_max": 0.0, "categories": [], "n_categories": 0}
    s_max = max(h.strength for h in act)
    cats = sorted({h.category for h in act})
    return {"S": min(1.0, s_max + CATEGORY_BONUS * (len(cats) - 1)), "S_max": s_max,
            "categories": cats, "n_categories": len(cats)}
