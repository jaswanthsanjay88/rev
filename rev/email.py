"""
Email utilities for parsing, cleaning, and structuring email inputs in rev.
"""

import re
from typing import Dict, Optional

_QUOTE_HEADERS = [
    re.compile(r"^\s*On .{0,300}wrote:\s*$", re.I),
    re.compile(r"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", re.I),
    re.compile(r"^\s*_{8,}\s*$"),
    re.compile(r"^\s*From:\s.+$", re.I),
]
_SIGNATURE_MARKERS = [
    re.compile(r"^\s*--\s*$"),
    re.compile(r"^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$", re.I),
    re.compile(r"^\s*sent from my (iphone|android|mobile|ipad)", re.I),
]
_DISCLAIMER = re.compile(
    r"(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|"
    r"if you (have )?received this (e-?mail|message) in error)",
    re.I,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _strip_disclaimer(paragraph: str) -> str:
    """Drop boilerplate disclaimer text from paragraph."""
    if not _DISCLAIMER.search(paragraph):
        return paragraph
    parts = [p.strip() for p in _SENTENCE.split(paragraph) if p.strip()]
    return " ".join(p for p in parts if not _DISCLAIMER.search(p))


def clean_email_body(body: str, max_chars: int = 3000) -> str:
    """Remove quoted history, signatures, and legal disclaimers."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    lines = []
    for line in text.split("\n"):
        if any(p.match(line) for p in _QUOTE_HEADERS) and lines:
            break
        if line.lstrip().startswith(">"):
            continue
        lines.append(line.rstrip())
    cut = len(lines)
    for i in range(max(1, min(int(len(lines) * 0.6), len(lines) - 8)), len(lines)):
        if len(lines[i].strip()) <= 40 and any(p.match(lines[i]) for p in _SIGNATURE_MARKERS):
            cut = i
            break
    lines = lines[:cut]
    paragraphs = [_strip_disclaimer(p) for p in re.split(r"\n\s*\n", "\n".join(lines))]
    text = re.sub(r"[ \t]+", " ", "\n\n".join(p.strip() for p in paragraphs if p.strip()))
    return text[:max_chars]


def email_state(subject: str, body: str, sender: Optional[str] = None, clean: bool = True, **extra) -> Dict:
    """Construct a clean state dictionary for email classification."""
    state = {
        "subject": (subject or "").strip(),
        "body": clean_email_body(body) if clean else (body or ""),
    }
    if sender:
        state["from"] = sender
    state.update({k: v for k, v in extra.items() if v is not None})
    return state
