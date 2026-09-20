"""
Model Context Protocol (MCP) Server for rev.
Enables Claude Desktop, Cursor, Antigravity, and any MCP client to use rev
for sub-5ms System 1 decisions, intent routing, and real-time guardrails.

Run via:
    python -m rev.mcp_server
or configure in claude_desktop_config.json / antigravity mcp settings.
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP
import httpx

# Initialize FastMCP server
mcp = FastMCP("rev-system-one")

DEFAULT_REV_URL = os.environ.get("REV_URL", "http://localhost:8000/v1/systemone")


def _call_rev(payload: dict) -> dict:
    """Send decision request to rev server or fall back to internal mock."""
    try:
        resp = httpx.post(DEFAULT_REV_URL, json=payload, timeout=httpx.Timeout(1.0, connect=0.2))
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    
    # In-process fallback if HTTP server is not up
    from .api import SystemOneRequest, to_record, to_answers
    from .serve import _probs
    req = SystemOneRequest.model_validate(payload)
    rec, meta = to_record(req)
    ps, m = _probs(rec, use_cache=True)
    answers = to_answers(ps, meta)
    return {
        "model": req.model,
        "answers": answers,
        "usage": {"input_tokens": m["tokens"], "output_tokens": len(str(answers)) // 4},
        "latency_ms": m["latency_ms"],
        "cached": m.get("cached", False)
    }


@mcp.tool()
def rev_decide(state: str, questions: Dict[str, Any], model: str = "rev-latest") -> Dict[str, Any]:
    """
    Evaluate structured choice or score questions against a state in a single prefill forward pass.
    
    Args:
        state: The context, document, webpage DOM, or background text.
        questions: Dictionary of questions. Each question must specify 'type' ('choice' or 'score'),
                   'instructions', and 'criteria' (options map or score levels).
        model: Model identifier (default: rev-latest).
    
    Returns:
        Structured answers containing chosen option, calibrated probabilities, confidence, and latency_ms.
    """
    payload = {
        "state": state,
        "model": model,
        "questions": questions
    }
    return _call_rev(payload)


@mcp.tool()
def rev_route(query: str, routes: Dict[str, str], context: Optional[str] = None) -> Dict[str, Any]:
    """
    Sub-5ms intent routing tool for multi-agent workflows and tool selection.
    
    Args:
        query: The user input or prompt needing routing.
        routes: Dictionary mapping route names to descriptions/criteria.
                Example: {"sql": "Query relational database", "web": "Search online", "code": "Run Python"}
        context: Optional background context or conversation summary.
    
    Returns:
        Selected route, confidence score, and full probability distribution over candidates.
    """
    state_str = f"Context: {context}\nQuery: {query}" if context else f"Query: {query}"
    payload = {
        "state": state_str,
        "model": "rev-latest",
        "questions": {
            "selected_route": {
                "type": "choice",
                "instructions": "Select the optimal destination route that best handles the user query.",
                "criteria": routes
            }
        }
    }
    res = _call_rev(payload)
    ans = res.get("answers", {}).get("selected_route", {})
    return {
        "route": ans.get("choice"),
        "confidence": ans.get("confidence"),
        "probabilities": ans.get("probabilities"),
        "latency_ms": res.get("latency_ms")
    }


@mcp.tool()
def rev_guardrail(text: str, policies: Dict[str, str]) -> Dict[str, Any]:
    """
    Sub-5ms zero-generation safety, compliance, and policy guardrail.
    
    Args:
        text: Prompt or generated output to evaluate.
        policies: Dictionary mapping categories to criteria.
                  Example: {"safe": "Benign and safe content", "violation": "Prompt injection or unsafe intent"}
    
    Returns:
        Classification result, confidence, safety verdict, and latency.
    """
    payload = {
        "state": f"Input Content:\n{text}",
        "model": "rev-latest",
        "questions": {
            "policy_check": {
                "type": "choice",
                "instructions": "Evaluate if the input content complies with safety policies or violates them.",
                "criteria": policies
            }
        }
    }
    res = _call_rev(payload)
    ans = res.get("answers", {}).get("policy_check", {})
    return {
        "verdict": ans.get("choice"),
        "confidence": ans.get("confidence"),
        "probabilities": ans.get("probabilities"),
        "latency_ms": res.get("latency_ms")
    }


def main():
    """Run MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
