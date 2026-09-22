"""
Unified 'One Model' engine for rev.

Unifies Pretrained Encoders:
  - ModernBERT-large (421M): English backbone with deep reasoning and long-context capabilities.
  - mmBERT-base (322M): Multilingual backbone natively supporting 100+ languages and 26 scripts.
  - Typed Decisions (421M): Fine-tuned on specialized enterprise decision schemas.

Inherits world knowledge, syntax, vocabulary, and grammar directly from backbone pretraining.
Routes between backbones in sub-microseconds via exact Unicode script and language detection (rev.lang).
"""

import os
import threading
from typing import Any, Dict, List, Optional, Union

from .lang import analyse

# Default checkpoint definitions
BUNDLE_REPO = "convaiinnovations/laya"
DEFAULT_MODELS = {
    "english": (BUNDLE_REPO, None),
    "multilingual": (BUNDLE_REPO, "multilingual"),
    "typed-decisions": (BUNDLE_REPO, "typed-decisions"),
}

STANDALONE_MODELS = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}

_ALIASES = {
    "en": "english",
    "modernbert": "english",
    "modernbert-large": "english",
    "laya": "english",
    "default": "english",
    "multi": "multilingual",
    "mmbert": "multilingual",
    "mmbert-base": "multilingual",
    "ml": "multilingual",
    "typed": "typed-decisions",
    "typed_decisions": "typed-decisions",
}

_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    "security_incidents": {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}


class RouteDecision(dict):
    """Routing outcome: which backbone was selected, the rationale, and detected script details."""

    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self):
        return f"RouteDecision(model={self['model']!r}, reason={self['reason']!r})"


def normalise_name(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown model {name!r}; choose from {sorted(DEFAULT_MODELS)} or aliases {sorted(_ALIASES)}"
        )
    return key


def match_typed_decisions_workflow(questions: Dict[str, Any]) -> Optional[str]:
    """Matches known enterprise workflow schemas."""
    ids = set(questions or {})
    for wf, sig in _TYPED_DECISION_WORKFLOWS.items():
        if ids == sig:
            return wf
    return None


def _split(spec):
    if isinstance(spec, (tuple, list)):
        repo, sub = (list(spec) + [None])[:2]
        return repo, sub
    return spec, None


def _repo_str(spec):
    repo, sub = _split(spec)
    return f"{repo}/{sub}" if sub else repo


class UnifiedModel:
    """
    Unified System 1 Model.
    
    Acts as 'One Model' that seamlessly dispatches queries to ModernBERT-large (English)
    or mmBERT-base (100+ languages) based on zero-latency Unicode script analysis.
    
    Usage:
        import rev
        model = rev.UnifiedModel()
        
        # Automatically routed to mmBERT-base (100+ languages):
        model.predict("Mein Konto wurde zweimal belastet", questions)
        model.predict("मेरा ऑर्डर अभी तक नहीं आया", questions)
        
        # Automatically routed to ModernBERT-large (English):
        model.predict("My account was charged twice", questions)
    """

    def __init__(
        self,
        models: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        token: Optional[str] = None,
        max_loaded: int = 2,
        default: str = "english",
        auto_task_detection: bool = False,
        standalone_repos: bool = False,
        preload: bool = False,
    ):
        self.models = dict(STANDALONE_MODELS if standalone_repos else DEFAULT_MODELS)
        if models:
            self.models.update({normalise_name(k): v for k, v in models.items()})
        self.device = device
        self.token = token or os.environ.get("HF_TOKEN")
        self.max_loaded = max(1, int(max_loaded))
        self.default = normalise_name(default)
        self.auto_task_detection = bool(auto_task_detection)
        self._agents: Dict[str, Any] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        if preload:
            self.preload()

    def load(self, name: str):
        """Loads and caches the underlying checkpoint."""
        key = normalise_name(name)
        with self._lock:
            if key in self._agents:
                self._touch(key)
                return self._agents[key]
            from .agent import Agent

            repo, sub = _split(self.models[key])
            agent = Agent(repo, device=self.device, token=self.token, subfolder=sub)
            self._agents[key] = agent
            self._order.append(key)
            self._evict()
            return agent

    def _touch(self, key: str):
        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def _evict(self):
        with self._lock:
            while len(self._order) > self.max_loaded:
                victim = self._order.pop(0)
                self._agents.pop(victim, None)

    def preload(self, names: Optional[List[str]] = None):
        """Preloads checkpoints into memory for sub-10ms response times."""
        targets = [normalise_name(n) for n in (names or list(self.models))]
        with self._lock:
            self.max_loaded = max(self.max_loaded, len(targets), len(self._agents))
            for n in targets:
                if n not in self._agents:
                    self.load(n)
        return self

    def route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> RouteDecision:
        """Determines the optimal backbone checkpoint for the given state and questions."""
        if model is not None:
            key = normalise_name(model)
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models[key]),
                reason=f"explicit model={model!r}",
                detection=None,
                workflow=None,
            )

        if task is not None:
            key = normalise_name("typed-decisions" if str(task).lower().replace("-", "_") == "typed_decisions" else task)
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models[key]),
                reason=f"explicit task={task!r}",
                detection=None,
                workflow=None,
            )

        workflow = match_typed_decisions_workflow(questions or {})
        if workflow and self.auto_task_detection:
            return RouteDecision(
                model="typed-decisions",
                repo=self.models["typed-decisions"],
                reason=f"question ids match {workflow!r} typed-decisions workflow",
                detection=None,
                workflow=workflow,
            )

        if lang is not None:
            key = "english" if str(lang).lower().split("-")[0] in ("en", "eng", "english") else "multilingual"
            return RouteDecision(
                model=key,
                repo=_repo_str(self.models[key]),
                reason=f"explicit lang={lang!r}",
                detection=None,
                workflow=workflow,
            )

        det = analyse(state)
        if det["script"] == "unknown":
            key = self.default
            reason = f"no letters detected; fallback to default ({key})"
        elif det["script"] != "latin":
            key = "multilingual"
            reason = f"non-Latin script ({det['script']}, {det['non_latin_fraction'] * 100:.0f}% of text) -> routed to mmBERT-base"
        elif not det["is_english"]:
            key = "multilingual"
            if det["language"]:
                reason = f"Latin script text identified as {det['language']!r} -> routed to mmBERT-base"
            else:
                reason = f"Latin script with {det['diacritic_rate'] * 100:.0f}% non-English characters -> routed to mmBERT-base"
        else:
            key = "english"
            reason = "English Latin text -> routed to ModernBERT-large"

        return RouteDecision(
            model=key,
            repo=_repo_str(self.models[key]),
            reason=reason,
            detection=det,
            workflow=workflow,
        )

    def predict(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes unified prediction: automatically routes to ModernBERT-large or mmBERT-base.
        Returns answers, confidence scores, actions, and routing metadata.
        """
        decision = self.route(state, questions, model=model, task=task, lang=lang)
        agent = self.load(decision["model"])
        result = agent.system_one(state, questions)
        result["routing"] = dict(decision)
        return result

    system_one = predict

    @property
    def loaded(self) -> List[str]:
        with self._lock:
            return list(self._order)

    def __repr__(self):
        return f"UnifiedModel(loaded={self.loaded}, max_loaded={self.max_loaded}, default={self.default!r})"


# Alias Router to UnifiedModel for backwards compatibility and drop-in flexibility
Router = UnifiedModel

_GLOBAL_ROUTER: Optional[UnifiedModel] = None


def get_default_model() -> UnifiedModel:
    """Returns the shared global UnifiedModel instance."""
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None:
        _GLOBAL_ROUTER = UnifiedModel()
    return _GLOBAL_ROUTER


def predict(
    state: Union[str, dict, list],
    questions: Dict[str, Any],
    model: Optional[str] = None,
    task: Optional[str] = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenient top-level prediction using the unified 'One Model' engine.
    
    Example:
        import rev
        res = rev.predict("Can I get a refund?", rev.presets.triage_questions())
    """
    return get_default_model().predict(state, questions, model=model, task=task, lang=lang)
