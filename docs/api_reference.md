# API Reference: `rev` (`rev-decision`)

Comprehensive reference for the Python SDK, data models, routing engine, scoring rules, and utility modules.

---

## Table of Contents
- [Top-Level Functions](#top-level-functions)
  - [`rev.predict()`](#revpredict)
  - [`rev.get_default_model()`](#revget_default_model)
- [Unified Decision Engine (`rev.Model` / `rev.UnifiedModel`)](#unified-decision-engine-revmodel--revunifiedmodel)
  - [Initialization](#initialization)
  - [Methods](#methods)
- [RouteDecision](#routedecision)
- [High-Cardinality Candidate Shortlisting (`rev.shortlist`)](#high-cardinality-candidate-shortlisting-revshortlist)
  - [`rev.shortlist_choice()`](#revshortlist_choice)
  - [`rev.predict_shortlist()`](#revpredict_shortlist)
  - [`rev.embed_fn_from_agent()`](#revembed_fn_from_agent)
- [Strictly Proper Scoring & Calibration (`rev.common`)](#strictly-proper-scoring--calibration-revcommon)
  - [`rev.proper_reward()`](#revproper_reward)
  - [`rev.confidence_from_probs()`](#revconfidence_from_probs)
  - [`rev.clamp_temperature()`](#revclamp_temperature)
  - [`rev.ece_score()`](#revece_score)
- [Language & Script Detection (`rev.lang`)](#language--script-detection-revlang)
  - [`rev.lang.analyse()`](#revlanganalyse)
- [Email Cleaning Utilities (`rev.email`)](#email-cleaning-utilities-revemail)
  - [`rev.email.clean_email_body()`](#revemailclean_email_body)
  - [`rev.email.email_state()`](#revemailemail_state)
- [TypeSafe System One Specification (`rev.api`)](#typesafe-system-one-specification-revapi)
  - [`Choice`](#choice), [`Noul`](#noul), [`Score`](#score)

---

## Top-Level Functions

### `rev.predict()`

Executes a unified single-pass decision across all questions using the automatically routed backbone (ModernBERT-large for English, mmBERT-base for multilingual).

```python
rev.predict(
    state: Union[str, Dict[str, Any], List[Any]],
    questions: Dict[str, Dict[str, Any]],
    model: Optional[str] = None,
    task: Optional[str] = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]
```

#### Parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `state` | `str \| dict \| list` | *Required* | Document, conversational message, email body, or structured context to evaluate. |
| `questions` | `dict[str, dict]` | *Required* | Mapping of `question_id` to question schema definitions (`choice`, `noul`, or `score`). |
| `model` | `str \| None` | `None` | Explicit checkpoint override (`"english"`, `"multilingual"`, `"typed-decisions"`). Bypasses automatic detection. |
| `task` | `str \| None` | `None` | Explicit task workflow key (e.g. `"customer_service"`, `"security_incidents"`). |
| `lang` | `str \| None` | `None` | Language code hint (e.g. `"en"`, `"es"`, `"hi"`, `"zh"`). If non-English, dispatches to multilingual backbone. |

#### Returns
A dictionary containing typed answers, probabilities, confidence scores, and routing metadata:
```python
{
    "answers": {
        "intent": {
            "type": "choice",
            "choice": "refund",
            "confidence": 0.984,
            "probabilities": {"refund": 0.984, "technical_help": 0.012, "other": 0.004},
            "entropy": 0.042,
            "margin": 0.972
        },
        "urgent": {
            "type": "noul",
            "noul": 0.92,
            "confidence": 0.92
        },
        "frustration": {
            "type": "score",
            "score": 2.45,
            "confidence": 0.89,
            "probabilities": [0.01, 0.08, 0.37, 0.54]
        }
    },
    "routing": {
        "model": "english",
        "reason": "English Latin text -> routed to ModernBERT-large",
        "detection": {
            "script": "latin",
            "is_english": True,
            "language": "en",
            "diacritic_rate": 0.0
        }
    },
    "usage": {"input_tokens": 84, "question_branches": 3}
}
```

---

### `rev.get_default_model()`

Retrieves the shared global `UnifiedModel` singleton instance.

```python
rev.get_default_model() -> UnifiedModel
```

---

## Unified Decision Engine (`rev.Model` / `rev.UnifiedModel`)

`rev.Model` is the primary orchestrator that unifies multiple pretrained encoders behind an automatic script-routing runtime.

### Initialization

```python
rev.Model(
    models: Optional[Dict[str, str]] = None,
    device: Optional[str] = None,
    token: Optional[str] = None,
    max_loaded: int = 2,
    default: str = "english",
    auto_task_detection: bool = False,
    standalone_repos: bool = False,
    preload: bool = False,
)
```

#### Parameters
- `models`: Optional mapping of model alias to custom Hugging Face repo ID or local checkpoint directory.
- `device`: PyTorch device string (`"cuda"`, `"cpu"`, `"mps"`). Defaults to CUDA if available.
- `token`: Hugging Face API token for downloading gated checkpoints. Defaults to `os.environ.get("HF_TOKEN")`.
- `max_loaded`: Maximum number of encoder weights kept loaded in GPU/RAM simultaneously. Uses LRU eviction. Default `2`.
- `default`: Fallback model name when text contains no identifiable script or letters. Default `"english"`.
- `auto_task_detection`: If `True`, analyzes question keys to detect specialized typed-decision workflows.
- `preload`: If `True`, proactively downloads and loads checkpoints during initialization.

### Methods

#### `model.predict(state, questions, model=None, task=None, lang=None) -> dict`
Executes single-pass inference on the routed backbone.

#### `model.route(state, questions=None, model=None, task=None, lang=None) -> RouteDecision`
Determines which backbone should process the request in sub-microseconds without loading model weights.

#### `model.load(name: str) -> Agent`
Loads and returns the underlying `rev.Agent` runtime for a specific backbone key.

#### `model.preload(names: Optional[List[str]] = None) -> UnifiedModel`
Loads requested backbones into memory ahead of time for zero-cold-start latency.

---

## RouteDecision

A specialized dictionary subclass containing the routing verdict:

```python
class RouteDecision(dict):
    @property
    def model(self) -> str: ...       # Selected backbone ('english' or 'multilingual')
    @property
    def reason(self) -> str: ...      # Human-readable justification
```

---

## High-Cardinality Candidate Shortlisting (`rev.shortlist`)

For questions with 20 to 255+ options (e.g. BANKING77, deep product catalogs), passing all options simultaneously into a cross-attentive sequence exceeds token budgets. `rev.shortlist` implements bi-encoder coarse-to-fine candidate filtering.

### `rev.shortlist_choice()`

Filters criteria labels down to top-$k$ candidates based on cosine similarity against the state.

```python
rev.shortlist_choice(
    state: Any,
    criteria: Union[Dict[str, str], List[str]],
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    k: int = 20,
    *,
    instructions: Optional[str] = None,
) -> List[str]
```

### `rev.predict_shortlist()`

Wraps model prediction: automatically detects choice questions exceeding $k$ options, runs bi-encoder shortlisting, and executes cross-attentive evaluation over the pruned candidate set.

```python
rev.predict_shortlist(
    agent: Any,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    k: int = 20,
    **predict_kwargs: Any,
) -> Dict[str, Any]
```

### `rev.embed_fn_from_agent()`

Builds a mean-pooled sentence embedding function directly from an already-loaded `rev.Agent` instance without loading third-party embedding libraries.

```python
rev.embed_fn_from_agent(agent: Agent) -> Callable[[Sequence[str]], np.ndarray]
```

---

## Strictly Proper Scoring & Calibration (`rev.common`)

### `rev.proper_reward()`

Evaluates probabilistic predictions against ground truth using strictly proper scoring rules (LogScore + Spherical + RPS).

```python
rev.proper_reward(
    probs: torch.Tensor,
    targets: torch.Tensor,
    qtypes: torch.Tensor,
    mask: torch.Tensor,
    w_spherical: float = 0.5,
    w_rps: float = 0.5,
) -> torch.Tensor
```

### `rev.confidence_from_probs()`

Computes normalized Shannon entropy confidence:
$$C = 1 - \frac{H(p)}{\log(K)}$$
Returns `1.0` for Dirac distributions and `0.0` for uniform uncertainty.

```python
rev.confidence_from_probs(probs: np.ndarray) -> float
```

### `rev.clamp_temperature()`

Clamps post-hoc calibration temperatures strictly within $[0.5, 5.0]$ to eliminate numerical overflow and pathological over-sharpening.

```python
rev.clamp_temperature(t: float) -> float
```

### `rev.ece_score()`

Computes Expected Calibration Error (ECE) across 15 equal-width confidence bins:

```python
rev.ece_score(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float
```

---

## Language & Script Detection (`rev.lang`)

Zero-dependency Unicode script and European language detector operating in $<1\ \mu\text{s}$.

### `rev.lang.analyse()`

```python
rev.lang.analyse(text: Union[str, dict, list, None]) -> Dict[str, Any]
```

#### Returns
```python
{
    "script": "devanagari",          # Primary detected Unicode script (e.g. latin, devanagari, cyrillic, han, arabic)
    "is_english": False,             # Boolean indicator
    "language": "hi",                # ISO 639-1 language code if identified
    "non_latin_fraction": 0.88,      # Ratio of non-Latin characters
    "diacritic_rate": 0.0            # Diacritic character density
}
```

---

## Email Cleaning Utilities (`rev.email`)

### `rev.email.clean_email_body()`

Strips email reply chains, historical threads (`On [Date] wrote:`), quoted headers, disclaimer boilerplate, and sign-offs in $<0.2\ \text{ms}$.

```python
rev.email.clean_email_body(text: str) -> str
```

### `rev.email.email_state()`

Constructs a structured dictionary payload from raw RFC 822 email fields with automatic reply stripping.

```python
rev.email.email_state(
    subject: str,
    body: str,
    sender: Optional[str] = None,
    clean: bool = True
) -> Dict[str, str]
```

---

## TypeSafe System One Specification (`rev.api`)

`rev` complies with the `POST /v1/systemone` specification:

```python
from rev.api import Choice, Noul, Score

# Choice: Categorical decision across mutually exclusive options
choice_q = Choice(
    instructions="Which team owns this ticket?",
    criteria={"billing": "Invoices and cards", "support": "Technical bugs"}
)

# Noul: Continuous [0.0, 1.0] probability for binary propositions
noul_q = Noul(instructions="Is this user asking for an urgent escalation?")

# Score: Continuous expected score over ordinal rank intervals
score_q = Score(
    instructions="Customer frustration index",
    criteria=["neutral", "annoyed", "outraged"]
)
```
