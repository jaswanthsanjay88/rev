"""
Unit tests for rev's Unified 'One Model' architecture, routing, scoring rules, shortlisting, and presets.
"""

import math
import numpy as np
import torch

import rev
from rev import (
    UnifiedModel,
    Router,
    RouteDecision,
    proper_reward,
    confidence_from_probs,
    clamp_temperature,
    ece_score,
    render_options,
    shortlist_choice,
    predict_shortlist,
    presets,
    lang,
    email,
)


def test_script_and_language_detection():
    # 1. English Latin
    det_en = lang.analyse("The customer is asking for a refund on their latest billing invoice.")
    assert det_en["script"] == "latin"
    assert det_en["is_english"] is True

    # 2. Devanagari (Hindi)
    det_hi = lang.analyse("मेरा ऑर्डर अभी तक नहीं मिला है, कृपया सहायता करें।")
    assert det_hi["script"] == "devanagari"
    assert det_hi["is_english"] is False

    # 3. Cyrillic (Russian)
    det_ru = lang.analyse("Пожалуйста, отмените мою подписку как можно скорее.")
    assert det_ru["script"] == "cyrillic"
    assert det_ru["is_english"] is False

    # 4. Han (Chinese)
    det_zh = lang.analyse("请问这个产品的保修期是多长时间？")
    assert det_zh["script"] == "han"
    assert det_zh["is_english"] is False

    # 5. Arabic
    det_ar = lang.analyse("أريد استرداد المبلغ المدفوع مقابل هذه الخدمة")
    assert det_ar["script"] == "arabic"
    assert det_ar["is_english"] is False

    # 6. German (Latin with German stopwords & umlauts)
    det_de = lang.analyse("Mein Konto wurde zweimal belastet und ich möchte eine Rückerstattung.")
    assert det_de["script"] == "latin"
    assert det_de["is_english"] is False
    assert det_de["language"] == "de"

    # 7. French
    det_fr = lang.analyse("Bonjour, je souhaite annuler ma commande passée hier soir.")
    assert det_fr["script"] == "latin"
    assert det_fr["is_english"] is False
    assert det_fr["language"] == "fr"


def test_unified_model_routing():
    model = UnifiedModel()

    # English text routes to English (ModernBERT-large)
    dec_en = model.route("The server encountered an error while processing the transaction.")
    assert dec_en["model"] == "english"
    assert "English" in dec_en["reason"]

    # Multilingual text routes to Multilingual (mmBERT-base)
    dec_hi = model.route("कृपया मुझे तुरंत बताएं कि क्या हुआ")
    assert dec_hi["model"] == "multilingual"
    assert "devanagari" in dec_hi["reason"].lower()

    dec_es = model.route("El pedido no ha llegado a mi dirección postal.")
    assert dec_es["model"] == "multilingual"

    # Explicit override takes absolute precedence
    dec_override = model.route("Hello world", model="multilingual")
    assert dec_override["model"] == "multilingual"
    assert "explicit model" in dec_override["reason"]

    dec_lang = model.route("Hello world", lang="es")
    assert dec_lang["model"] == "multilingual"
    assert "explicit lang" in dec_lang["reason"]


def test_strictly_proper_scoring_rules():
    # 2-class distribution: target=[1, 0], predicted=[0.9, 0.1]
    q_good = torch.tensor([[0.9, 0.1]])
    q_bad = torch.tensor([[0.1, 0.9]])
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, True]])
    qtype = torch.tensor([0])  # choice

    r_good = proper_reward(q_good, target, qtype, mask)
    r_bad = proper_reward(q_bad, target, qtype, mask)

    # Reward for accurate prediction must strictly exceed poor prediction
    assert r_good.item() > r_bad.item()

    # Score question with Ranked Probability Score (RPS)
    qtype_score = torch.tensor([1])  # score
    # 3 levels: target is level 0 ([1, 0, 0])
    target_3 = torch.tensor([[1.0, 0.0, 0.0]])
    mask_3 = torch.tensor([[True, True, True]])
    # Near miss (level 1) vs distant miss (level 2)
    q_near = torch.tensor([[0.0, 1.0, 0.0]])
    q_far = torch.tensor([[0.0, 0.0, 1.0]])

    r_near = proper_reward(q_near, target_3, qtype_score, mask_3)
    r_far = proper_reward(q_far, target_3, qtype_score, mask_3)

    # RPS must penalize the distant error more heavily than near error
    assert r_near.item() > r_far.item()


def test_normalized_shannon_entropy_confidence():
    # Certain prediction (p = [1.0, 0.0]) -> confidence = 1.0
    p_cert = np.array([1.0, 0.0])
    assert math.isclose(confidence_from_probs(p_cert, 2), 1.0, abs_tol=1e-4)

    # Uniform random guess (p = [0.5, 0.5]) -> confidence = 0.0
    p_unif = np.array([0.5, 0.5])
    assert math.isclose(confidence_from_probs(p_unif, 2), 0.0, abs_tol=1e-4)

    # 4 options uniform guess (p = [0.25, 0.25, 0.25, 0.25]) -> confidence = 0.0
    p_unif4 = np.array([0.25, 0.25, 0.25, 0.25])
    assert math.isclose(confidence_from_probs(p_unif4, 4), 0.0, abs_tol=1e-4)


def test_bounded_temperature_clamping():
    # Normal temperature passes through
    assert clamp_temperature(1.5) == 1.5

    # Pathological temperatures below 0.5 are clamped to 0.5
    assert clamp_temperature(0.1) == 0.5

    # Excessive temperatures above 5.0 are clamped to 5.0
    assert clamp_temperature(10.0) == 5.0

    # NaN / string fallback to 1.0
    assert clamp_temperature("invalid") == 1.0
    assert clamp_temperature(float("nan")) == 1.0


def test_shortlist_choice():
    criteria = {
        "refund": "money back for an order",
        "tech_support": "computer bug and crash",
        "account_login": "forgot password or reset pin",
        "shipping": "track package delivery status",
    }
    # Mock bi-encoder returning deterministic embeddings based on keyword overlap
    def mock_embed(texts):
        vecs = []
        for t in texts:
            t_low = t.lower()
            v = [
                float("refund" in t_low or "money" in t_low),
                float("bug" in t_low or "crash" in t_low),
                float("password" in t_low or "login" in t_low),
                float("delivery" in t_low or "track" in t_low or "shipping" in t_low),
            ]
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    # Query: "I need my money back" -> refund should be rank 1
    shortlisted = shortlist_choice(
        "I need my money back",
        criteria,
        embed_fn=mock_embed,
        k=2,
    )
    assert len(shortlisted) == 2
    assert shortlisted[0] == "refund"


def test_presets():
    triage = presets.triage_questions()
    assert "intent" in triage
    assert "is_urgent" in triage
    assert triage["intent"]["type"] == "choice"

    guard = presets.guard_questions()
    assert "jailbreak" in guard
    assert "prompt_injection" in guard

    observability = presets.observability_questions()
    assert "outcome" in observability
    assert "risk" in observability


def test_email_cleaning():
    raw_email = (
        "Hello Team,\n\n"
        "We are seeing 500 error codes on the API endpoint.\n\n"
        "--\n"
        "Best regards,\n"
        "John Doe\n\n"
        "This email is confidential and intended solely for the recipient.\n"
    )
    cleaned = email.clean_email_body(raw_email)
    assert "500 error codes" in cleaned
    assert "confidential" not in cleaned
    assert "John Doe" not in cleaned


if __name__ == "__main__":
    try:
        import pytest
        pytest.main(["-v", __file__])
    except ImportError:
        print("Running tests via standalone runner...")
        test_script_and_language_detection()
        print("  [OK] test_script_and_language_detection passed")
        test_unified_model_routing()
        print("  [OK] test_unified_model_routing passed")
        test_strictly_proper_scoring_rules()
        print("  [OK] test_strictly_proper_scoring_rules passed")
        test_normalized_shannon_entropy_confidence()
        print("  [OK] test_normalized_shannon_entropy_confidence passed")
        test_bounded_temperature_clamping()
        print("  [OK] test_bounded_temperature_clamping passed")
        test_shortlist_choice()
        print("  [OK] test_shortlist_choice passed")
        test_presets()
        print("  [OK] test_presets passed")
        test_email_cleaning()
        print("  [OK] test_email_cleaning passed")
        print("\nAll 8 test suites passed successfully!")
