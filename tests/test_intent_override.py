import importlib
import os
import asyncio

def reload_main_with_env(enabled: str, keywords: str):
    os.environ["INTENT_OVERRIDE_FORCE_PRICE_ENABLED"] = enabled
    os.environ["INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS"] = keywords

    # Ensure required env vars exist for import (main.py reads them at import time)
    os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    import app.main as main
    return importlib.reload(main)

def test_force_price_mode_when_cost_words_and_service_present(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    msg = "How much does a colonoscopy cost?"
    merged = {"service_query": "colonoscopy"}

    assert main.should_force_price_mode(msg, merged) is True

def test_does_not_force_when_no_service_query(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    msg = "How much does it cost?"
    merged = {"service_query": ""}

    assert main.should_force_price_mode(msg, merged) is False

def test_does_not_force_when_disabled(monkeypatch):
    main = reload_main_with_env("false", "cost,how much,price")

    msg = "How much does a colonoscopy cost?"
    merged = {"service_query": "colonoscopy"}

    assert main.should_force_price_mode(msg, merged) is False


def test_message_contains_zip_ignores_bare_cpt_codes(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    assert main.message_contains_zip("My ZIP is 06119") is True
    assert main.message_contains_zip("How much does a colonoscopy cost in 06119?") is True
    assert main.message_contains_zip("CPT 70551") is False


def test_plan_like_is_not_treated_as_carrier(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    assert main.extract_plan_like_from_text("I have PPO") == "PPO"
    assert main.extract_carrier_from_text("I have PPO") is None
    assert main.message_contains_payment_info("I have PPO") is True


def test_apply_payment_hints_sets_plan_like_without_fake_carrier(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    merged = {"service_query": "colonoscopy", "payer_like": "Aetna", "plan_like": None}
    intent = {"mode": "price"}

    main.apply_payment_hints_from_message("I have PPO", merged, intent)

    assert merged["payment_mode"] == "insurance"
    assert merged["plan_like"] == "PPO"
    assert merged["payer_like"] is None


def test_apply_payment_hints_clears_stale_plan_when_carrier_is_named(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    merged = {"service_query": "colonoscopy", "payer_like": None, "plan_like": "PPO"}
    intent = {"mode": "price"}

    main.apply_payment_hints_from_message("I have Aetna", merged, intent)

    assert merged["payment_mode"] == "insurance"
    assert merged["payer_like"] == "Aetna"
    assert merged["plan_like"] is None


def test_education_query_with_price_keyword_stays_out_of_education(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    assert main.is_health_education_query("what is the cost of a colonoscopy") is False


def test_extract_health_topic_strips_article_prefix(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    assert main.extract_health_topic("What is a colonoscopy?") == "colonoscopy"


def test_clear_topic_switch_context_drops_pricing_state(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    merged = {
        "zipcode": "06119",
        "service_query": "colonoscopy",
        "code_type": "CPT",
        "code": "45378",
        "payment_mode": "insurance",
        "payer_like": "Aetna",
        "plan_like": "PPO",
        "cash_only": False,
        "radius_miles": 25,
        "refiner_choice": "1",
        "refiner_id": 42,
        "variant_confirmed": True,
        "variant_id": 7,
        "variant_name": "Colonoscopy",
        "_variant_candidates": [1, 2, 3],
        "_variant_single": {"id": 7},
        "_awaiting": "payment",
    }

    main.clear_topic_switch_context(merged, clear_zip=False)

    assert merged == {"zipcode": "06119"}


def test_symptom_query_preserves_zip_in_same_message(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    intent = asyncio.run(main.extract_intent("My knee hurts near 06119", {}))

    assert intent["mode"] == "care"
    assert intent["zipcode"] == "06119"

def test_apply_override_changes_mode_to_price(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    intent = {"mode": "general"}
    merged = {"service_query": "colonoscopy"}
    updated = main.apply_intent_override_if_needed(intent, "How much does a colonoscopy cost?", merged, "sess-123")

    assert updated["mode"] == "price"
    assert updated["intent_overridden"] is True
    assert updated["override_reason"] == "cost_keyword_plus_service_query"

def test_apply_override_does_not_change_if_already_price(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    intent = {"mode": "price"}
    merged = {"service_query": "colonoscopy"}
    updated = main.apply_intent_override_if_needed(intent, "How much does a colonoscopy cost?", merged, "sess-123")

    assert updated["mode"] == "price"
    assert updated["intent_overridden"] is True
