"""Unit tests for TriageLM schema and validation utilities."""

import pytest
from src.schema import (
    ExtractedEntities,
    RoutingTeamEnum,
    SentimentEnum,
    TriageOutput,
    UrgencyEnum,
    extract_json_block,
    validate_triage_json,
)


def test_valid_triage_output_serialization():
    """Tests that a valid TriageOutput instantiates and dumps valid JSON."""
    obj = TriageOutput(
        intent="lost_or_stolen_card",
        sentiment=SentimentEnum.ANGRY,
        urgency=UrgencyEnum.CRITICAL,
        route_to=RoutingTeamEnum.CARD_SECURITY,
        action_required=True,
        entities=ExtractedEntities(card_type="credit"),
    )

    json_str = obj.model_dump_json()
    is_valid, parsed, err = validate_triage_json(json_str)

    assert is_valid is True
    assert err is None
    assert parsed.intent == "lost_or_stolen_card"
    assert parsed.sentiment == SentimentEnum.ANGRY
    assert parsed.urgency == UrgencyEnum.CRITICAL
    assert parsed.route_to == RoutingTeamEnum.CARD_SECURITY
    assert parsed.action_required is True
    assert parsed.entities.card_type == "credit"


def test_markdown_code_fence_stripping():
    """Tests that markdown wrapped JSON from LLM output is parsed successfully."""
    raw_markdown = """
    Here is the requested triage response:
    ```json
    {
      "intent": "card_payment_fee_charged",
      "sentiment": "frustrated",
      "urgency": "medium",
      "route_to": "dispute_management",
      "action_required": true,
      "entities": {
        "amount": "150",
        "currency": "INR"
      }
    }
    ```
    Hope this helps!
    """
    is_valid, parsed, err = validate_triage_json(raw_markdown)

    assert is_valid is True
    assert err is None
    assert parsed.intent == "card_payment_fee_charged"
    assert parsed.entities.amount == "150"
    assert parsed.entities.currency == "INR"


def test_invalid_json_handling():
    """Tests that invalid JSON returns false with descriptive error."""
    invalid_raw = "{ intent: 'broken', 'sentiment' "
    is_valid, parsed, err = validate_triage_json(invalid_raw)

    assert is_valid is False
    assert parsed is None
    assert "JSONDecodeError" in err


def test_schema_violation_handling():
    """Tests that invalid enum values trigger a schema validation error."""
    invalid_enum_raw = """
    {
      "intent": "refund",
      "sentiment": "super_angry_invalid",
      "urgency": "critical",
      "route_to": "card_security",
      "action_required": true,
      "entities": {}
    }
    """
    is_valid, parsed, err = validate_triage_json(invalid_enum_raw)

    assert is_valid is False
    assert parsed is None
    assert "SchemaValidationError" in err
