"""Canonical Pydantic schema and JSON validation utilities for TriageLM."""

import json
import re
from enum import Enum
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError


class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class UrgencyEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RoutingTeamEnum(str, Enum):
    CARD_SECURITY = "card_security"
    PAYMENTS_OPS = "payments_ops"
    DISPUTE_MANAGEMENT = "dispute_management"
    ACCOUNT_SERVICES = "account_services"
    GENERAL_SUPPORT = "general_support"


class ExtractedEntities(BaseModel):
    amount: Optional[str] = Field(
        default=None, description="Monetary amount mentioned (e.g., '$50', '2500 INR', '500')"
    )
    currency: Optional[str] = Field(
        default=None, description="Currency code or symbol (e.g., 'USD', 'INR', 'EUR', 'GBP')"
    )
    card_type: Optional[str] = Field(
        default=None, description="Card category (e.g., 'debit', 'credit', 'virtual')"
    )
    merchant: Optional[str] = Field(
        default=None, description="Merchant or vendor name (e.g., 'Amazon', 'Zomato', 'ATM')"
    )
    date: Optional[str] = Field(
        default=None, description="Timeframe or date mentioned (e.g., 'yesterday', 'today', '2 days ago')"
    )
    transaction_id: Optional[str] = Field(
        default=None, description="Reference number or transaction ID if provided"
    )

    class Config:
        extra = "ignore"


class TriageOutput(BaseModel):
    """Target output schema for multi-task customer triage."""

    intent: str = Field(
        ..., description="Standardized intent category of the customer inquiry"
    )
    sentiment: SentimentEnum = Field(
        ..., description="Customer sentiment: positive, neutral, frustrated, angry"
    )
    urgency: UrgencyEnum = Field(
        ..., description="Ticket urgency: low, medium, high, critical"
    )
    route_to: RoutingTeamEnum = Field(
        ..., description="Target support team for escalation"
    )
    action_required: bool = Field(
        ..., description="Whether human or automated intervention is required"
    )
    entities: ExtractedEntities = Field(
        default_factory=ExtractedEntities,
        description="Structured entities extracted directly from message",
    )

    class Config:
        extra = "ignore"
        use_enum_values = True


def extract_json_block(text: str) -> str:
    """Extracts JSON substring from LLM response, stripping markdown fencing if present."""
    text = text.strip()
    # Match ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Find the outermost curly braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()

    return text


def validate_triage_json(raw_text: str) -> Tuple[bool, Optional[TriageOutput], Optional[str]]:
    """Validates raw LLM output against the canonical Pydantic schema.

    Returns:
        (is_valid, parsed_object, error_message)
    """
    clean_json_str = extract_json_block(raw_text)
    try:
        data = json.loads(clean_json_str)
        triage_obj = TriageOutput.model_validate(data)
        return True, triage_obj, None
    except json.JSONDecodeError as exc:
        return False, None, f"JSONDecodeError: {str(exc)}"
    except ValidationError as exc:
        return False, None, f"SchemaValidationError: {str(exc)}"
    except Exception as exc:
        return False, None, f"UnexpectedError: {str(exc)}"


def get_canonical_json_schema() -> Dict[str, Any]:
    """Returns the JSON Schema specification for prompt guidance and constrained decoding."""
    return TriageOutput.model_json_schema()
