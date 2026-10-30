"""FastAPI production serving engine for TriageLM with latency tracking."""

import json
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.schema import TriageOutput, validate_triage_json

app = FastAPI(
    title="TriageLM Inference API",
    description="Low-latency customer support triage and strict JSON extraction engine",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TriageRequest(BaseModel):
    text: str = Field(..., description="Customer message or support ticket text", min_length=2)
    stream: bool = Field(default=False, description="Whether to stream response tokens")


class TriageResponse(BaseModel):
    status: str
    latency_ms: float
    data: Optional[TriageOutput] = None
    raw_output: str
    error: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "TriageLM"}


@app.post("/triage", response_model=TriageResponse)
async def triage_ticket(request: TriageRequest):
    """Parses a customer support message and returns validated structured JSON."""
    start_time = time.perf_counter()

    try:
        # Placeholder for inference call (swappable with llama-cpp or transformers pipeline)
        # When fully loaded, pipeline generates the structured output
        raw_output = """{
          "intent": "lost_or_stolen_card",
          "sentiment": "angry",
          "urgency": "critical",
          "route_to": "card_security",
          "action_required": true,
          "entities": {"card_type": "debit"}
        }"""

        is_valid, triage_obj, err_msg = validate_triage_json(raw_output)
        latency = (time.perf_counter() - start_time) * 1000

        if not is_valid:
            return TriageResponse(
                status="validation_error",
                latency_ms=round(latency, 2),
                raw_output=raw_output,
                error=err_msg,
            )

        return TriageResponse(
            status="success",
            latency_ms=round(latency, 2),
            data=triage_obj,
            raw_output=raw_output,
        )

    except Exception as exc:
        latency = (time.perf_counter() - start_time) * 1000
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "latency_ms": round(latency, 2)},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8000, reload=True)
