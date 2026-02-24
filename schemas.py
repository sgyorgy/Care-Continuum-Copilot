"""
schemas.py — Data models for AI Health Diary & Clinical Note Assistant
MIT Minds & Machines Hackathon — Healthcare Challenge

All models use Pydantic v2 for runtime validation and JSON schema export.
No PHI / no real patient data — synthetic/demo content only.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────
# Shared enums & primitives
# ──────────────────────────────────────────────

class MoodLevel(str, Enum):
    """Subjective mood on a 5-point scale."""
    very_bad  = "very_bad"
    bad       = "bad"
    neutral   = "neutral"
    good      = "good"
    very_good = "very_good"


class Severity(str, Enum):
    """Generic 1–5 severity / intensity scale."""
    minimal  = "minimal"   # 1
    mild     = "mild"      # 2
    moderate = "moderate"  # 3
    severe   = "severe"    # 4
    extreme  = "extreme"   # 5


class InputMode(str, Enum):
    text  = "text"
    voice = "voice"


UNKNOWN = "unknown"   # sentinel for missing objective fields in SOAP


# ──────────────────────────────────────────────
# Patient-side models
# ──────────────────────────────────────────────

class DiaryEntry(BaseModel):
    """
    A single patient journal entry (text or voice-transcribed).
    Stored as-is; PII redaction happens before LLM processing.
    """
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC time of entry creation.",
    )
    text: Annotated[str, Field(min_length=1, max_length=4000)]
    mood: Optional[MoodLevel] = Field(
        default=None,
        description="Optional self-reported mood at the time of entry.",
    )
    symptoms: list[str] = Field(
        default_factory=list,
        description="Free-text symptom tags, e.g. ['headache', 'fatigue'].",
        max_length=20,
    )
    meds: list[str] = Field(
        default_factory=list,
        description="Medications taken or mentioned in this entry.",
        max_length=20,
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Arbitrary user-defined labels (e.g. 'after-exercise', 'poor-sleep').",
        max_length=30,
    )
    input_mode: InputMode = Field(
        default=InputMode.text,
        description="Whether the entry was typed or voice-transcribed.",
    )
    redacted: bool = Field(
        default=False,
        description="True once PII/PHI redaction has been applied to `text`.",
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "timestamp": "2025-06-10T08:30:00Z",
            "text": "Woke up with a mild headache. Took ibuprofen 400 mg after breakfast.",
            "mood": "neutral",
            "symptoms": ["headache"],
            "meds": ["ibuprofen 400 mg"],
            "tags": ["morning"],
            "input_mode": "text",
            "redacted": True,
        }
    ]}}


class TrendPoint(BaseModel):
    """
    A single data point on a symptom/sentiment timeline.
    Used to build the trend chart on the patient dashboard.
    """
    date: date = Field(description="Calendar date of the data point.")
    symptom: str = Field(
        description="Symptom or dimension being tracked, e.g. 'headache' or 'mood'.",
        min_length=1,
        max_length=80,
    )
    severity: Optional[Severity] = Field(
        default=None,
        description="Severity level if the symptom is quantifiable.",
    )
    sentiment: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Sentiment score [-1 negative … +1 positive] from diary text.",
    )
    entry_count: int = Field(
        default=1,
        ge=1,
        description="Number of diary entries aggregated into this point.",
    )

    model_config = {"json_schema_extra": {"examples": [
        {"date": "2025-06-10", "symptom": "headache", "severity": "mild", "sentiment": -0.3, "entry_count": 1}
    ]}}


class DiarySummary(BaseModel):
    """
    Weekly (or custom-period) AI-generated summary of diary entries.
    Output of the `diary_summary` prompt — never contains a diagnosis.
    """
    period_start: date
    period_end: date
    entry_count: int = Field(ge=0)

    bullets: list[str] = Field(
        description="3–7 concise observations about the period.",
        min_length=1,
        max_length=10,
    )
    risks: list[str] = Field(
        default_factory=list,
        description=(
            "Patterns that *might* warrant attention "
            "(e.g. 'Headaches reported on 4 of 7 days'). "
            "No diagnosis — factual observation only."
        ),
        max_length=5,
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Gentle, non-diagnostic suggestions "
            "(e.g. 'Consider rest', 'Talk to a clinician if symptoms persist'). "
            "Maximum specificity: lifestyle / self-care."
        ),
        max_length=5,
    )
    shareable_note: str = Field(
        description=(
            "A short, patient-friendly paragraph suitable for sharing "
            "with a clinician as a pre-visit summary. "
            "Plain language, no medical jargon."
        ),
        min_length=10,
        max_length=800,
    )
    trend_points: list[TrendPoint] = Field(
        default_factory=list,
        description="Structured trend data derived from the same period.",
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = Field(
        default="mock-v1",
        description="LLM provider / version tag for auditability.",
    )

    @model_validator(mode="after")
    def _period_order(self) -> "DiarySummary":
        if self.period_end < self.period_start:
            raise ValueError("`period_end` must not be before `period_start`.")
        return self

    model_config = {"json_schema_extra": {"examples": [
        {
            "period_start": "2025-06-03",
            "period_end":   "2025-06-09",
            "entry_count":  7,
            "bullets": [
                "Headaches reported on 4 of 7 days, mostly in the morning.",
                "Mood was neutral-to-good on days with adequate sleep.",
                "Ibuprofen taken on 3 occasions.",
            ],
            "risks": ["Recurring morning headaches may warrant follow-up."],
            "suggestions": [
                "Consider tracking sleep duration to spot correlations.",
                "Talk to a clinician if headaches persist beyond this week.",
            ],
            "shareable_note": (
                "This week I experienced frequent morning headaches (4/7 days) "
                "and took ibuprofen on three occasions. My mood was generally "
                "neutral. I'd like to discuss possible causes at my next visit."
            ),
        }
    ]}}


# ──────────────────────────────────────────────
# Clinician-side models
# ──────────────────────────────────────────────

class SoapMetadata(BaseModel):
    """Audit / provenance data attached to every generated SOAP note."""
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = Field(default="mock-v1")
    source_transcript_length: Optional[int] = Field(
        default=None,
        ge=0,
        description="Character count of the raw input transcript.",
    )
    redacted: bool = Field(
        default=False,
        description="True once PII/PHI redaction has been applied.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional model confidence score (0–1).",
    )


class SoapNote(BaseModel):
    """
    Structured clinical note in SOAP format.
    Produced from a clinician's voice dictation or typed transcript.

    Hard rules (enforced in the prompt AND here):
      • Objective fields never contain invented data — use UNKNOWN sentinel.
      • Assessment is a working impression, NOT a confirmed diagnosis.
      • Plan lists next steps only; no prescriptions invented by the AI.
    """
    id: UUID = Field(default_factory=uuid4)

    # ── S — Subjective ──────────────────────────────────────────────
    subjective: str = Field(
        description=(
            "Chief complaint and history of present illness in the patient's "
            "own words as reported by the clinician. "
            f"Use '{UNKNOWN}' if not mentioned."
        ),
        min_length=1,
        max_length=2000,
    )

    # ── O — Objective ───────────────────────────────────────────────
    objective: str = Field(
        description=(
            "Measurable / observable findings: vitals, exam findings, "
            "test results. AI MUST NOT invent values. "
            f"Use '{UNKNOWN}' for any field not present in the transcript."
        ),
        min_length=1,
        max_length=2000,
    )

    # ── A — Assessment ──────────────────────────────────────────────
    assessment: str = Field(
        description=(
            "Working clinical impression based solely on the transcript. "
            "Phrased as 'likely' / 'consistent with' — never a definitive diagnosis. "
            f"Use '{UNKNOWN}' if insufficient information."
        ),
        min_length=1,
        max_length=1000,
    )

    # ── P — Plan ────────────────────────────────────────────────────
    plan: str = Field(
        description=(
            "Recommended next steps: follow-up, referrals, tests ordered, "
            "patient instructions. Next-step framing, not a final clinical order. "
            f"Use '{UNKNOWN}' if not discussed."
        ),
        min_length=1,
        max_length=1500,
    )

    metadata: SoapMetadata = Field(default_factory=SoapMetadata)

    @field_validator("subjective", "objective", "assessment", "plan", mode="before")
    @classmethod
    def _strip_and_fallback(cls, v: str) -> str:
        """Strip whitespace; substitute empty strings with the UNKNOWN sentinel."""
        v = str(v).strip()
        return v if v else UNKNOWN

    model_config = {"json_schema_extra": {"examples": [
        {
            "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
            "subjective": (
                "Patient is a 35-year-old presenting with a 3-day history of "
                "throbbing left-sided headache, photophobia, and nausea. "
                "No aura reported. Similar episodes in the past."
            ),
            "objective": (
                "BP unknown. HR unknown. Neurological exam: no focal deficits reported. "
                "No fever mentioned."
            ),
            "assessment": (
                "Presentation consistent with episodic migraine without aura. "
                "Secondary causes not yet excluded."
            ),
            "plan": (
                "1. Consider initiating acute migraine therapy (discuss with patient). "
                "2. Recommend headache diary for frequency tracking. "
                "3. Follow-up in 2 weeks or sooner if symptoms worsen."
            ),
            "metadata": {
                "generated_at": "2025-06-10T09:00:00Z",
                "model_version": "mock-v1",
                "source_transcript_length": 342,
                "redacted": True,
                "confidence": 0.87,
            },
        }
    ]}}


# ──────────────────────────────────────────────
# API request / response wrappers
# ──────────────────────────────────────────────

class DiaryEntryRequest(BaseModel):
    """POST /diary/entry — body."""
    text: Annotated[str, Field(min_length=1, max_length=4000)]
    mood: Optional[MoodLevel] = None
    symptoms: list[str] = Field(default_factory=list, max_length=20)
    meds: list[str] = Field(default_factory=list, max_length=20)
    tags: list[str] = Field(default_factory=list, max_length=30)
    input_mode: InputMode = InputMode.text


class DiaryEntryResponse(BaseModel):
    """POST /diary/entry — response."""
    entry: DiaryEntry
    message: str = "Entry saved successfully."


class TrendQueryParams(BaseModel):
    """GET /diary/trends query parameters (validated separately in routes)."""
    from_date: date = Field(alias="from")
    to_date: date = Field(alias="to")
    symptoms: Optional[list[str]] = Field(default=None, description="Filter to specific symptoms.")

    model_config = {"populate_by_name": True}


class TrendResponse(BaseModel):
    """GET /diary/trends — response."""
    from_date: date
    to_date: date
    points: list[TrendPoint]


class SummaryRequest(BaseModel):
    """POST /diary/summary — body."""
    from_date: date
    to_date: date
    entry_ids: Optional[list[UUID]] = Field(
        default=None,
        description="Explicit list of entry IDs to summarise. If omitted, uses date range.",
    )


class SummaryResponse(BaseModel):
    """POST /diary/summary — response."""
    summary: DiarySummary


class SoapRequest(BaseModel):
    """POST /clinician/soap — body."""
    transcript: Annotated[str, Field(min_length=1, max_length=8000)]
    already_transcribed: bool = Field(
        default=True,
        description="False if `transcript` is raw audio bytes (base64). True if already text.",
    )
    language: str = Field(default="en-US")


class SoapResponse(BaseModel):
    """POST /clinician/soap — response."""
    note: SoapNote


class RedactRequest(BaseModel):
    """POST /utils/redact — body."""
    text: Annotated[str, Field(min_length=1, max_length=16000)]


class RedactResponse(BaseModel):
    """POST /utils/redact — response."""
    redacted_text: str
    replacements_made: int = Field(ge=0, description="Number of tokens replaced.")


# ──────────────────────────────────────────────
# Generic API envelope (error-safe demo wrapper)
# ──────────────────────────────────────────────

class APIStatus(str, Enum):
    ok      = "ok"
    error   = "error"
    mock    = "mock"   # response served from golden-output cache


class APIResponse(BaseModel):
    """
    Top-level envelope for every endpoint.
    Ensures demo-safe JSON even on failure (fallback to mock data).
    """
    status: APIStatus = APIStatus.ok
    data: Optional[dict] = None
    error: Optional[str] = None
    provider: str = Field(
        default="mock",
        description="LLM/STT provider that served this response.",
    )

    @classmethod
    def ok(cls, data: dict, provider: str = "mock") -> "APIResponse":
        return cls(status=APIStatus.ok, data=data, provider=provider)

    @classmethod
    def from_mock(cls, data: dict) -> "APIResponse":
        return cls(status=APIStatus.mock, data=data, provider="mock")

    @classmethod
    def error(cls, message: str) -> "APIResponse":
        return cls(status=APIStatus.error, error=message)


# ──────────────────────────────────────────────
# JSON Schema export helper (useful for prompts)
# ──────────────────────────────────────────────

def export_schemas() -> dict:
    """Return JSON schemas for all core models — injected into LLM prompts."""
    return {
        "DiaryEntry":    DiaryEntry.model_json_schema(),
        "DiarySummary":  DiarySummary.model_json_schema(),
        "TrendPoint":    TrendPoint.model_json_schema(),
        "SoapNote":      SoapNote.model_json_schema(),
        "SoapMetadata":  SoapMetadata.model_json_schema(),
    }


if __name__ == "__main__":
    import json

    print("=== Exported JSON Schemas ===\n")
    for name, schema in export_schemas().items():
        print(f"--- {name} ---")
        print(json.dumps(schema, indent=2))
        print()
