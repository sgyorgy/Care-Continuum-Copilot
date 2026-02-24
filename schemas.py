"""
schemas.py — Contract-first, demo-safe, provider-swappable data models.

Design goals:
- Strict + explicit schemas for every API boundary (JSON in/out)
- Timezone-safe timestamps with consistent UTC "Z" JSON serialization
- "Demo never breaks": sensible defaults + normalization (e.g., SOAP sections)
- Built-in safety/provenance hooks (PII suspicion flags, not-medical-advice, provider info)
- Great JSON Schema output (docs / validators / frontend typing)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple
from uuid import UUID, uuid4
import hashlib
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -----------------------------------------------------------------------------
# Constants / helpers
# -----------------------------------------------------------------------------

UTC = timezone.utc

# Lightweight PII/PHI suspicion patterns (heuristic; do NOT claim perfect detection).
# Keep conservative: flag suspicious stuff without blocking flows.
_PII_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # Conservative phone-ish pattern (will have false positives; that's okay for "suspicion" only)
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3}[\s-]?\d{3,4}\b")),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("url", re.compile(r"\bhttps?://\S+\b", re.IGNORECASE)),
    # Generic "ID-like" long digit sequences (kept conservative)
    ("long_digits", re.compile(r"\b\d{9,}\b")),
)

_ALLOWED_EVIDENCE_KEYS = {"subjective", "objective", "assessment", "plan"}

def _ensure_tz_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        # Convert naïve to UTC by assumption; avoids silent local-time bugs.
        return dt.replace(tzinfo=UTC)
    return dt

def _to_utc(dt: datetime) -> datetime:
    return _ensure_tz_aware(dt).astimezone(UTC)

def _dt_to_json(dt: datetime) -> str:
    # Stable ISO8601 "Z" (microseconds trimmed to reduce churn).
    dtu = _to_utc(dt).replace(microsecond=0)
    return dtu.isoformat().replace("+00:00", "Z")

def _compact_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _slug(s: str, *, max_len: int = 48) -> str:
    """
    Simple, UI-friendly slug for tags/flags/metrics.
    """
    s2 = _compact_ws(s).lower()
    s2 = re.sub(r"[^a-z0-9_-]+", "-", s2).strip("-_")
    if len(s2) > max_len:
        s2 = s2[:max_len].rstrip("-_")
    return s2

def _pii_signals(text: str) -> List[str]:
    hits: List[str] = []
    for name, pattern in _PII_PATTERNS:
        if pattern.search(text or ""):
            hits.append(name)
    return hits

def _sha256_hex(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class EntrySource(str, Enum):
    text = "text"
    voice = "voice"


class MoodLabel(str, Enum):
    low = "low"
    neutral = "neutral"
    good = "good"
    unknown = "unknown"


class TrendDirection(str, Enum):
    up = "up"
    down = "down"
    flat = "flat"
    mixed = "mixed"
    unknown = "unknown"


class RiskLevel(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"
    unknown = "unknown"


class DemoMode(str, Enum):
    """
    Golden = fully offline (prebaked outputs)
    Local  = mock/local providers
    Azure  = Azure STT + Azure OpenAI/Foundry (+ optional TA4H)
    """
    golden = "golden"
    local = "local"
    azure = "azure"
    unknown = "unknown"


# -----------------------------------------------------------------------------
# Shared / base models
# -----------------------------------------------------------------------------

class ModelBase(BaseModel):
    """
    Base config:
    - forbid extra fields (catches mistakes early)
    - validate assignment (useful when UI edits objects)
    - allow population by field name even when alias exists (critical for DateRange.from)
    - consistent datetime JSON ("Z") for frontend stability
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        frozen=False,
        populate_by_name=True,
        json_encoders={datetime: _dt_to_json},
    )


class SafetyMeta(ModelBase):
    """
    Non-blocking safety metadata.
    You can surface these in the UI or logs without breaking the flow.
    """
    not_medical_advice: bool = Field(
        default=True,
        description="Always true; this system does not provide medical advice.",
    )
    no_diagnosis_claims: bool = Field(
        default=True,
        description="Always true; system must avoid diagnosing diseases/conditions.",
    )
    synthetic_demo_only: bool = Field(
        default=True,
        description="Always true; demos/tests must use synthetic or anonymized content only.",
    )
    pii_suspected: bool = Field(
        default=False,
        description="Heuristic flag if PII patterns are detected in free text (not perfect).",
    )
    pii_signals: List[str] = Field(
        default_factory=list,
        description="Which heuristic PII patterns were detected (e.g., email, phone).",
        max_length=12,
    )
    redaction_applied: bool = Field(
        default=False,
        description="Whether a redaction step was applied upstream.",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Optional safety notes (e.g., 'objective data not provided; set to Not provided.').",
        max_length=24,
    )


class ProvenanceMeta(ModelBase):
    """
    Where did this output come from? (great for demos + debugging + judging clarity)
    """
    request_id: Optional[str] = Field(default=None, max_length=64)
    demo_mode: DemoMode = Field(default=DemoMode.unknown)
    stt_provider: Optional[str] = Field(default=None, max_length=64)
    llm_provider: Optional[str] = Field(default=None, max_length=64)
    model: Optional[str] = Field(default=None, max_length=128)
    latency_ms: Optional[int] = Field(default=None, ge=0, le=10_000)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    @field_validator("generated_at", mode="before")
    @classmethod
    def _tz_generated(cls, v: Any) -> Any:
        if isinstance(v, datetime):
            return _to_utc(v)
        return v


class DateRange(ModelBase):
    frm: date = Field(..., alias="from", description="Start date (inclusive).")
    to: date = Field(..., description="End date (inclusive).")

    @model_validator(mode="after")
    def _validate_range(self) -> "DateRange":
        if self.frm > self.to:
            raise ValueError("DateRange.from must be <= DateRange.to")
        return self


# -----------------------------------------------------------------------------
# Patient diary domain
# -----------------------------------------------------------------------------

class Symptom(ModelBase):
    name: str = Field(..., min_length=1, max_length=64, description="Symptom name (free text).")
    severity: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="Optional severity scale 0-10 (0 = none).",
    )
    duration_hours: Optional[float] = Field(
        default=None,
        ge=0,
        le=24 * 365,
        description="Optional duration in hours.",
    )
    onset: Optional[datetime] = Field(
        default=None,
        description="Optional onset datetime.",
    )
    notes: Optional[str] = Field(default=None, max_length=500, description="Optional short notes.")

    @field_validator("onset", mode="before")
    @classmethod
    def _tz_symptom_onset(cls, v: Any) -> Any:
        if isinstance(v, datetime):
            return _to_utc(v)
        return v

    @model_validator(mode="after")
    def _normalize(self) -> "Symptom":
        self.name = _compact_ws(self.name)
        if self.notes is not None:
            self.notes = _compact_ws(self.notes)
        return self


class Medication(ModelBase):
    name: str = Field(..., min_length=1, max_length=64, description="Medication name (free text).")
    dose: Optional[str] = Field(default=None, max_length=64, description="Dose (e.g., '200mg').")
    route: Optional[str] = Field(default=None, max_length=32, description="Route (e.g., oral).")
    frequency: Optional[str] = Field(default=None, max_length=64, description="Frequency (e.g., once daily).")
    started_on: Optional[date] = Field(default=None, description="Optional start date.")
    stopped_on: Optional[date] = Field(default=None, description="Optional stop date.")
    notes: Optional[str] = Field(default=None, max_length=500, description="Optional short notes.")

    @model_validator(mode="after")
    def _validate_dates(self) -> "Medication":
        self.name = _compact_ws(self.name)
        for f in ("dose", "route", "frequency", "notes"):
            val = getattr(self, f)
            if isinstance(val, str):
                setattr(self, f, _compact_ws(val))

        if self.started_on and self.stopped_on and self.started_on > self.stopped_on:
            raise ValueError("Medication.started_on must be <= Medication.stopped_on")
        return self


class DiaryEntry(ModelBase):
    id: UUID = Field(default_factory=uuid4, description="UUID for the diary entry.")
    timestamp: datetime = Field(..., description="Entry timestamp (timezone-aware).")
    source: EntrySource = Field(default=EntrySource.text)
    text: str = Field(..., min_length=1, max_length=5000, description="Free-form diary text.")
    mood: MoodLabel = Field(default=MoodLabel.unknown)
    symptoms: List[Symptom] = Field(default_factory=list)
    medications: List[Medication] = Field(default_factory=list)
    tags: List[str] = Field(
        default_factory=list,
        description="Short tags for filtering (normalized).",
        max_length=32,  # list length constraint
    )
    audio_ref: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional reference to audio blob (if voice input).",
    )
    safety: SafetyMeta = Field(default_factory=SafetyMeta)
    provenance: ProvenanceMeta = Field(default_factory=ProvenanceMeta)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _tz_timestamp(cls, v: Any) -> Any:
        if isinstance(v, datetime):
            return _to_utc(v)
        return v

    @field_validator("text", mode="after")
    @classmethod
    def _normalize_text(cls, v: str) -> str:
        return v.strip()

    @field_validator("tags", mode="after")
    @classmethod
    def _normalize_tags(cls, v: List[str]) -> List[str]:
        cleaned: List[str] = []
        for tag in v or []:
            t = _slug(tag, max_len=24)
            if t and t not in cleaned:
                cleaned.append(t)
        return cleaned[:32]

    @model_validator(mode="after")
    def _attach_safety(self) -> "DiaryEntry":
        # Non-blocking PII suspicion
        signals = _pii_signals(self.text)
        if signals:
            self.safety.pii_suspected = True
            self.safety.pii_signals = sorted(set(self.safety.pii_signals + signals))
            self.safety.notes.append("PII suspected in diary text (heuristic).")

        # Demo-safe: voice entries ideally have audio_ref, but do not hard-fail.
        if self.source == EntrySource.voice and not self.audio_ref:
            self.safety.notes.append("Voice entry missing audio_ref; continuing demo-safe.")

        return self


class TrendPoint(ModelBase):
    """
    A single time series point. Keep flexible for quick iteration.
    """
    date: date = Field(..., description="Point date.")
    metric: str = Field(..., min_length=1, max_length=64, description="Metric key (e.g., 'mood', 'headache_severity').")
    value: Optional[float] = Field(default=None, description="Numeric value if applicable.")
    label: Optional[str] = Field(default=None, max_length=64, description="Optional categorical label.")
    entry_id: Optional[UUID] = Field(default=None, description="Optional reference to the originating diary entry.")
    notes: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _norm(self) -> "TrendPoint":
        self.metric = _slug(self.metric, max_len=64) or self.metric.strip()
        if self.label is not None:
            self.label = _compact_ws(self.label)
        if self.notes is not None:
            self.notes = _compact_ws(self.notes)
        return self


class TrendInsight(ModelBase):
    metric: str = Field(..., min_length=1, max_length=64)
    direction: TrendDirection = Field(default=TrendDirection.unknown)
    notes: Optional[str] = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _norm(self) -> "TrendInsight":
        self.metric = _slug(self.metric, max_len=64) or self.metric.strip()
        if self.notes is not None:
            self.notes = _compact_ws(self.notes)
        return self


class DiarySummary(ModelBase):
    period: DateRange = Field(..., description="Summary period.")
    highlights: List[str] = Field(
        default_factory=list,
        description="Key changes or patterns (bullets).",
        max_length=12,
    )
    trends: List[TrendInsight] = Field(default_factory=list, max_length=12)
    gentle_suggestions: List[str] = Field(
        default_factory=list,
        description="Non-diagnostic suggestions (e.g., rest, hydration, talk to clinician).",
        max_length=8,
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.unknown,
        description="Optional risk signal (heuristic); do not over-interpret.",
    )
    risk_flags: List[str] = Field(
        default_factory=list,
        description="Short normalized labels (e.g., 'persistent_symptoms', 'worsening_trend').",
        max_length=12,
    )
    shareable_previsit_summary: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Clinician-friendly paragraph + bullets; copy/export target.",
    )
    safety: SafetyMeta = Field(default_factory=SafetyMeta)
    provenance: ProvenanceMeta = Field(default_factory=ProvenanceMeta)

    @field_validator("highlights", "gentle_suggestions", mode="after")
    @classmethod
    def _clean_bullets(cls, v: List[str]) -> List[str]:
        cleaned: List[str] = []
        for item in v or []:
            s = _compact_ws(item)
            if s:
                cleaned.append(s)
        # De-dup while preserving order
        out: List[str] = []
        seen = set()
        for s in cleaned:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    @field_validator("risk_flags", mode="after")
    @classmethod
    def _norm_flags(cls, v: List[str]) -> List[str]:
        out: List[str] = []
        for f in v or []:
            s = _slug(f, max_len=32)
            if s and s not in out:
                out.append(s)
        return out[:12]

    @field_validator("shareable_previsit_summary", mode="after")
    @classmethod
    def _clean_shareable(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _attach_safety(self) -> "DiarySummary":
        signals = _pii_signals(self.shareable_previsit_summary)
        if signals:
            self.safety.pii_suspected = True
            self.safety.pii_signals = sorted(set(self.safety.pii_signals + signals))
            self.safety.notes.append("PII suspected in shareable summary (heuristic).")
        return self

    def to_plaintext(self) -> str:
        """
        A nice human-readable export (copy/paste).
        """
        lines: List[str] = []
        lines.append(f"Pre-visit summary ({self.period.frm.isoformat()} → {self.period.to.isoformat()})")
        if self.highlights:
            lines.append("\nHighlights:")
            lines.extend([f"- {h}" for h in self.highlights])
        if self.trends:
            lines.append("\nTrends:")
            for t in self.trends:
                note = f" — {t.notes}" if t.notes else ""
                lines.append(f"- {t.metric}: {t.direction.value}{note}")
        if self.gentle_suggestions:
            lines.append("\nSuggestions (non-diagnostic):")
            lines.extend([f"- {g}" for g in self.gentle_suggestions])
        lines.append("\n---\n")
        lines.append(self.shareable_previsit_summary.strip())
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Clinician documentation domain (SOAP)
# -----------------------------------------------------------------------------

class SoapNote(ModelBase):
    """
    SOAP is intentionally simple: lists of clinician-readable bullet lines.
    We aggressively normalize into a safe, non-empty structure.
    """
    subjective: List[str] = Field(default_factory=list, description="Patient-reported symptoms/history.")
    objective: List[str] = Field(default_factory=list, description="Vitals/exam/labs if provided; otherwise 'Not provided.'")
    assessment: List[str] = Field(default_factory=list, description="Clinical impression WITHOUT diagnosis claims.")
    plan: List[str] = Field(default_factory=list, description="Next steps (tests, follow-up, education).")

    # Anti-hallucination / judging-friendly extras (optional):
    not_provided: List[str] = Field(
        default_factory=list,
        description="Which content areas were missing from input (e.g., ['objective']).",
        max_length=16,
    )
    evidence_quotes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Optional supporting snippets keyed by section: subjective/objective/assessment/plan (short, few items).",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata (e.g., source='transcript', confidence notes).",
    )
    safety: SafetyMeta = Field(default_factory=SafetyMeta)
    provenance: ProvenanceMeta = Field(default_factory=ProvenanceMeta)

    _MAX_LINE_LEN: ClassVar[int] = 400
    _MAX_EVIDENCE_QUOTES_PER_KEY: ClassVar[int] = 3
    _MAX_EVIDENCE_QUOTE_LEN: ClassVar[int] = 220

    @staticmethod
    def _normalize_lines(lines: Sequence[str]) -> List[str]:
        cleaned: List[str] = []
        for line in lines or []:
            s = _compact_ws(str(line))
            if not s:
                continue
            if len(s) > SoapNote._MAX_LINE_LEN:
                s = s[: SoapNote._MAX_LINE_LEN].rstrip() + "…"
            cleaned.append(s)
        # De-dup case-insensitively while preserving order
        out: List[str] = []
        seen = set()
        for s in cleaned:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    @staticmethod
    def _normalize_evidence(evidence: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for k, quotes in (evidence or {}).items():
            key = _slug(k, max_len=24)
            if key not in _ALLOWED_EVIDENCE_KEYS:
                continue
            cleaned: List[str] = []
            for q in quotes or []:
                s = _compact_ws(str(q))
                if not s:
                    continue
                if len(s) > SoapNote._MAX_EVIDENCE_QUOTE_LEN:
                    s = s[: SoapNote._MAX_EVIDENCE_QUOTE_LEN].rstrip() + "…"
                cleaned.append(s)
            # de-dup preserve order
            uniq: List[str] = []
            seen = set()
            for s in cleaned:
                ks = s.lower()
                if ks not in seen:
                    seen.add(ks)
                    uniq.append(s)
            if uniq:
                out[key] = uniq[: SoapNote._MAX_EVIDENCE_QUOTES_PER_KEY]
        return out

    @model_validator(mode="after")
    def _canonicalize(self) -> "SoapNote":
        self.subjective = self._normalize_lines(self.subjective)
        self.objective = self._normalize_lines(self.objective)
        self.assessment = self._normalize_lines(self.assessment)
        self.plan = self._normalize_lines(self.plan)
        self.evidence_quotes = self._normalize_evidence(self.evidence_quotes)

        # Ensure non-empty sections with safe defaults.
        missing: List[str] = []
        if not self.subjective:
            self.subjective = ["Not provided."]
            missing.append("subjective")
            self.safety.notes.append("Subjective empty; set to 'Not provided.'")
        if not self.objective:
            self.objective = ["Not provided."]
            missing.append("objective")
            self.safety.notes.append("Objective empty; set to 'Not provided.'")
        if not self.assessment:
            self.assessment = ["Not provided."]
            missing.append("assessment")
            self.safety.notes.append("Assessment empty; set to 'Not provided.'")
        if not self.plan:
            self.plan = ["Not provided."]
            missing.append("plan")
            self.safety.notes.append("Plan empty; set to 'Not provided.'")

        # Merge into not_provided (stable ordering)
        merged = []
        for m in (self.not_provided or []) + missing:
            s = _slug(m, max_len=24)
            if s and s not in merged:
                merged.append(s)
        self.not_provided = merged[:16]

        # Non-blocking PII suspicion across sections
        joined = "\n".join(self.subjective + self.objective + self.assessment + self.plan)
        signals = _pii_signals(joined)
        if signals:
            self.safety.pii_suspected = True
            self.safety.pii_signals = sorted(set(self.safety.pii_signals + signals))
            self.safety.notes.append("PII suspected in SOAP content (heuristic).")

        return self

    def to_plaintext(self) -> str:
        def section(title: str, lines: List[str]) -> str:
            bullets = "\n".join([f"- {l}" for l in lines])
            return f"{title}:\n{bullets}"

        parts = [
            section("S — Subjective", self.subjective),
            section("O — Objective", self.objective),
            section("A — Assessment", self.assessment),
            section("P — Plan", self.plan),
        ]
        return "\n\n".join(parts)


class ClinicalNoteResult(ModelBase):
    """
    Endpoint output contract for: transcript/dictation -> cleaned text + SOAP.
    """
    cleaned_transcript: str = Field(..., min_length=1, max_length=12000)
    soap_note: SoapNote
    source_digest: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="Optional SHA-256 digest of input transcript/audio for traceability without storing raw data.",
    )
    safety: SafetyMeta = Field(default_factory=SafetyMeta)
    provenance: ProvenanceMeta = Field(default_factory=ProvenanceMeta)

    @field_validator("cleaned_transcript", mode="after")
    @classmethod
    def _clean_transcript(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _attach_safety(self) -> "ClinicalNoteResult":
        signals = _pii_signals(self.cleaned_transcript)
        if signals:
            self.safety.pii_suspected = True
            self.safety.pii_signals = sorted(set(self.safety.pii_signals + signals))
            self.safety.notes.append("PII suspected in cleaned transcript (heuristic).")
        return self


# -----------------------------------------------------------------------------
# Utility schemas (API helpers)
# -----------------------------------------------------------------------------

class RedactionRequest(ModelBase):
    text: str = Field(..., min_length=1, max_length=20000)


class RedactionResult(ModelBase):
    redacted_text: str = Field(..., min_length=1, max_length=20000)
    replaced: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts per redacted category (e.g., {'email': 1}).",
    )
    safety: SafetyMeta = Field(default_factory=SafetyMeta)
    provenance: ProvenanceMeta = Field(default_factory=ProvenanceMeta)


class ErrorEnvelope(ModelBase):
    """
    Consistent, frontend-friendly error contract.
    """
    error: str = Field(..., min_length=1, max_length=200)
    detail: Optional[str] = Field(default=None, max_length=2000)
    hint: Optional[str] = Field(default=None, max_length=500)
    request_id: Optional[str] = Field(default=None, max_length=64)


# -----------------------------------------------------------------------------
# Endpoint-specific request/response contracts
# (prevents clients from sending server-owned fields like id/safety/provenance)
# -----------------------------------------------------------------------------

class DiaryEntryCreateRequest(ModelBase):
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Optional; if omitted server may set now().",
    )
    source: EntrySource = Field(default=EntrySource.text)
    text: str = Field(..., min_length=1, max_length=5000)
    mood: MoodLabel = Field(default=MoodLabel.unknown)
    tags: List[str] = Field(default_factory=list, max_length=32)
    audio_ref: Optional[str] = Field(default=None, max_length=256)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _tz_optional_ts(cls, v: Any) -> Any:
        if isinstance(v, datetime):
            return _to_utc(v)
        return v


class DiaryEntryCreateResponse(ModelBase):
    entry: DiaryEntry


class DiaryTrendsQuery(ModelBase):
    period: DateRange = Field(..., description="Trend period.")
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Optional list of metric keys to include (if omitted, include all available).",
        max_length=32,
    )

    @field_validator("metrics", mode="after")
    @classmethod
    def _norm_metrics(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        out: List[str] = []
        for m in v:
            s = _slug(m, max_len=64)
            if s and s not in out:
                out.append(s)
        return out[:32]


class DiaryTrendsResponse(ModelBase):
    period: DateRange
    points: List[TrendPoint] = Field(default_factory=list, max_length=2000)
    insights: List[TrendInsight] = Field(default_factory=list, max_length=24)
    safety: SafetyMeta = Field(default_factory=SafetyMeta)
    provenance: ProvenanceMeta = Field(default_factory=ProvenanceMeta)


class DiarySummaryRequest(ModelBase):
    period: DateRange = Field(..., description="Summary period.")
    # Optional explicit entries. If omitted, backend may load by period.
    entries: Optional[List[DiaryEntry]] = Field(default=None, description="Optional diary entries to summarize.", max_length=200)


class DiarySummaryResponse(ModelBase):
    summary: DiarySummary


class SoapFromTranscriptRequest(ModelBase):
    transcript: str = Field(..., min_length=1, max_length=20000)

    def digest(self) -> str:
        return _sha256_hex(self.transcript)


class SoapFromTranscriptResponse(ModelBase):
    result: ClinicalNoteResult


# -----------------------------------------------------------------------------
# JSON Schema export (docs / frontend typing)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SchemaExport:
    name: str
    model: type[BaseModel]


EXPORT_MODELS: Tuple[SchemaExport, ...] = (
    SchemaExport("SafetyMeta", SafetyMeta),
    SchemaExport("ProvenanceMeta", ProvenanceMeta),
    SchemaExport("DateRange", DateRange),
    SchemaExport("Symptom", Symptom),
    SchemaExport("Medication", Medication),
    SchemaExport("DiaryEntry", DiaryEntry),
    SchemaExport("TrendPoint", TrendPoint),
    SchemaExport("TrendInsight", TrendInsight),
    SchemaExport("DiarySummary", DiarySummary),
    SchemaExport("SoapNote", SoapNote),
    SchemaExport("ClinicalNoteResult", ClinicalNoteResult),
    SchemaExport("RedactionRequest", RedactionRequest),
    SchemaExport("RedactionResult", RedactionResult),
    SchemaExport("ErrorEnvelope", ErrorEnvelope),
    # Endpoint contracts
    SchemaExport("DiaryEntryCreateRequest", DiaryEntryCreateRequest),
    SchemaExport("DiaryEntryCreateResponse", DiaryEntryCreateResponse),
    SchemaExport("DiaryTrendsQuery", DiaryTrendsQuery),
    SchemaExport("DiaryTrendsResponse", DiaryTrendsResponse),
    SchemaExport("DiarySummaryRequest", DiarySummaryRequest),
    SchemaExport("DiarySummaryResponse", DiarySummaryResponse),
    SchemaExport("SoapFromTranscriptRequest", SoapFromTranscriptRequest),
    SchemaExport("SoapFromTranscriptResponse", SoapFromTranscriptResponse),
)

def get_all_json_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Returns dict: {ModelName: JSONSchemaDict}
    Great for:
    - generating OpenAPI components
    - frontend type generation (via quicktype or similar)
    """
    out: Dict[str, Dict[str, Any]] = {}
    for item in EXPORT_MODELS:
        out[item.name] = item.model.model_json_schema()
    return out


__all__ = [
    # enums
    "EntrySource",
    "MoodLabel",
    "TrendDirection",
    "RiskLevel",
    "DemoMode",
    # shared
    "SafetyMeta",
    "ProvenanceMeta",
    "DateRange",
    # diary
    "Symptom",
    "Medication",
    "DiaryEntry",
    "TrendPoint",
    "TrendInsight",
    "DiarySummary",
    # clinician
    "SoapNote",
    "ClinicalNoteResult",
    # utils
    "RedactionRequest",
    "RedactionResult",
    "ErrorEnvelope",
    # endpoint contracts
    "DiaryEntryCreateRequest",
    "DiaryEntryCreateResponse",
    "DiaryTrendsQuery",
    "DiaryTrendsResponse",
    "DiarySummaryRequest",
    "DiarySummaryResponse",
    "SoapFromTranscriptRequest",
    "SoapFromTranscriptResponse",
    # schema export
    "get_all_json_schemas",
]
