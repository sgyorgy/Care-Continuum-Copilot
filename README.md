# Care Continuum Copilot — Spec (Hackathon Build)

**Tagline:** From patient diary to clinician-ready SOAP notes — structured insights across the care continuum.

## 1) Context & Goal

Modern healthcare produces a lot of loosely structured text and speech (patient diaries, symptom logs, clinician voice notes). This project builds an AI-powered assistant that:
1) helps patients log and understand their health over time, and  
2) helps clinicians turn raw dictation into clean, structured clinical documentation.

The project intentionally combines both sides into one flow so that a patient’s tracked timeline can be shared as a pre-visit summary, and a clinician can instantly convert dictation to a high-quality SOAP note.

## 2) Non-Goals (Safety & Scope)

- **Not a diagnostic tool.** No disease prediction, no “you have X” statements.
- **Not medical advice.** We only provide gentle, non-prescriptive suggestions (e.g., “consider rest,” “consider talking to a clinician”).
- **No real PHI/PII.** We use synthetic or anonymized content for all demos and testing.

## 3) Users & Core User Stories

### Patient (Health Diary)
- As a patient, I can add diary entries via text (and optionally via recorded voice).
- As a patient, I can see a timeline and trend visualizations of symptoms/mood over time.
- As a patient, I can generate a weekly summary of “what changed” and “what patterns repeat.”
- As a patient, I can produce a shareable pre-visit summary for my clinician.

### Clinician (Clinical Note Cleaner)
- As a clinician, I can paste or dictate a raw note and get a cleaned transcript.
- As a clinician, I can generate a **structured SOAP note** (Subjective, Objective, Assessment, Plan).
- As a clinician, I can edit the SOAP note, then export/copy it into a medical record system.

## 4) Product Overview (Two Tabs, One Continuum)

**UI structure**
- **Patient tab:** diary entry → trends → weekly summary → “Share with clinician”
- **Clinician tab:** transcript/dictation → cleaned text → SOAP → export

**Continuum**
- The “Share with clinician” output is designed to be readable by clinicians and can serve as the Subjective context for documentation.

## 5) Functional Requirements

### 5.1 Patient Diary
**Inputs**
- Text entry (required)
- Voice entry (optional for demo; can be pre-recorded upload)

**Processing**
- (Optional) Speech-to-text
- Entity extraction (symptoms, medications, durations, severities) — if available
- Sentiment/mood tracking (simple scale or inferred label)
- Temporal trend computation over a selected time range

**Outputs**
- Timeline list of entries
- Trend chart(s)
  - symptom frequency / severity over time
  - mood/sentiment over time
- Weekly summary (bullets)
- Gentle suggestions (non-diagnostic)
- Shareable pre-visit summary card (copy/export)

### 5.2 Clinical Note Cleaner (Voice/Text → SOAP)
**Inputs**
- Transcript text (required for MVP)
- Voice recording (optional)

**Processing**
- (Optional) Speech-to-text
- Cleaning & formatting (remove filler, normalize punctuation)
- SOAP structuring into JSON schema
- Guardrails:
  - “unknown/not provided” for missing objective data
  - avoid inventing vitals/labs/med history that isn’t present
  - no diagnosis claims

**Outputs**
- Cleaned transcript
- SOAP note view (editable)
- Export/copy as text and JSON

## 6) Data Model & Schemas (Contract-First)

All AI outputs must validate to schemas. If a model/provider fails, we fall back to a stable “mock” output so the demo remains robust.

### 6.1 DiaryEntry
```json
{
  "id": "uuid",
  "timestamp": "2026-02-24T18:30:00Z",
  "source": "text|voice",
  "text": "Free-form diary text",
  "mood": "low|neutral|good|unknown",
  "symptoms": [
    { "name": "headache", "severity": 3, "duration_hours": 4 }
  ],
  "medications": [
    { "name": "ibuprofen", "dose": "200mg", "frequency": "once" }
  ],
  "tags": ["sleep", "stress"]
}
