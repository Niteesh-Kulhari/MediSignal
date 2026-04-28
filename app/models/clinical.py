from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

class TriageResult(BaseModel):
    urgency: Literal["Emergency", "Urgent", "Routine", "Needs Review"]
    confidence: float
    supporting_signals: list[str] = Field(
        description="Phrases from patient input that drove this classification"
    )
    icd10_hints: list[str] = Field(
        default_factory=list,
        description="Suggested ICD-10 codes, grounded via UMLS"
    )


class ExtractedLabValue(BaseModel):
    name: str   # e.g. "Troponin I"
    value: str  # e.g. "2.4"
    unit: str   # e.g. "ng/mL"
    reference_range: str | None = None
    abnormal: bool = False
    source_page: int | None = None

class SOAPNote(BaseModel):
    subjective: str  # Patient-reported symptoms, history
    objective: str   # Extracted lab values, vitals, measurements
    assessment: str  # Synthesized clinical impression
    plan: str        # Recommended next steps
    sources_cited: list[str] = Field(
        description="MIMIC record IDs or document names used in synthesis"
    )
    urgency: TriageResult | None = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class PatientSession(BaseModel):
    session_id: str
    status: Literal[
        "created",
        "ingesting",
        "triaging",
        "retrieving",
        "synthesizing",
        "completed",
        "failed",
        "needs_review"
    ] = "created"
    modalities_received: list[str] = Field(default_factory=list)
    transcript: str | None=None # From IntakeAgent
    lab_values: list[ExtractedLabValue] = Field(default_factory=list)
    triage: TriageResult | None = None
    soap_note: SOAPNote | None = None
    agent_outputs: list[dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
