from pydantic import BaseModel
from datetime import datetime
from typing import Any

class SessionRecord(BaseModel):
    """Map 1:1 to the 'sessions' table in Supabase"""
    session_id: str
    status: str
    patient_session_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_patient_session(cls, ps) -> "SessionRecord":
        now = datetime.utcnow()
