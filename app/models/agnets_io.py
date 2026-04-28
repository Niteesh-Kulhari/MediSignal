from pydantic import BaseModel, Field
from typing import Literal, Any
from datetime import datetime
import uuid

class AgentInput(BaseModel):
    """Standard input passed to every agent."""
    session_id: str
    modality: Literal["audio", "pdf", "text", "structured"]
    # For text/structured: the content string
    # For audio/pdf: base64-encoded bytes or a file path
    raw_payload: str | bytes
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class AgentOutput(BaseModel):
    """Standard input passed to every agent."""
    session_id: str
    agent_name: str
    result: dict[str, Any]
    confidence: float
    low_confidence: bool
    latency_ms: int
    model_used: str
    error: str | None = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def error_output(
        cls,
        session_id: str,
        agent_name: str,
        error: str,
        latency_ms: int = 0
    ) -> "AgentOutput" :
        """Factory for clean error returns - agents should use this"""
        return cls(
            session_id=session_id,
            agent_name=agent_name,
            result={},
            confidence=0.0,
            low_confidence=True,
            latency_ms=latency_ms,
            model_used="none",
            error=error,
        )