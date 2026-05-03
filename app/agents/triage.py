from app.agents.base import AgentBase
from app.models.agnets_io import AgentInput, AgentOutput
from app.config import settings


class TriageAgent(AgentBase):
    """
    Day1: Stub returning 'Needs Review'
    Day3: Zero-Shot Classification via HF inference API.
    """

    name = "triage_agent"
    confidence_threshold = settings.triage_confidence_threshold

    async def _execute(self, input: AgentInput) -> AgentOutput:
        #TODO Day 3: call zero-shot classification Model
        return AgentOutput(
            session_id= input.session_id,
            agent_name=self.name,
            result={
                "urgency": "Needs Review",
                "confidence": 0.50,
                "supporting_signals": [],
                "icd10_hints": [],
            },
            confidence=0.50,
            low_confidence=True,
            latency_ms=0,
            model_used=settings.hf_zeroshot_model,
            error="Triage not yet implemented"
        )