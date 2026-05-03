from app.agents.base import AgentBase
from app.models.agnets_io import AgentInput, AgentOutput
from app.config import settings


class SynthesisAgent(AgentBase):
    """
    Day 1: Stub returning placeholder SOAP notes.
    Day 4: Full Rag-grounded SOAP generation via Anthropic API.
    """

    name = "synthesis_agent"
    confidence_threshold = settings.synthesis_confidence_threshold

    async def _execute(self, input: AgentInput) -> AgentOutput:
        #TODO Day 4: Rag retrieval + LLM synthesis
        return AgentOutput(
            session_id=input.session_id,
            agent_name=self.name,
            result={
                "soap_note": {
                    "subjective": "[STUB] Patient input pending",
                    "objective": "[STUB] Lab extraction pending",
                    "assessment": "[STUB] Clinical synthesis pending",
                    "plan": "[STUB] Plan generation pending",
                    "sources_cited": [],
                }
            },
            confidence=0.50,
            low_confidence=True,
            latency_ms=0,
            model_used="claude-3-5-sonnet-20241022",
            error="Synthesis not yet implemented — Day 4",
        )