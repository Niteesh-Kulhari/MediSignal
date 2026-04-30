from app.agents.base import AgentBase
from app.models.agnets_io import AgentInput, AgentOutput
from app.config import settings

class DocQAAgent(AgentBase):
    """
    Day1: Stub returning empty lab values.
    Day2: LayoutLM DocQA on upload PDFs.
    """

    name = "docqa_agent"
    confidence_threshold = settings.docqa_confidence_threshold

    async def _execute(self, input: AgentInput) -> AgentOutput:
        #TODO Day2: parse PDF, run DocQA model, extract lab values
        return AgentOutput(
            session_id=input.session_id,
            agent_name=self.name,
            result={
                "lab_values": [],
                "raw_text": "[STUB] PDF parsing pending",
            },
            confidence=0.50,
            low_confidence=True,
            latency_ms=0,
            model_used=settings.hf_docqa_model,
            error="DocQA not yet implemented - Day 2"
        )