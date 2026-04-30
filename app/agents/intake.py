from app.agents.base import AgentBase
from app.models.agnets_io import AgentInput, AgentOutput
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class IntakeAgent(AgentBase):
    """
    Day 1: Stub the echoes text input.
    Day 2: call the whisper ASR via HF intrface API for audio
    """

    name = "intake_agent"
    confidence_threshold = settings.intake_confidence_threshold


    async def _execute(self, input: AgentInput) -> AgentOutput:
        if input.modality == "text":
            # Text input: pass through directly, full confidence
            return AgentOutput(
                session_id=input.session_id,
                agent_name=self.name,
                result={"transcript": input.raw_payload},
                confidence = 0.99,
                low_confidence=False,
                latency_ms=0,
                model_used="passthrough",
            )
        elif input.modality == "audio":
            #TODO Day2: call Whisper ASR
            return AgentOutput(
                session_id=input.session_id,
                agent_name=self.name,
                result={"transcript": "[STUB] Audio transcript pending"},
                confidence=0.50, #Low triggers retry path
                low_confidence=True,
                latency_ms=0,
                model_used=settings.hf_asr_model,
                error="ASR not yet implemented - Day 2",
            )
        else:
            return AgentOutput.error_output(
                session_id=input.session_id,
                agent_name=self.name,
                error=f"Intake Agent does not handle modality: {input.modality}",
                
            )
 