from abc import ABC, abstractmethod
from app.models.agnets_io import AgentInput, AgentOutput
from app.config import settings
import time
import logging

logger = logging.getLogger(__name__)

class AgentBase(ABC):
    """
    All agents inherit from this class
    Enforces: structured I/O, confidence thresholds, timing, error handling.
    """

    name: str="base"
    confidence_threshold: float=0.75

    async def run(self, input: AgentInput) -> AgentOutput:
        """
        Public entry point. wraps _execute() with timings + error handling.
        Agents implement _execute(), not run().
        """

        start = time.monotonic()

        try:
            output = await self._execute(input)
            output.latency_ms = int((time.monotonic() -start) * 1000)
            output.low_confidence = self.is_low_confidence(output.confidence)

            if(output.low_confidence):
                logger.warning(
                    f"[{self.name}] Low Confidence ({output.confidence:.2f})"
                    f"on session {input.session_id}"
                )
            return output
        except Exception as e:
            latency_ms = int((time.monotonic() -start) * 1000)
            logger.error(f"[{self.name}] Unhandeled error: {e}", exc_info=True)
            return AgentOutput.error_output(
                session_id=input.session_id,
                agent_name=self.name,
                error=str(e),
                latency_ms=latency_ms,
            )
        
    @abstractmethod
    async def _execute(self, input: AgentInput) -> AgentOutput:
        """Implement actual agent logic here."""
        ...

    def is_low_confidence(self, score: float) -> bool:
        return score < (self.confidence_threshold)