from dataclasses import dataclass, field
from typing import Literal
from app.models.clinical import PatientSession
from app.agents.intake import IntakeAgent
from app.agents.docqa import DocQAAgent
from app.agents.triage import TriageAgent
from app.agents.synthesis import SynthesisAgent
from app.models.agnets_io import AgentInput
from app.db import supabase_client
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorState:
    """Mutable state threaded through the pipeline"""
    session: PatientSession
    retry_count: int = 0
    max_retries: int = 2
    errors: list[str] = field(default_factory=list)

class MediSignalOrchestrator:
    """
    Routes inputs through the 4-agent pipeline.
    Handles low-confidence outputs with retry logic.
    """

    def __init__(self):
        self.intake_agent = IntakeAgent()
        self.docqa_agent = DocQAAgent()
        self.triage_agent = TriageAgent()
        self.synthesis_agent = SynthesisAgent()

    async def run(
            self,
            modality: str,
            raw_payload: str | bytes,
            metadata: dict = None,
    ) -> PatientSession:
        session_id = str(uuid.uuid4())
        session = PatientSession(session_id=session_id)
        state = OrchestratorState(session=session)

        logger.infor(f"[Orchestrator] Starting session {session_id}")

        # ---- Step 1: Intake --------
        state.session.status = "ingesting"
        intake_input = AgentInput(
            session_id= session_id,
            modality= modality,
            raw_payload=raw_payload,
            metadata=metadata or {},
        )

        intake_out = await self.intake_agent.run(intake_input)
        state.session.agent_outputs.append(intake_out.model_dump(mode="json"))

        if intake_out.error and not intake_out.result.get("transcript"):
            state.session.status = "failed"
            return state.session
        
        state.session.transcript = intake_out.result.get("transcript", "")
        state.session.modalities_received.append(modality)

        # -- Step 2: DocQA (Only PDF modality) ------
        if modality == "pdf":
            docqa_input = AgentInput(
                session_id=session_id,
                modality="pdf",
                raw_payload=raw_payload,
            )

            docqa_out = await self.docqa_agent.run(docqa_input)
            state.session.agent_outputs.append(docqa_out.model_dump(mode="json"))

        # Step 3: Triage -------
        state.session.status = "triaging"
        triage_input = AgentInput(
            session_id=session_id,
            modality="text",
            raw_payload=state.session.transcript or "",
        )

        triage_out = await self.triage_agent.run(triage_input)
        state.session.agent_outputs.append(triage_out.model_dump(mode="json"))

        # Low confidence -> escalate to human review
        if triage_out.low_confidence:
            logger.warning(
                f"[Orchestrator] Triage low confidence on {session_id}"
                f"- routing to need_review"
            )
            state.session.status = "needs_review"
            #still continues to synthesis so we have draft note

        # step 4: Synthesis ------
        state.session.status = "synthesizing"
        synthesis_input = AgentInput(
            session_id=session_id,
            modality="structured",
            raw_payload = state.session.transcript or "",
            metadata={
                "lab_values": [lv.model_dump() for lv in state.session.lab_values],
                "triage": triage_out.result
            },
        )

        synthesis_out = await self.synthesis_agent.run(synthesis_input)
        state.session.agent_outputs.append(synthesis_out.model_dump(mode="json"))


        #Finalize
        if state.session != "needs_review":
            state.session.status = "complete"

         # Persist to Supabase
        await supabase_client.upsert_session({
            "session_id": session_id,
            "status": state.session.status,
            "patient_session_json": state.session.model_dump(mode="json"),
        })

        logger.info(
            f"[Orchestrator] Session {session_id} finished "
            f"with status: {state.session.status}"
        )
        return state.session


