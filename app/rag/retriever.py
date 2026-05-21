from app.rag.embedder import embed_text
from app.db.supabase_client import similarity_search
from app.config import settings
from app.models.agnets_io import AgentInput
import logging

logger = logging.getLogger(__name__)



async def retrieve_similar_records(query: str, top_k:int=None, min_score:float=None) -> list[dict]:
    """
    Main retrieval function used by SynthesisAgent.
    """

    top_k = top_k or settings.rag_top_k
    min_score = min_score or settings.rag_min_score

    # 1. Embed the query
    query_embedding = await embed_text(query)

    # 2. Vector Search
    raw_results = await similarity_search(
        query_embeddings=query_embedding,
        match_count=top_k,
        min_score=min_score
    )

    if not raw_results:
        logger.warning(f"No result above min_score={min_score} for query: {query[:80]}")
        return []
