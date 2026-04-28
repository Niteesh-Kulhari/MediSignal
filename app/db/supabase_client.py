import asyncio
from supabase import create_client, Client
from app.config import settings
from typing import Any
import logging

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
    return _client



# client1 = get_client()
# client2 = get_client()

# print(id(client1))
# print(id(client2))

async def upsert_session(session_data: dict[str, any]) -> dict | None:
    """Insert or update a session record."""

    try:
        client = get_client()
        result = await asyncio.to_thread(
            lambda: client.table("session")
            .upsert(session_data, on_conflict="session_id")
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"upsert_session failed: {e}")
        return None



async def get_session(session_id: str) -> dict | None:
    """Retrieve a session from DB"""

    try:
        client = get_client()
        result = await asyncio.to_thread(
            lambda: client.table("session")
            .select("*")
            .eq("session_id", session_id)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"get_session failed: {e}")
        return None
    

async def upsert_clinical_record(record: dict[str, Any]) -> dict | None:
    """
    Insert a MIMIC record with its embedding into clinical_recoreds table.
    record must have: content, embedding, source, metadata(optional)
    """

    try:
        client = get_client();
        result = await asyncio.to_thread(
            lambda: client.table("clinical_records")
            .upsert(record)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"upsert_clinical_record failed: {e}")
        return None
    
async def similarity_search(
        query_embeddings: list[float],
        match_count: int = 10,
        min_score: float=0.65
) -> list[dict]:
    """Call the pgvector similarity search RPC function.
    Return records sorted by cosine similarity descending
    """

    try:
        client = get_client()
        result = await asyncio.to_thread(
            lambda: client.rpc(
                "match_clinical_records",
                {
                    "query_embeddings": query_embeddings,
                    "match_count": match_count,
                    "min_score": min_score,

                }
            ).execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"similarity_search failed: {e}")
        return []