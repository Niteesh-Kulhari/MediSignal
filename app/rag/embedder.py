import asyncio
import httpx
import numpy as np
from app.config import settings
import logging

logger = logging.getLogger(__name__)

HF_API_URL = (
    f"https://router.huggingface.co/hf-inference/models/"
    f"{settings.hf_embedding_model}/pipeline/feature-extraction"
)
HEADERS = {
    "Authorization": f"Bearer {settings.hf_api_key}",
    "Content-Type": "application/json"
}


async def embed_text(text: str) -> list[float]:
    """
    Embed a single text string using HF inference API.
    Return a 768-dim float list (all-mpnet-base-v2 output).
    """

    # Truncate to avoid token limit errors
    text = text[:1500].strip()

    print("Input Text:")
    print(text)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": text, "options": {"wait_for_model": True}},
        )

        response.raise_for_status()
        raw = response.json()

        # print("\nRaw Response:")
        # print(raw)
    embedding = _parse_embedding(raw)
    return embedding
    
def _parse_embedding(raw) -> list[float]:
    """
    HF feature-extraction response vary by model.
    all-mpnet-base-v2 returns: List[List[float]] (token embeddings)
    We mean-pool to get sentence embedding.
    """

    if isinstance(raw[0], float):
        return raw
    elif isinstance(raw[0], list) and isinstance(raw[0][0], float):
        arr = np.array(raw)
        return arr.mean(axis=0).tolist()
    else:
        raise ValueError(f"Unexpected embeddding shape from HF API: {type(raw[0])}")

async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts. HF Inference API supports batch inputs.
    Falls back to sequential if batch fails (cold model).
    """

    texts = [t[:1500].strip() for t in texts]

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                HF_API_URL,
                headers=HEADERS,
                json={"inputs": texts, "options": {"wait_for_model": True}},
            )
            response.raise_for_status()
            raw = response.json()

        return [_parse_embedding(r) for r in raw]
    except Exception as e:
        logger.warning(f"Batch embedding failed ({e}), falling back to sequential")
        import asyncio
        return await asyncio.gather(*[embed_text(t) for t in texts])

# async def test_embed_text():
#     """
#     Test function for embed_text.
#     """

#     sample_text = "Hello world. This is a test embedding."

#     try:
#         embedding = await embed_text(sample_text)

#         print("\nEmbedding Type:", type(embedding))

#         if isinstance(embedding, list):
#             print("Embedding Length:", len(embedding))

#             # Print first 5 values
#             print("First 5 values:", embedding[:5])

#             # Convert to numpy array (optional)
#             embedding_np = np.array(embedding, dtype=np.float32)
#             print("Numpy Shape:", embedding_np.shape)

#     except Exception as e:
#         logger.exception("Embedding test failed")
#         print("Error:", str(e))


# if __name__ == "__main__":
#     asyncio.run(test_embed_text())