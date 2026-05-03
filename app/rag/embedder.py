import httpx
import numpy as np
from app.config import settings
import logging
import asyncio

logger = logging.getLogger(__name__)

HF_API_URL = (
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/"
    f"{settings.hf_embedding_model}"
)

HEADERS = {
    "Authorization": f"Bearer {settings.hf_api_key}",
    "Content-Type": "application/json"
}

async def emebd_text(text: str) -> list[float]:
    """
    Embed a single text string using HF inference API.
    """

    text = text[:1500].strip()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": text, "options": {"wait_for_model": True}},
        )

        response.raise_for_status()
        raw = response.json()

        print("RAW:", raw)
        return raw


async def main():
    text = "Hello, this is a test sentence for embeddings"
    embedding = await emebd_text(text)

    print("\nEmbedding length:", len(embedding))
    print("First 10 values:", embedding[:10])


if __name__ == "__main__":
    asyncio.run(main())