from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    #API_KEYS
    anthropic_api_key: str
    hf_api_key: str
    supabase_url: str
    supabase_key: str
    supabase_db_url: str
    redis_url: str="redis://localhost:6379/0"

    #------HUGGINGFACE Model Names--------
    hf_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    hf_asr_model: str = "openai/whisper-large-v3"
    hf_docqa_model: str = "impira/layoutlm-document-qa"
    hf_zeroshot_model: str = "facebook/bart-large-mnli"

    # --- Embedding ---
    embedding_dim: int = 768
    embedding_batch_size: int = 20


    # --- RAG ---
    rag_top_k: int = 10          # retrieve this many before re-ranking
    rag_rerank_top_n: int = 3    # keep this many after re-ranking
    rag_min_score: float = 0.65  # discard results below this similarity

    # --- Agent Confidence Thresholds ---
    # Below these → low_confidence=True → orchestrator retries or escalates
    intake_confidence_threshold: float = 0.80
    docqa_confidence_threshold: float = 0.75
    triage_confidence_threshold: float = 0.80
    synthesis_confidence_threshold: float = 0.75

# --- Triage Labels (what Zero-Shot classifies into) ---
    triage_labels: list[str] = [
        "Emergency",   # life-threatening, immediate intervention
        "Urgent",      # needs same-day attention
        "Routine",     # can wait for scheduled appointment
        "Needs Review" # low confidence, route to human
    ]

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settins() -> Settings:
    return Settings()

settings = get_settins()