"""Central configuration, read from environment variables.

Everything that varies between "runs on my laptop with no keys" and
"runs in CI / production with Claude" is a setting here. Tests set these
to the lightweight 'fake'/'memory' implementations so the suite needs no
heavy ML dependencies and no API keys.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- providers (swap implementations without touching code) ---
    embedding_provider: str = "bge"         # "bge" | "minilm" | "fake"
    vector_store: str = "chroma"            # "chroma"  | "memory"
    llm_provider: str = "claude"            # "claude" | "openai" | "router" | "fake"

    # --- models ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # used when provider=minilm
    claude_model: str = "claude-haiku-4-5-20251001"
    anthropic_api_key: str = ""             # RAG_ANTHROPIC_API_KEY or set in .env
    # Judge for the eval harness. Default reuses the cheap model so a run never
    # fails on an unknown model string; set RAG_EVAL_JUDGE_MODEL to a frontier
    # model (recommended) for higher-quality correctness/groundedness scores.
    eval_judge_model: str = "claude-haiku-4-5-20251001"

    # --- router (Phase 4 cost control) ---
    # When llm_provider="router": cheap tier handles easy queries, frontier the
    # hard ones. Default is all-Claude (one key); the cost delta between Haiku
    # and Sonnet is real, so the saving is measurable.
    router_cheap_model: str = "claude-haiku-4-5-20251001"
    router_frontier_model: str = "claude-sonnet-4-6"

    # OpenAI is optional — only used if you set llm_provider/router to the OpenAI
    # cheap tier and provide a key.
    openai_api_key: str = Field(            # reads OPENAI_API_KEY or RAG_OPENAI_API_KEY
        default="",
        validation_alias=AliasChoices("RAG_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str = "gpt-4o-mini"

    # --- guardrails (Phase 4) ---
    enforce_groundedness: bool = True       # refuse if the answer isn't supported by contexts
    groundedness_min_overlap: float = 0.4
    rate_limit: str = "30/minute"           # per-client request cap (slowapi)

    # --- chunking ---
    chunk_tokens: int = 350                 # BGE handles 512; keeps sections intact
    chunk_overlap: int = 60

    # --- retrieval / generation ---
    top_k: int = 8
    score_threshold: float = 0.25           # below this -> refuse to answer
    max_answer_tokens: int = 500

    # --- paths ---
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "clinical_guidelines"
    sources_file: str = "./data/sources.yaml"
    raw_dir: str = "./data/raw"


@lru_cache
def get_settings() -> Settings:
    return Settings()
