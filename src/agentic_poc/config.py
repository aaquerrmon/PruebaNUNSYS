"""Configuración centralizada validada con pydantic-settings.

Toda la configuración llega por variables de entorno (o `.env`). No hay
credenciales reales: los endpoints de inferencia son locales (vLLM) y la
clave que exige el cliente OpenAI es una clave ficticia interna.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LOCAL_INFERENCE_HOSTS = frozenset(
    {
        "llm",
        "embeddings",
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "host.docker.internal",
    }
)


class SettingsError(RuntimeError):
    """Configuración inválida detectada al arrancar (mensaje claro, sin secretos)."""


class Settings(BaseSettings):
    """Variables de entorno de la POC (ver `.env.example`)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Entorno y logging
    app_env: str = "development"
    log_level: str = "INFO"
    log_format: str = "text"
    privacy_log_questions: bool = False

    # LLM local
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "local-llm"
    llm_api_key: str = "local-vllm-key"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1
    llm_enable_thinking: bool = False
    allow_remote_inference: bool = False

    # Embeddings locales
    embeddings_base_url: str = "http://127.0.0.1:8001/v1"
    embeddings_model: str = "local-embeddings"
    embeddings_api_key: str = "local-vllm-key"
    embeddings_dimensions: int = 384
    embeddings_timeout_seconds: float = 30.0
    embeddings_query_prefix: str = "query:"
    embeddings_passage_prefix: str = "passage:"
    embeddings_batch_size: int = 8

    # Modelos (solo para descarga inicial; no configuran inferencia remota)
    llm_hf_repo: str = "Qwen/Qwen3-4B-AWQ"
    embeddings_hf_repo: str = "intfloat/multilingual-e5-small"
    vllm_image: str = "vllm/vllm-openai:v0.31.0"

    # ERP
    erp_database_url: str = "sqlite:///data/erp/northwind.db"
    erp_max_rows: int = 500
    erp_query_timeout_seconds: float = 10.0

    # Producción simulada
    production_api_url: str = "http://127.0.0.1:8090"
    production_timeout_seconds: float = 10.0
    production_max_retries: int = 1

    # RAG
    vector_store_path: Path = Path("data/vector_store")
    documents_dir: Path = Path("data/documents")
    max_pdf_bytes: int = 10 * 1024 * 1024
    max_pdf_pages: int = 200
    rag_top_k: int = 5
    rag_score_threshold: float = 0.35
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120

    # Memoria conversacional y persistencia
    app_db_path: Path = Path("data/app.db")
    memory_max_turns: int = 5
    memory_retention_messages: int = 50

    # API
    # NoDecode: llega en crudo desde el entorno (CSV o JSON) y lo separa el
    # validador. Sin esto, pydantic-settings intenta json.loads() antes de los
    # validadores y un "a,b" del entorno revienta el arranque.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://127.0.0.1:8585"]
    )
    api_host: str = "127.0.0.1"
    api_port: int = 8080

    # Servicios locales (solo bind en 127.0.0.1 por defecto)
    production_bind_host: str = "127.0.0.1"
    production_port: int = 8090
    chainlit_host: str = "127.0.0.1"
    chainlit_port: int = 8585

    # UI
    backend_url: str = "http://127.0.0.1:8080"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                except ValueError:
                    parsed = None
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            return [item.strip() for item in text.split(",") if item.strip()]
        return value

    @field_validator(
        "llm_base_url",
        "embeddings_base_url",
        "production_api_url",
        "backend_url",
    )
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"prod", "production"}

    def local_inference_hosts(self) -> set[str]:
        hosts = set(LOCAL_INFERENCE_HOSTS)
        for url in (self.llm_base_url, self.embeddings_base_url):
            host = urlparse(url).hostname
            if host:
                hosts.add(host)
        return hosts

    def assert_local_inference(self) -> None:
        """Fallar al arrancar si alguna URL de inferencia no es local.

        Garantiza que no exista ningún camino hacia un proveedor remoto: si la
        configuración apunta fuera de la red local, la aplicación no arranca.
        """
        if self.allow_remote_inference:
            return
        for name, url in (
            ("LLM_BASE_URL", self.llm_base_url),
            ("EMBEDDINGS_BASE_URL", self.embeddings_base_url),
        ):
            host = urlparse(url).hostname
            if host is None or host not in LOCAL_INFERENCE_HOSTS:
                raise SettingsError(
                    f"{name} apunta a '{host or url}' y no es un destino local. "
                    "Esta POC solo admite inferencia local (vLLM). "
                    "Configura las URLs locales o, si realmente necesitas un "
                    "entorno controlado, define ALLOW_REMOTE_INFERENCE=true."
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instancia única de configuración (se invalida en tests con `cache_clear`)."""
    return Settings()
