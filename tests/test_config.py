"""Tests de la guardia anti-inferencia-remota y validaciones de configuración."""

from __future__ import annotations

import pytest

from agentic_poc.config import Settings, SettingsError, get_settings


def _settings(**overrides: str) -> Settings:
    base: dict[str, object] = {
        "ALLOW_REMOTE_INFERENCE": "false",
        "LLM_BASE_URL": "http://127.0.0.1:8000/v1",
        "EMBEDDINGS_BASE_URL": "http://127.0.0.1:8001/v1",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_acepta_destinos_locales() -> None:
    _settings().assert_local_inference()  # no lanza


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://203.0.113.10:8000/v1",
        "https://generativelanguage.googleapis.com/v1beta",
    ],
)
def test_rechaza_destinos_remotos(url: str) -> None:
    with pytest.raises(SettingsError):
        _settings(LLM_BASE_URL=url).assert_local_inference()


def test_permite_remoto_solo_con_flag_explicito() -> None:
    _settings(
        ALLOW_REMOTE_INFERENCE="true", LLM_BASE_URL="https://api.openai.com/v1"
    ).assert_local_inference()


def test_acepta_hosts_de_compose() -> None:
    _settings(
        LLM_BASE_URL="http://llm:8000/v1", EMBEDDINGS_BASE_URL="http://embeddings:8001/v1"
    ).assert_local_inference()


def test_strip_trailing_slash() -> None:
    cfg = _settings(LLM_BASE_URL="http://127.0.0.1:8000/v1/")
    assert cfg.llm_base_url == "http://127.0.0.1:8000/v1"


def test_cors_desde_csv() -> None:
    cfg = _settings(CORS_ORIGINS="http://a:1,http://b:2")
    assert cfg.cors_origins == ["http://a:1", "http://b:2"]


def test_cors_desde_variable_de_entorno_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regresión: desde el entorno el valor llega como string crudo.

    Sin `NoDecode`, pydantic-settings intenta json.loads() antes de los
    validadores y el arranque falla con CORS_ORIGINS="a,b" (fallo real del
    servicio `bootstrap` en Compose).
    """
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8585,http://127.0.0.1:8585")
    cfg = Settings()
    assert cfg.cors_origins == ["http://localhost:8585", "http://127.0.0.1:8585"]


def test_cors_desde_variable_de_entorno_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", '["http://a:1", "http://b:2"]')
    cfg = Settings()
    assert cfg.cors_origins == ["http://a:1", "http://b:2"]


def test_get_settings_cacheado(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
