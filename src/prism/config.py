"""
Shared configuration: environment settings (pydantic-settings) + repo YAML config.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from prism.logging import REDACT_SUBSTRINGS, REDACTED


class Settings(BaseSettings):
    """Environment-driven settings. Secrets never appear in logs."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- GitHub App ---
    github_app_id: int = Field(default=0, description="GitHub App ID (numeric)")
    github_private_key_path: str = Field(default="", description="Path to App PEM private key file")
    github_webhook_secret: str = Field(default="", description="HMAC secret for webhooks")

    # --- Queue / Redis ---
    redis_url: str | None = Field(default=None, description="Redis URL; unset => dev inline mode")

    # --- Review backend selection ---
    review_backend: Literal["vllm", "hf_endpoint", "api", "stub"] = Field(default="vllm")

    # Primary model (vLLM, OpenAI-compatible)
    vllm_base_url: str = Field(default="http://localhost:8000")
    vllm_model: str = Field(default="code-review-llama3-lora")
    vllm_timeout_s: float = Field(default=120.0)

    # Hugging Face Inference Endpoint
    hf_endpoint_url: str = Field(default="")
    hf_token: str = Field(default="")
    hf_timeout_s: float = Field(default=120.0)

    # Generic OpenAI-compatible API backend (also used as the stronger fallback)
    fallback_api_base: str = Field(default="https://api.openai.com/v1")
    fallback_api_key: str = Field(default="")
    fallback_model: str = Field(default="gpt-4o")

    # --- Confidence gate ---
    # >= auto_post -> post directly. [verify, auto_post) -> second-pass verify.
    # < verify -> drop.
    confidence_auto_post: float = Field(default=0.80, ge=0.0, le=1.0)
    confidence_verify: float = Field(default=0.55, ge=0.0, le=1.0)

    # --- Limits ---
    max_files_per_pr: int = Field(default=30, gt=0)
    max_concurrent_files: int = Field(default=4, gt=0)
    max_file_bytes: int = Field(default=200_000, gt=0)
    github_timeout_s: float = Field(default=30.0)

    # --- Misc ---
    repo_config_path: str = Field(default="config.yaml")
    feedback_dir: str = Field(default="feedback_data")
    log_level: str = Field(default="INFO")
    log_format: Literal["json", "console"] = Field(
        default="json", description="LOG_FORMAT: json (prod) | console (local dev)"
    )

    # --- Rate limiting (webhook endpoint) ---
    rate_limit_webhook_per_min: int = Field(
        default=30, gt=0, description="Token-bucket capacity for POST /webhooks/github"
    )

    # --- Model serving / training ---
    prism_adapter_dir: str | None = Field(
        default=None,
        description="Path to the fine-tuned LoRA adapter dir (mounted into vLLM)",
    )
    mlflow_tracking_uri: str | None = Field(
        default=None, description="MLflow tracking server URI (training/eval runs)"
    )
    structured_output_max_retries: int = Field(
        default=2, ge=0, description="Retries when model output fails schema validation"
    )

    # --- Eval gate ---
    eval_min_f1: float = Field(
        default=0.70, ge=0.0, le=1.0, description="Eval gate: build fails below this F1"
    )

    @field_validator("confidence_verify")
    @classmethod
    def _verify_below_auto_post(cls, v: float, info: Any) -> float:
        auto = info.data.get("confidence_auto_post", 1.0)
        if v > auto:
            raise ValueError("confidence_verify must be <= confidence_auto_post")
        return v

    def private_key_pem(self) -> str:
        """Load the App private key from file. Never from env contents."""
        if not self.github_private_key_path:
            raise RuntimeError("GITHUB_PRIVATE_KEY_PATH is not set")
        return Path(self.github_private_key_path).read_text(encoding="utf-8")

    def redacted(self) -> dict[str, Any]:
        """Return all settings as a dict with secret values masked.

        Safe to log or print for startup diagnostics / debugging — values of
        any field whose name contains a secret-like substring (secret, token,
        key, password, ...) are replaced with ``***REDACTED***``.
        """
        data = self.model_dump()
        out: dict[str, Any] = {}
        for name, value in data.items():
            lowered = name.lower()
            if value not in (None, "") and any(sub in lowered for sub in REDACT_SUBSTRINGS):
                out[name] = REDACTED
            else:
                out[name] = value
        return out


def safe_dump(settings: Settings | None = None) -> dict[str, Any]:
    """Dump ``settings`` (or the cached global) with secrets masked.

    Use for startup logging, /debug endpoints, and support bundles.
    """
    return (settings if settings is not None else get_settings()).redacted()


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# Repo YAML config
# ---------------------------------------------------------------------------

DEFAULT_IGNORE_GLOBS = [
    "**/package-lock.json",
    "**/yarn.lock",
    "**/pnpm-lock.yaml",
    "**/poetry.lock",
    "**/Pipfile.lock",
    "**/*.min.js",
    "**/*.min.css",
    "**/*.map",
    "**/vendor/**",
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/__pycache__/**",
    "**/*.pb.go",  # protobuf generated
    "**/*.g.dart",
]


class RepoConfig(dict[str, Any]):
    """Thin dict wrapper around the YAML repo config with sane defaults."""


def load_repo_config(path: str | os.PathLike[str] | None = None) -> RepoConfig:
    """Load YAML repo config; missing file -> defaults."""
    cfg_path = Path(path) if path else Path(get_settings().repo_config_path)
    data: dict[str, Any] = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = RepoConfig()
    cfg["repos_allowlist"] = data.get("repos_allowlist", [])  # empty => all repos
    cfg["severity_threshold"] = data.get("severity_threshold", "low")  # low|medium|high|critical
    cfg["ignore_globs"] = list(data.get("ignore_globs", [])) + DEFAULT_IGNORE_GLOBS
    cfg["categories"] = data.get("categories", {})  # e.g. {"style": false}
    cfg["model_routing"] = data.get("model_routing", {})
    cfg["max_findings_per_file"] = int(data.get("max_findings_per_file", 10))
    return cfg


def is_repo_allowed(cfg: RepoConfig, owner: str, repo: str) -> bool:
    allow = cfg.get("repos_allowlist") or []
    if not allow:
        return True
    full = f"{owner}/{repo}".lower()
    return any(a.lower() == full for a in allow)


def is_category_enabled(cfg: RepoConfig, category: str) -> bool:
    cats = cfg.get("categories") or {}
    return bool(cats.get(category, True))
