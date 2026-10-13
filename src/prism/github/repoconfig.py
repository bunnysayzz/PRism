"""Per-repo config from ``.prism.yml`` in the repo root.

Fetched at review time with the installation token (so private repos work),
then merged over the server config by ``prism.config.merge_repo_config``.
Never raises, and never logs the file contents.
"""

from __future__ import annotations

import base64
import binascii
from typing import TYPE_CHECKING, Any

import yaml

from prism.logging import get_logger

if TYPE_CHECKING:
    from prism.github.client import GitHubClient

log = get_logger(__name__)

_CONFIG_PATH = ".prism.yml"


async def fetch_repo_file_config(client: GitHubClient, owner: str, repo: str) -> dict[str, Any]:
    """Fetch and parse ``.prism.yml`` from the repo root.

    GET /repos/{owner}/{repo}/contents/.prism.yml, base64-decode ``content``,
    ``yaml.safe_load`` it, and return the mapping.

    Returns {} on 404 (no file is normal), on any transport/API error, on
    bad base64 or bad YAML, and when the YAML is not a mapping.
    Never raises. Never logs the file contents.
    """
    path = f"/repos/{owner}/{repo}/contents/{_CONFIG_PATH}"
    try:
        payload = await client.get(path)
    except Exception as exc:  # 404 included: a missing .prism.yml is normal
        log.info(
            "repo_config_fetch_failed",
            repo=f"{owner}/{repo}",
            reason=type(exc).__name__,
        )
        return {}
    try:
        raw = payload.get("content", "") if isinstance(payload, dict) else ""
        text = base64.b64decode("".join(str(raw).split())).decode("utf-8")
        data = yaml.safe_load(text)
    except (binascii.Error, ValueError, UnicodeDecodeError, yaml.YAMLError) as exc:
        log.info(
            "repo_config_parse_failed",
            repo=f"{owner}/{repo}",
            reason=type(exc).__name__,
        )
        return {}
    if not isinstance(data, dict):
        log.info("repo_config_not_a_mapping", repo=f"{owner}/{repo}")
        return {}
    return data
