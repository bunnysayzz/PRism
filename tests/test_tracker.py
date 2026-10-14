"""Tests for issue-tracker ticket context and rules/team-standards config."""

import pytest

from prism.config import (
    RepoConfig,
    build_team_standards,
    load_repo_config,
    merge_repo_config,
)
from prism.github.tracker import (
    TICKET_RE,
    fetch_ticket_context,
    parse_ticket_refs,
)

# ---------------------------------------------------------------------------
# parse_ticket_refs
# ---------------------------------------------------------------------------


def test_parse_ticket_refs_jira_style():
    assert parse_ticket_refs("Fixes PROJ-123 in this change") == ["PROJ-123"]


def test_parse_ticket_refs_github_issue():
    assert parse_ticket_refs("closes #45") == ["#45"]


def test_parse_ticket_refs_dedupe_preserves_order():
    assert parse_ticket_refs("PROJ-123 and #45 then PROJ-123 again") == ["PROJ-123", "#45"]


def test_parse_ticket_refs_branch_and_body():
    assert parse_ticket_refs("feature/PROJ-456-do-login body mentions ABC-1") == [
        "PROJ-456",
        "ABC-1",
    ]


def test_parse_ticket_refs_ignores_lowercase():
    assert parse_ticket_refs("proj-123 is not a ticket") == []


def test_parse_ticket_refs_no_false_positives():
    assert parse_ticket_refs("nothing here, released in v1.2.3") == []
    assert parse_ticket_refs("") == []


def test_ticket_re_matches_spec():
    assert TICKET_RE.pattern == r"\b([A-Z]{2,10}-\d+)\b"
    assert TICKET_RE.search("AB-12").group(1) == "AB-12"
    assert TICKET_RE.search("ab-12") is None


# ---------------------------------------------------------------------------
# fetch_ticket_context
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, pr=None, exc=None):
        self.pr = pr
        self.exc = exc
        self.paths: list[str] = []

    async def get(self, path):
        self.paths.append(path)
        if self.exc is not None:
            raise self.exc
        return self.pr


def _pr(branch="feature/PROJ-123-login", body=""):
    return {"head": {"ref": branch}, "body": body}


def _linear_cfg(token="SECRET_TOKEN_ABC"):
    return RepoConfig({"tracker": {"provider": "linear", "api_token": token}})


@pytest.mark.asyncio
async def test_fetch_none_when_no_refs():
    client = _FakeClient(_pr(branch="main", body="no tickets here"))
    assert await fetch_ticket_context(client, "o", "r", 1, _linear_cfg()) is None


@pytest.mark.asyncio
async def test_fetch_none_when_no_token():
    client = _FakeClient(_pr())
    cfg = RepoConfig({"tracker": {"provider": "linear"}})
    assert await fetch_ticket_context(client, "o", "r", 1, cfg) is None
    assert await fetch_ticket_context(client, "o", "r", 1, RepoConfig({})) is None


@pytest.mark.asyncio
async def test_fetch_none_when_provider_unknown():
    client = _FakeClient(_pr())
    cfg = RepoConfig({"tracker": {"provider": "youtrack", "api_token": "tok"}})
    assert await fetch_ticket_context(client, "o", "r", 1, cfg) is None


@pytest.mark.asyncio
async def test_fetch_none_when_pr_fetch_fails():
    client = _FakeClient(exc=RuntimeError("github down"))
    assert await fetch_ticket_context(client, "o", "r", 1, _linear_cfg()) is None


@pytest.mark.asyncio
async def test_fetch_none_when_only_github_refs():
    client = _FakeClient(_pr(branch="fix/stuff", body="closes #45"))
    assert await fetch_ticket_context(client, "o", "r", 1, _linear_cfg()) is None


@pytest.mark.asyncio
async def test_fetch_linear_success(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_post(url, headers, payload):
        seen["url"] = url
        seen["headers"] = headers
        seen["payload"] = payload
        return {
            "data": {
                "issue": {
                    "identifier": "PROJ-123",
                    "title": "Fix login redirect",
                    "description": "Do the thing. " * 1000,  # > 1500 chars
                }
            }
        }

    monkeypatch.setattr("prism.github.tracker._post_json", fake_post)
    client = _FakeClient(_pr())
    out = await fetch_ticket_context(client, "o", "r", 7, _linear_cfg())

    assert client.paths == ["/repos/o/r/pulls/7"]
    assert seen["url"] == "https://api.linear.app/graphql"
    assert seen["headers"]["Authorization"] == "SECRET_TOKEN_ABC"
    assert seen["payload"]["variables"]["id"] == "PROJ-123"
    assert out is not None
    assert out.startswith("Ticket PROJ-123: Fix login redirect\n")
    assert "Acceptance criteria / description:" in out
    description = out.split("Acceptance criteria / description:\n", 1)[1]
    assert len(description) <= 1500  # truncated
    # Token is never echoed into the returned context string.
    assert "SECRET_TOKEN_ABC" not in out


@pytest.mark.asyncio
async def test_fetch_linear_issue_missing_returns_none(monkeypatch):
    async def fake_post(url, headers, payload):
        return {"data": {"issue": None}}

    monkeypatch.setattr("prism.github.tracker._post_json", fake_post)
    client = _FakeClient(_pr())
    assert await fetch_ticket_context(client, "o", "r", 1, _linear_cfg()) is None


@pytest.mark.asyncio
async def test_fetch_linear_api_error_returns_none(monkeypatch):
    async def fake_post(url, headers, payload):
        raise RuntimeError("linear down (and SECRET_TOKEN_ABC is not leaked)")

    monkeypatch.setattr("prism.github.tracker._post_json", fake_post)
    client = _FakeClient(_pr())
    # Never raises, and the token never escapes in exceptions or the result.
    assert await fetch_ticket_context(client, "o", "r", 1, _linear_cfg()) is None


@pytest.mark.asyncio
async def test_fetch_jira_success(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_get(url, headers):
        seen["url"] = url
        seen["headers"] = headers
        return {
            "key": "ABC-9",
            "fields": {
                "summary": "Add export button",
                "description": "Ship it this sprint.",
            },
        }

    monkeypatch.setattr("prism.github.tracker._get_json", fake_get)
    client = _FakeClient(_pr(branch="feat", body="implements ABC-9"))
    cfg = RepoConfig(
        {
            "tracker": {
                "provider": "jira",
                "api_token": "JIRA_TOKEN_XYZ",
                "base_url": "https://acme.atlassian.net",
            }
        }
    )
    out = await fetch_ticket_context(client, "o", "r", 3, cfg)

    assert seen["url"] == "https://acme.atlassian.net/rest/api/3/issue/ABC-9"
    assert seen["headers"]["Authorization"] == "Bearer JIRA_TOKEN_XYZ"
    assert out == (
        "Ticket ABC-9: Add export button\nAcceptance criteria / description:\nShip it this sprint."
    )
    assert "JIRA_TOKEN_XYZ" not in out


@pytest.mark.asyncio
async def test_fetch_jira_adf_description_extracted(monkeypatch):
    async def fake_get(url, headers):
        return {
            "key": "ABC-9",
            "fields": {
                "summary": "T",
                "description": {
                    "type": "doc",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "First line."}],
                        },
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "Second line."}],
                        },
                    ],
                },
            },
        }

    monkeypatch.setattr("prism.github.tracker._get_json", fake_get)
    client = _FakeClient(_pr(body="ABC-9"))
    cfg = RepoConfig(
        {
            "tracker": {
                "provider": "jira",
                "api_token": "tok",
                "base_url": "https://acme.atlassian.net",
            }
        }
    )
    out = await fetch_ticket_context(client, "o", "r", 3, cfg)
    assert out is not None
    assert "First line.\nSecond line." in out


@pytest.mark.asyncio
async def test_fetch_jira_missing_base_url_returns_none(monkeypatch):
    async def fake_get(url, headers):  # pragma: no cover - must not be called
        raise AssertionError("should not be called")

    monkeypatch.setattr("prism.github.tracker._get_json", fake_get)
    client = _FakeClient(_pr())
    cfg = RepoConfig({"tracker": {"provider": "jira", "api_token": "tok"}})
    assert await fetch_ticket_context(client, "o", "r", 1, cfg) is None


# ---------------------------------------------------------------------------
# rules + tracker config schema
# ---------------------------------------------------------------------------


def test_load_repo_config_rules(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "rules:\n  - always use transactions\n  - 42\n  - '   '\n  - ''\n",
        encoding="utf-8",
    )
    cfg = load_repo_config(cfg_file)
    assert cfg["rules"] == ["always use transactions", "42"]


def test_load_repo_config_rules_default(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("severity_threshold: high\n", encoding="utf-8")
    assert load_repo_config(cfg_file)["rules"] == []


def test_load_repo_config_tracker_passthrough(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("tracker:\n  provider: linear\n  api_token: tok123\n", encoding="utf-8")
    cfg = load_repo_config(cfg_file)
    assert cfg["tracker"] == {"provider": "linear", "api_token": "tok123"}


def test_load_repo_config_tracker_non_dict_coerced(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("tracker: nope\n", encoding="utf-8")
    assert load_repo_config(cfg_file)["tracker"] == {}


def test_load_repo_config_tracker_default(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("severity_threshold: high\n", encoding="utf-8")
    assert load_repo_config(cfg_file)["tracker"] == {}


def test_merge_rules_union_deduped_server_first():
    server = RepoConfig({"rules": ["server rule", "shared"]})
    merged = merge_repo_config(server, {"rules": ["shared", "repo rule"]})
    assert merged["rules"] == ["server rule", "shared", "repo rule"]


def test_merge_rules_repo_absent_keeps_server():
    server = RepoConfig({"rules": ["server rule"]})
    assert merge_repo_config(server, {})["rules"] == ["server rule"]


def test_merge_rules_does_not_mutate_server():
    server = RepoConfig({"rules": ["server rule"]})
    merge_repo_config(server, {"rules": ["repo rule"]})
    assert server["rules"] == ["server rule"]


def test_merge_tracker_repo_wins_outright():
    server = RepoConfig({"tracker": {"provider": "jira", "api_token": "server-tok", "extra": 1}})
    merged = merge_repo_config(server, {"tracker": {"provider": "linear", "api_token": "repo-tok"}})
    # Never merged: the repo mapping replaces the server mapping entirely.
    assert merged["tracker"] == {"provider": "linear", "api_token": "repo-tok"}


def test_merge_tracker_absent_keeps_server():
    server = RepoConfig({"tracker": {"provider": "jira", "api_token": "server-tok"}})
    merged = merge_repo_config(server, {})
    assert merged["tracker"] == {"provider": "jira", "api_token": "server-tok"}


def test_merge_tracker_empty_dict_keeps_server():
    server = RepoConfig({"tracker": {"provider": "jira", "api_token": "server-tok"}})
    merged = merge_repo_config(server, {"tracker": {}})
    assert merged["tracker"] == {"provider": "jira", "api_token": "server-tok"}


# ---------------------------------------------------------------------------
# build_team_standards
# ---------------------------------------------------------------------------


def test_build_team_standards_empty():
    assert build_team_standards(RepoConfig({"instructions": "", "rules": []})) == ""


def test_build_team_standards_instructions_only():
    out = build_team_standards(RepoConfig({"instructions": "be kind", "rules": []}))
    assert out == "\n\n## Repository review instructions\nbe kind"
    assert "Team standards" not in out


def test_build_team_standards_rules_only():
    out = build_team_standards(RepoConfig({"instructions": "", "rules": ["a", "b"]}))
    assert out == ("\n\n## Team standards (follow these as binding review rules)\n- a\n- b")
    assert "Repository review instructions" not in out


def test_build_team_standards_both():
    out = build_team_standards(
        RepoConfig({"instructions": "be kind", "rules": ["use transactions"]})
    )
    assert out == (
        "\n\n## Repository review instructions\nbe kind"
        "\n\n## Team standards (follow these as binding review rules)\n- use transactions"
    )
