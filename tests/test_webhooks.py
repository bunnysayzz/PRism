"""Tests for webhook signature verification and event dispatch."""

import hashlib
import hmac
import json

from prism.github.webhooks import parse_command_event, parse_event, verify_signature

SECRET = "test-webhook-secret"
BODY = json.dumps({"action": "opened", "zen": "keep it simple"}).encode()


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_verify_signature_valid():
    assert verify_signature(SECRET, BODY, _sign(BODY)) is True


def test_verify_signature_tampered_body():
    tampered = BODY + b"evil"
    assert verify_signature(SECRET, tampered, _sign(BODY)) is False


def test_verify_signature_wrong_secret():
    assert verify_signature(SECRET, BODY, _sign(BODY, secret="other")) is False


def test_verify_signature_missing_header():
    assert verify_signature(SECRET, BODY, None) is False
    assert verify_signature(SECRET, BODY, "") is False


def test_verify_signature_malformed_header():
    assert verify_signature(SECRET, BODY, "md5=deadbeef") is False
    assert verify_signature(SECRET, BODY, "sha256=nothex!!") is False


def test_verify_signature_empty_secret():
    assert verify_signature("", BODY, _sign(BODY, secret="")) is False


def _headers(event: str = "pull_request") -> dict:
    return {"X-GitHub-Event": event, "X-Hub-Signature-256": "sha256=abc"}


def _pr_payload(action: str = "opened") -> dict:
    return {
        "action": action,
        "installation": {"id": 12345},
        "repository": {"name": "myrepo", "owner": {"login": "myorg"}},
        "pull_request": {"number": 42},
    }


def test_parse_event_opened():
    job = parse_event(_headers(), _pr_payload("opened"))
    assert job is not None
    assert (job.installation_id, job.owner, job.repo, job.pr_number) == (
        12345,
        "myorg",
        "myrepo",
        42,
    )


def test_parse_event_synchronize():
    job = parse_event(_headers(), _pr_payload("synchronize"))
    assert job is not None
    assert job.pr_number == 42


def test_parse_event_ignored_actions():
    for action in ("closed", "review_requested", "labeled", "edited"):
        assert parse_event(_headers(), _pr_payload(action)) is None


def test_parse_event_ignored_events():
    assert parse_event(_headers("push"), _pr_payload("opened")) is None
    assert parse_event(_headers("ping"), {"zen": "hi"}) is None


def test_parse_event_malformed_payload():
    assert parse_event(_headers(), {"action": "opened"}) is None
    assert parse_event({}, _pr_payload("opened")) is None


def _wip_payload(title: str, draft: bool = False) -> dict:
    payload = _pr_payload("opened")
    payload["pull_request"]["title"] = title
    payload["pull_request"]["draft"] = draft
    return payload


def test_parse_event_skips_wip_title():
    for title in ("WIP: add feature", "[WIP] fix thing", "wip: refactor"):
        assert parse_event(_headers(), _wip_payload(title)) is None


def test_parse_event_skips_do_not_merge_title():
    for title in ("DO NOT MERGE yet", "do not merge: needs review"):
        assert parse_event(_headers(), _wip_payload(title)) is None


def test_parse_event_skips_draft():
    assert parse_event(_headers(), _wip_payload("Add feature", draft=True)) is None


def test_parse_event_reviews_normal_opened_pr():
    job = parse_event(_headers(), _wip_payload("Add feature", draft=False))
    assert job is not None
    assert job.pr_number == 42


# ---------------------------------------------------------------------------
# @prism fix command parsing
# ---------------------------------------------------------------------------


def _comment_payload(
    body: str = "nice, @prism fix please",
    action: str = "created",
    is_pr: bool = True,
) -> dict:
    payload = {
        "action": action,
        "installation": {"id": 999},
        "repository": {"name": "myrepo", "owner": {"login": "myorg"}},
        "comment": {
            "id": 777,
            "body": body,
            "user": {"login": "azhar"},
        },
        "issue": {"number": 42},
    }
    if is_pr:
        payload["issue"]["pull_request"] = {"url": "https://api.github.com/x/pulls/42"}
    return payload


def _comment_headers(event: str = "issue_comment") -> dict:
    return {"X-GitHub-Event": event, "X-Hub-Signature-256": "sha256=abc"}


def test_parse_command_event_match():
    cmd = parse_command_event(_comment_headers(), _comment_payload())
    assert cmd is not None
    assert (cmd.installation_id, cmd.owner, cmd.repo, cmd.pr_number) == (
        999,
        "myorg",
        "myrepo",
        42,
    )
    assert cmd.comment_id == 777
    assert cmd.requester == "azhar"


def test_parse_command_event_mention_case_insensitive():
    cmd = parse_command_event(_comment_headers(), _comment_payload(body="@PRISM FIX it"))
    assert cmd is not None
    assert cmd.requester == "azhar"


def test_parse_command_event_header_name_case_insensitive():
    headers = {"x-github-event": "issue_comment", "x-hub-signature-256": "sha256=abc"}
    assert parse_command_event(headers, _comment_payload()) is not None


def test_parse_command_event_wrong_event():
    assert parse_command_event(_comment_headers("pull_request"), _comment_payload()) is None
    assert parse_command_event(_comment_headers("ping"), _comment_payload()) is None


def test_parse_command_event_wrong_action():
    assert parse_command_event(_comment_headers(), _comment_payload(action="edited")) is None
    assert parse_command_event(_comment_headers(), _comment_payload(action="deleted")) is None


def test_parse_command_event_no_mention():
    assert parse_command_event(_comment_headers(), _comment_payload(body="looks good")) is None
    assert parse_command_event(_comment_headers(), _comment_payload(body="")) is None


def test_parse_command_event_issue_not_pr():
    assert parse_command_event(_comment_headers(), _comment_payload(is_pr=False)) is None


def test_parse_command_event_malformed_payload():
    assert parse_command_event(_comment_headers(), {"action": "created"}) is None
    broken = _comment_payload()
    del broken["installation"]
    assert parse_command_event(_comment_headers(), broken) is None
    broken = _comment_payload()
    del broken["comment"]["user"]
    assert parse_command_event(_comment_headers(), broken) is None


def test_parse_command_event_never_raises_on_garbage():
    assert parse_command_event({}, []) is None
    assert parse_command_event(_comment_headers(), []) is None
    assert parse_command_event("nope", _comment_payload()) is None  # type: ignore[arg-type]
    assert parse_command_event(_comment_headers(), None) is None  # type: ignore[arg-type]
