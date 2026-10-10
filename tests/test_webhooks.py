"""Tests for webhook signature verification and event dispatch."""

import hashlib
import hmac
import json

from prism.github.webhooks import parse_event, verify_signature

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
