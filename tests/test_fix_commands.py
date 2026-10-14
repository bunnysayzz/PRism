"""Tests for the `@prism fix` command handler."""

import base64

import pytest

from prism.github.client import GitHubAPIError
from prism.github.commands import (
    extract_applicable_suggestions,
    extract_suggestion,
    handle_fix_command,
)
from prism.github.webhooks import FixCommand

BOT_BODY = """<!-- PRism -->
**HIGH** · `correctness`

**Title**

explanation

Suggested fix:

```suggestion
result = compute(x)
```

_confidence: 0.95_
"""


def _comment(path="a.py", line=2, body=BOT_BODY):
    return {"path": path, "line": line, "body": body}


def _cmd(**overrides):
    base = {
        "installation_id": 999,
        "owner": "o",
        "repo": "r",
        "pr_number": 7,
        "comment_id": 777,
        "requester": "azhar",
    }
    base.update(overrides)
    return FixCommand(**base)


class FakeClient:
    """Records mutating calls; optionally fails reads/writes with a status.

    fail_status applies to get/paginate/request and to /git/* posts; the
    /issues comments post always succeeds so the degraded path is observable.
    """

    def __init__(self, files=None, comments=None, fail_status=None):
        self.files = files or {}
        self.comments = comments or []
        self.fail_status = fail_status
        self.posts = []  # list of (path, json)
        self.patches = []  # list of (path, json)

    def _maybe_fail(self, path):
        if self.fail_status and ("/git/" in path or not path.endswith("/comments")):
            raise GitHubAPIError(f"GET {path} -> {self.fail_status}", status=self.fail_status)

    async def get(self, path, **kwargs):
        self._maybe_fail(path)
        if path == "/repos/o/r/pulls/7":
            return {"head": {"sha": "abc123", "ref": "fix-branch"}}
        if path.startswith("/repos/o/r/contents/"):
            rel = path.split("/contents/")[1]
            content = base64.b64encode(self.files[rel].encode()).decode()
            return {"content": content, "encoding": "base64"}
        if path == "/repos/o/r/git/commits/abc123":
            return {"tree": {"sha": "treesha0"}}
        raise AssertionError(f"unexpected GET {path}")

    async def post(self, path, **kwargs):
        self._maybe_fail(path)
        self.posts.append((path, kwargs.get("json")))
        if path.endswith("/git/blobs"):
            return {"sha": "blobsha1"}
        if path.endswith("/git/trees"):
            return {"sha": "treesha1"}
        if path.endswith("/git/commits"):
            return {"sha": "commitsha1"}
        if path.endswith("/comments"):
            return {"id": 1}
        raise AssertionError(f"unexpected POST {path}")

    async def request(self, method, path, **kwargs):
        self._maybe_fail(path)
        assert method == "PATCH"
        self.patches.append((path, kwargs.get("json")))
        return {"object": {"sha": "commitsha1"}}

    async def paginate(self, path, **kwargs):
        self._maybe_fail(path)
        return self.comments


# ---------------------------------------------------------------------------
# extract_suggestion
# ---------------------------------------------------------------------------


def test_extract_suggestion_present():
    assert extract_suggestion(BOT_BODY) == "result = compute(x)"


def test_extract_suggestion_absent():
    assert extract_suggestion("just prose, no fence") is None
    assert extract_suggestion("") is None


def test_extract_suggestion_first_fence_wins():
    body = "```suggestion\na = f(b)\n```\ntext\n```suggestion\nc = g(d)\n```"
    assert extract_suggestion(body) == "a = f(b)"


# ---------------------------------------------------------------------------
# extract_applicable_suggestions
# ---------------------------------------------------------------------------


def test_extract_filters_non_drop_in_and_missing_fields():
    prose = BOT_BODY.replace("result = compute(x)", "consider refactoring this")
    comments = [
        _comment(),  # valid
        _comment(body=prose),  # prose suggestion -> not drop-in
        _comment(body="no fence at all"),  # no suggestion
        _comment(path=None),  # missing path
        {"path": "b.py", "body": BOT_BODY},  # missing line
        _comment(path="a.py", line=2),  # duplicate (path, line)
    ]
    targets = extract_applicable_suggestions(comments)
    assert len(targets) == 1
    assert (targets[0].path, targets[0].line) == ("a.py", 2)
    assert targets[0].suggestion == "result = compute(x)"


def test_extract_caps_at_five():
    comments = [_comment(path=f"f{i}.py") for i in range(7)]
    assert len(extract_applicable_suggestions(comments)) == 5


# ---------------------------------------------------------------------------
# handle_fix_command
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_fix_command_ok_applies_and_pushes():
    files = {"a.py": "x = 1\nresult = bad(x)\ny = 2\n"}
    client = FakeClient(files=files, comments=[_comment()])
    result = await handle_fix_command(client, _cmd())

    assert result == {"status": "ok", "applied": ["a.py"]}

    blob_posts = [p for p in client.posts if p[0].endswith("/git/blobs")]
    assert len(blob_posts) == 1
    new_content = base64.b64decode(blob_posts[0][1]["content"]).decode()
    assert new_content == "x = 1\nresult = compute(x)\ny = 2\n"

    tree_posts = [p for p in client.posts if p[0].endswith("/git/trees")]
    assert tree_posts[0][1]["tree"][0]["path"] == "a.py"

    assert client.patches == [("/repos/o/r/git/refs/heads/fix-branch", {"sha": "commitsha1"})]

    summary = [p for p in client.posts if p[0].endswith("/issues/7/comments")]
    assert len(summary) == 1
    assert "`a.py`" in summary[0][1]["body"]


@pytest.mark.asyncio
async def test_handle_fix_command_noop_without_suggestions():
    client = FakeClient(files={}, comments=[_comment(body="no fence here")])
    result = await handle_fix_command(client, _cmd())
    assert result["status"] == "noop"
    assert client.posts == []
    assert client.patches == []


@pytest.mark.asyncio
async def test_handle_fix_command_degraded_on_403():
    client = FakeClient(fail_status=403)
    result = await handle_fix_command(client, _cmd())

    assert result == {"status": "degraded", "reason": "missing contents:write"}
    comments = [p for p in client.posts if p[0].endswith("/issues/7/comments")]
    assert len(comments) == 1
    body = comments[0][1]["body"]
    assert "@azhar" in body
    assert "Contents: Read & write" in body
    assert "@prism fix" in body


@pytest.mark.asyncio
async def test_handle_fix_command_never_raises():
    client = FakeClient(fail_status=500)
    result = await handle_fix_command(client, _cmd())
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_handle_fix_command_skips_bad_file_best_effort():
    files = {"a.py": "x = 1\nresult = bad(x)\n", "gone.py": "only one line\n"}
    comments = [_comment(path="a.py", line=2), _comment(path="gone.py", line=99)]
    client = FakeClient(files=files, comments=comments)
    result = await handle_fix_command(client, _cmd())
    assert result == {"status": "ok", "applied": ["a.py"]}
