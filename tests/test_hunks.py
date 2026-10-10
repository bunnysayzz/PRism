"""Tests for unified-diff hunk parsing."""

from prism.diff.hunks import (
    commentable_lines,
    extract_hunk_context,
    parse_hunks,
)

MULTI_HUNK_PATCH = """\
@@ -1,5 +1,6 @@ def foo():
     x = 1
     y = 2
+    z = 3
     return x + y
-    old_line = True
@@ -20,3 +21,5 @@ def bar():
     a = 1
+    b = 2
+    c = 3
     return a
"""


def test_parse_hunks_count():
    hunks = parse_hunks(MULTI_HUNK_PATCH)
    assert len(hunks) == 2
    assert (hunks[0].new_start, hunks[1].new_start) == (1, 21)


def test_parse_hunks_empty():
    assert parse_hunks(None) == []
    assert parse_hunks("") == []


def test_commentable_lines_multi_hunk():
    # Hunk 1 (new_start=1): context,context,added,context  -> new lines 1,2,3,4
    #   (deleted line has no new-file number)
    # Hunk 2 (new_start=21): context,added,added,context  -> new lines 21,22,23,24
    assert commentable_lines(MULTI_HUNK_PATCH) == {1, 2, 3, 4, 21, 22, 23, 24}


def test_commentable_lines_deleted_excluded():
    patch = "@@ -5,4 +5,3 @@\n line_a\n-removed1\n-removed2\n line_b\n"
    # new lines: 5 (line_a), 6 (line_b); removed lines have no new-file numbers
    assert commentable_lines(patch) == {5, 6}


def test_commentable_lines_no_newline_marker_ignored():
    patch = "@@ -1,2 +1,2 @@\n line1\n+line2\n\\ No newline at end of file\n"
    assert commentable_lines(patch) == {1, 2}


def test_extract_hunk_context_hits_second_hunk():
    ctx = extract_hunk_context(MULTI_HUNK_PATCH, 22, radius=1)
    assert "b = 2" in ctx
    assert "z = 3" not in ctx  # other hunk, not included


def test_extract_hunk_context_miss():
    assert extract_hunk_context(MULTI_HUNK_PATCH, 999) == ""
    assert extract_hunk_context(None, 1) == ""
