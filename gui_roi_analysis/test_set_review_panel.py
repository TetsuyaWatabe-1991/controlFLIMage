"""Tests for which frames the set-review panel draws (no GUI, no data files).

Run: python test_set_review_panel.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import set_review_panel as p  # noqa: E402


def test_evenly_spaced_keeps_ends_and_count() -> None:
    assert p.evenly_spaced(10, 3) == [0, 5, 9]
    four = p.evenly_spaced(5, 4)
    assert four[0] == 0 and four[-1] == 4 and len(four) == len(set(four)) == 4
    assert p.evenly_spaced(4, 10) == [0, 1, 2, 3]
    assert p.evenly_spaced(6, 1) == [0]
    assert p.evenly_spaced(0, 3) == []


def test_short_set_shows_every_frame() -> None:
    pre, unc, post = [0, 1, 2], [3, 10], [11, 12, 13]
    times = {11: 5.0, 12: 20.0, 13: 50.0}
    picks = p.select_review_picks(pre, unc, post, times, max_frames=18)
    assert [i for i, _ in picks] == [0, 1, 2, 3, 10, 11, 12, 13]
    assert [label for _, label in picks] == [
        "Pre 1", "Pre 2", "Pre 3", "Unc first", "Unc last", "Post 1", "Post 2", "Post 3",
    ]


def test_priority_window_is_kept_and_later_posts_fill_the_rest() -> None:
    pre, unc = [0, 1, 2], [3, 4]
    post = list(range(10, 30))  # 20 posts
    times = {i: 4.0 * (k + 1) for k, i in enumerate(post)}  # 4, 8, ... 80
    early = [i for i in post if times[i] <= 35]  # 4..32 -> 8 frames
    picks = p.select_review_picks(pre, unc, post, times, max_frames=18)
    shown = [i for i, _ in picks]
    assert shown[:5] == [0, 1, 2, 3, 4]
    assert early == shown[5:5 + len(early)]
    late_shown = shown[5 + len(early):]
    assert len(picks) == 18
    assert late_shown[0] == post[len(early)] or late_shown[0] > early[-1]
    assert late_shown[-1] == post[-1]
    assert len(late_shown) == 18 - 5 - len(early)


def test_early_posts_are_thinned_when_the_priority_window_does_not_fit() -> None:
    pre, unc = [0], [1, 2]
    post = list(range(10, 40))  # 30 posts, all at or before 35 min
    times = {i: 1.0 * k for k, i in enumerate(post)}
    picks = p.select_review_picks(pre, unc, post, times, max_frames=18)
    shown = [i for i, _ in picks]
    assert shown[:3] == [0, 1, 2]
    post_shown = shown[3:]
    assert len(picks) == 18
    assert post_shown[0] == post[0] and post_shown[-1] == post[-1]
    assert all(times[i] <= 35 for i in post_shown)
    # original post numbers, so a gap stays visible
    labels = [label for _, label in picks if label.startswith("Post")]
    assert labels[0] == "Post 1" and labels[-1] == "Post 30"
    assert len(labels) < 30


def test_one_uncaging_frame_is_not_repeated() -> None:
    picks = p.select_review_picks([0], [5], [6], {6: 10.0}, max_frames=18)
    assert [label for _, label in picks] == ["Pre 1", "Unc first", "Post 1"]


def test_missing_time_stays_in_the_priority_window() -> None:
    post = [10, 11, 12]
    picks = p.select_review_picks([], [], post, {12: 80.0}, max_frames=2)
    shown = [i for i, _ in picks]
    assert 12 not in shown
    assert shown == [10, 11]


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
