# -*- coding: utf-8 -*-
"""Headless tests for the grab wait. No FLIMage, no hardware.

Usage:
    python test_grab_timeout.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controlflimage_threading import (  # noqa: E402
    poll_until_grab_stops,
    release_stuck_grab,
    resolve_grab_timeout_sec,
    wait_until_path_exists,
)


class Clock:
    """Test clock. sleep() advances it; nothing waits on the wall clock."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


def test_timeout_is_300s_even_for_a_short_expected_duration() -> None:
    assert resolve_grab_timeout_sec(10) == 300.0
    assert resolve_grab_timeout_sec(30) == 300.0
    assert resolve_grab_timeout_sec(0) == 300.0
    previous = os.environ.get("FLIMAGE_GRAB_TIMEOUT_SEC")
    os.environ["FLIMAGE_GRAB_TIMEOUT_SEC"] = "7"
    try:
        assert resolve_grab_timeout_sec(10) == 7.0
    finally:
        if previous is None:
            os.environ.pop("FLIMAGE_GRAB_TIMEOUT_SEC", None)
        else:
            os.environ["FLIMAGE_GRAB_TIMEOUT_SEC"] = previous
    print("PASS timeout is 300 s unless FLIMAGE_GRAB_TIMEOUT_SEC is set")


def test_still_grabbing_past_the_old_50s_loop_is_a_timeout() -> None:
    """interval_sec=50 used to end the wait and report success. It must not."""
    clock = Clock()
    polls = {"n": 0}
    timeouts: list[float] = []

    def is_grabbing() -> int:
        polls["n"] += 1
        return 1

    ok, reason = poll_until_grab_stops(
        is_grabbing,
        timeout_sec=90,
        sleep_every_sec=0.5,
        clock=clock,
        sleep_fn=clock.sleep,
        on_timeout=timeouts.append,
    )
    old_loop_polls = int(50 / 0.5)
    assert ok is False and reason == "timeout"
    assert polls["n"] > old_loop_polls
    assert timeouts == [90.5]
    print(
        "PASS a grab still running at 90 s is a timeout "
        f"({polls['n']} polls, past the old {old_loop_polls})"
    )


def test_grab_that_stops_is_success_without_timeout() -> None:
    clock = Clock()
    answers = iter([1, 1, 0])
    timeouts: list[float] = []
    ok, reason = poll_until_grab_stops(
        lambda: next(answers),
        timeout_sec=90,
        sleep_every_sec=0.5,
        clock=clock,
        sleep_fn=clock.sleep,
        on_timeout=timeouts.append,
    )
    assert ok is True and reason == "done" and timeouts == []
    assert clock.t == 1.0
    print("PASS IsGrabbing 0 before the limit is success")


def test_repeated_query_failure_is_not_success() -> None:
    clock = Clock()

    def is_grabbing() -> int:
        raise RuntimeError("pipe closed")

    ok, reason = poll_until_grab_stops(
        is_grabbing,
        timeout_sec=1000,
        sleep_every_sec=0.5,
        clock=clock,
        sleep_fn=clock.sleep,
        on_timeout=lambda _elapsed: (_ for _ in ()).throw(AssertionError("timeout")),
        max_query_errors=5,
    )
    assert ok is False and reason == "query_failed"
    assert clock.t == 2.5
    print("PASS six failed IsGrabbing queries are a failure")


def test_stop_grab_that_clears_the_grab_does_not_kill() -> None:
    clock = Clock()
    state = {"grabbing": 1, "killed": 0, "shots": 0}

    def send_stop() -> None:
        state["grabbing"] = 0

    outcome = release_stuck_grab(
        send_stop,
        lambda: state["grabbing"],
        sleep_fn=clock.sleep,
        clock=clock,
        stop_wait_sec=5,
        kill=lambda: state.__setitem__("killed", state["killed"] + 1),
        screenshot=lambda: state.__setitem__("shots", state["shots"] + 1),
    )
    assert outcome == "stopped"
    assert state["killed"] == 0 and state["shots"] == 0
    print("PASS StopGrab that ends the grab does not kill FLIMage")


def test_grab_that_survives_stop_grab_is_killed() -> None:
    clock = Clock()
    state = {"killed": 0, "shots": 0}
    outcome = release_stuck_grab(
        lambda: None,
        lambda: 1,
        sleep_fn=clock.sleep,
        clock=clock,
        stop_wait_sec=5,
        kill=lambda: state.__setitem__("killed", 1),
        screenshot=lambda: state.__setitem__("shots", 1),
    )
    assert outcome == "killed"
    assert state["shots"] == 1 and state["killed"] == 1
    assert clock.t >= 5
    print("PASS a grab that survives StopGrab is killed after 5 s")


def test_missing_output_file_is_not_ready() -> None:
    clock = Clock()
    ready = wait_until_path_exists(
        os.path.join(tempfile.gettempdir(), "no_such_grab_output.flim"),
        timeout_sec=1,
        clock=clock,
        sleep_fn=clock.sleep,
        exists=lambda _path: False,
    )
    assert ready is False and clock.t > 1
    present = os.path.join(tempfile.gettempdir(), "present.flim")
    ready_now = wait_until_path_exists(
        present,
        timeout_sec=1,
        clock=Clock(),
        sleep_fn=lambda _seconds: None,
        exists=lambda path: path == present,
    )
    assert ready_now is True
    print("PASS a missing FLIM is not ready; an existing one is")


if __name__ == "__main__":
    test_timeout_is_300s_even_for_a_short_expected_duration()
    test_still_grabbing_past_the_old_50s_loop_is_a_timeout()
    test_grab_that_stops_is_success_without_timeout()
    test_repeated_query_failure_is_not_success()
    test_stop_grab_that_clears_the_grab_does_not_kill()
    test_grab_that_survives_stop_grab_is_killed()
    test_missing_output_file_is_not_ready()
    print("all tests passed")
