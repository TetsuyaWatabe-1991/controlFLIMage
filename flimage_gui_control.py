# -*- coding: utf-8 -*-
"""
Generic pywinauto wrapper for driving the live FLIMage WinForms GUI during
debugging, without touching the FLIMage C# code.

Design intent: stay flexible rather than enumerate every control up front.
`find()` is the escape hatch that returns the raw pywinauto control spec, so
anything pywinauto can do is available even if get()/set()/click() don't
cover it. get()/set()/click() only add the "do the right thing so the same
event handler a mouse click/keystroke would trigger actually fires" logic
(see module docstring notes below on why that matters).

Typical session:
    from flimage_gui_control import FlimageGuiControl
    gui = FlimageGuiControl()
    gui.connect()
    gui.dump(depth=2)              # discover auto_id / control_type / text
    gui.get("Power1")               # read whatever is currently displayed
    gui.set("Power1", "50", commit_key="{ENTER}")
    gui.click("FocusButton")
    gui.find("Power1").texts()      # raw pywinauto object for anything else

Why get()/set() dispatch by control_type instead of always using
set_text()/window_text(): WM_SETTEXT-style raw text writes do not reliably
fire the control's TextChanged/ValueChanged handler, so the application
logic tied to that handler (validation, pushing a value to hardware,
recalculating dependent state) may never run even though the control now
displays the new value. click() is used for CheckBox/RadioButton toggling
(instead of directly driving the toggle pattern) for the same reason: a
real click reproduces the WM_COMMAND notification a mouse click would send,
so the existing CheckedChanged/Click handler actually executes.
"""

import datetime
import re
import time

import psutil
from pywinauto import Desktop
from pywinauto.application import Application

from FLIM_pipeClient import FLIM_Com

# FLIMage.exe runs more than one top-level window at once (the main control
# panel plus a separate analysis/display window), and a broad ".*FLIMage.*"
# title match also catches unrelated windows (e.g. an editor with a FLIMage
# source file open). Matching by process name first, then title within that
# process's windows, avoids both kinds of false matches.
FLIMAGE_PROCESS_NAME = "FLIMage"


class FlimageGuiControl:
    def __init__(self, title_re=r"^FLIMage!", backend="uia", log_path=None):
        self.title_re = title_re
        self.backend = backend
        self.log_path = log_path
        self.app = None
        self.window = None
        self.flim = None

    def _flimage_pids(self):
        return {
            p.info["pid"]
            for p in psutil.process_iter(["pid", "name"])
            if FLIMAGE_PROCESS_NAME in (p.info["name"] or "")
        }

    def list_windows(self):
        """List FLIMage's own top-level window titles (helps pick title_re)."""
        pids = self._flimage_pids()
        return [
            w.window_text()
            for w in Desktop(backend=self.backend).windows()
            if w.element_info.process_id in pids
        ]

    def connect(self, timeout=15):
        pids = self._flimage_pids()
        if not pids:
            raise RuntimeError("No running FLIMage process found.")

        candidates = [
            w
            for w in Desktop(backend=self.backend).windows()
            if w.element_info.process_id in pids
        ]
        matches = [w for w in candidates if re.search(self.title_re, w.window_text() or "")]

        if not matches:
            listing = "\n".join("  - {0!r}".format(w.window_text()) for w in candidates)
            raise RuntimeError(
                "No FLIMage window matches title_re={0!r}. Open windows:\n{1}".format(
                    self.title_re, listing
                )
            )
        if len(matches) > 1:
            listing = "\n".join("  - {0!r}".format(w.window_text()) for w in matches)
            raise RuntimeError(
                "{0} FLIMage windows match title_re={1!r}, narrow it down:\n{2}".format(
                    len(matches), self.title_re, listing
                )
            )

        handle = matches[0].handle
        self.app = Application(backend=self.backend).connect(handle=handle, timeout=timeout)
        self.window = self.app.window(handle=handle)
        self.window.wait("exists", timeout=timeout)
        self._log("Connected to window: {0}".format(self.window.window_text()))
        return self

    def _log(self, message):
        line = "{0}, {1}".format(datetime.datetime.now(), message)
        print(line)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ---- discovery ----
    def dump(self, control=None, depth=None):
        """Print the control tree (auto_id / control_type / text) so you can
        find the right name for get/set/click without opening the C# source."""
        target = control or self.window
        target.print_control_identifiers(depth=depth)

    # ---- generic locator: escape hatch for anything not covered below ----
    def find(self, auto_id=None, control_type=None, title=None, timeout=5, **kwargs):
        spec = self.window.child_window(
            auto_id=auto_id, control_type=control_type, title=title, **kwargs
        )
        spec.wait("exists enabled visible", timeout=timeout)
        return spec

    # ---- convenience: read whatever a control currently shows ----
    # CheckBox uses UIA's Toggle pattern (get_toggle_state); RadioButton uses
    # the SelectionItem pattern instead (is_selected) -- they are not
    # interchangeable, and calling the wrong one raises/returns nothing
    # useful, silently falling through to window_text() (the label, not the
    # state) if not handled separately.
    def get(self, auto_id, **kwargs):
        ctrl = self.find(auto_id, **kwargs)
        ctype = ctrl.element_info.control_type
        try:
            if ctype == "Edit":
                return ctrl.get_value()
            if ctype == "CheckBox":
                return ctrl.get_toggle_state()
            if ctype == "RadioButton":
                return ctrl.is_selected()
            if ctype == "ComboBox":
                return ctrl.selected_text()
        except Exception:
            pass
        return ctrl.window_text()

    # ---- convenience: write + commit so the real handler fires ----
    def set(self, auto_id, value, commit_key=None, **kwargs):
        ctrl = self.find(auto_id, **kwargs)
        ctype = ctrl.element_info.control_type
        self._log("SET {0} ({1}) = {2!r}".format(auto_id, ctype, value))

        if ctype == "Edit":
            ctrl.set_edit_text(str(value))
            if commit_key:
                ctrl.type_keys(commit_key)
        elif ctype == "CheckBox":
            if bool(ctrl.get_toggle_state()) != bool(value):
                ctrl.click()
        elif ctype == "RadioButton":
            if not ctrl.is_selected():
                ctrl.click()
        elif ctype == "ComboBox":
            ctrl.select(str(value))
        else:
            ctrl.set_edit_text(str(value))
        return self.get(auto_id, **kwargs)

    # ---- convenience: click a button/menu item/checkbox ----
    def click(self, auto_id, use_mouse=False, **kwargs):
        ctrl = self.find(auto_id, **kwargs)
        self._log("CLICK {0}".format(auto_id))
        if use_mouse:
            ctrl.click_input()  # real OS mouse click; needs window visible/foreground
        else:
            ctrl.click()  # message-based; works even if window is in background
        return True

    # ---- PIPE control: reliable alternative to UIA click() for Focus/Grab ----
    # A UIA click() on FocusButton is delivered through the UI input queue, so
    # during a busy live-scan redraw it can sit unprocessed for a long time (or
    # never be observed to register) instead of actually toggling the button.
    # PIPE commands run on FLIMage's dedicated PipeCmdWorker thread instead,
    # bypassing the UI input queue entirely.
    #
    # IMPORTANT: "StopGrab"/"AbortGrab" is NOT equivalent to a FocusButton
    # click-to-stop. FLIMageMain.cs's ExternalCommand handles them as
    # `if (focusing) StopFocus(); StopGrab(true);` -- StopGrab(true) always
    # runs, even when only Focus (not a real Grab) was active. FLIMage_IO.cs's
    # StopGrab() unconditionally does DisposeDAQ()/ParkMirrors(true)/
    # CloseAllFlimWriters(), none of which FocusButton_Click's stop branch
    # (`else { StopFocus(); }`) touches. Use stop_focus() to faithfully
    # replicate a Focus-button click; reserve stop_grab() for aborting an
    # actual Grab/acquisition (GrabButtonClick's stop branch calls the same
    # StopGrab(true), so that one genuinely matches). Confirmed by Tetsuya
    # 2026-07-10 after stop_grab() caused unexpected behavior when used to
    # stop a Focus-only session.
    def pipe_connect(self, timeout=15):
        if self.flim is not None and self.flim.Connected:
            return self.flim
        self.flim = FLIM_Com()
        self.flim.start()
        if not self.flim.Connected:
            raise RuntimeError("Could not connect to FLIMage's PIPE server.")
        self._log("PIPE connected.")
        return self.flim

    def is_grabbing(self):
        reply = self.pipe_connect().sendCommand("IsGrabbing")
        return bool(int(reply.split(",")[1].strip()))

    def stop_focus(self, timeout=10, poll_interval=0.2):
        """Toggle Focus off via the PIPE "Focus" command, which calls the
        exact same FocusButton_Click() handler a real click does. Only sends
        the toggle while IsGrabbing is true (else "Focus" would start it).
        Returns True once stopped (or already stopped), False on timeout."""
        if not self.is_grabbing():
            self._log("stop_focus: not focusing/grabbing, nothing to do.")
            return True
        self.pipe_connect().sendCommand("Focus")
        self._log("PIPE Focus (toggle-off) sent.")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_grabbing():
                self._log("PIPE confirms focusing stopped.")
                return True
            time.sleep(poll_interval)
        self._log("stop_focus: timed out waiting for IsGrabbing to clear.")
        return False

    def stop_grab(self, timeout=10, poll_interval=0.2):
        """Send the PIPE "StopGrab" command to abort an actual Grab
        acquisition (matches GrabButtonClick's stop branch). Do NOT use this
        to stop a Focus-only session -- see the class-level note above; use
        stop_focus() for that. Polls IsGrabbing until it clears. Returns True
        once stopped, False on timeout."""
        self.pipe_connect().sendCommand("StopGrab")
        self._log("PIPE StopGrab sent.")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_grabbing():
                self._log("PIPE confirms grabbing/focusing stopped.")
                return True
            time.sleep(poll_interval)
        self._log("stop_grab: timed out waiting for IsGrabbing to clear.")
        return False


if __name__ == "__main__":
    gui = FlimageGuiControl()
    gui.connect()
    gui.dump(depth=2)
