"""Keep one nnU-Net process alive across highmag files.

Watch mode used to start ``nnUNetv2_predict`` for every new FLIM, which reloads
the weights each time. This module starts the respan_nnunet interpreter once,
loads ``nnUNetPredictor`` once, and sends each prepared folder over a localhost
socket. Stdout stays a log stream so prediction progress is not mixed into the
protocol.

``--fake`` speaks the same protocol without CUDA, for tests.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PORT_PREFIX = "RESPAN_NNUNET_PORT "
_READY_TIMEOUT_SEC = 900.0
_PREDICT_TIMEOUT_SEC = 1800.0
_PORT_TIMEOUT_SEC = 60.0


class ResidentError(RuntimeError):
    """The resident nnU-Net process stopped or rejected a job."""


def _worker_command(fake: bool = False, fold: str = "all") -> list[str]:
    script = str(Path(__file__).resolve())
    if fake:
        return [sys.executable, script, "--fake"]
    from respan_runner.paths import nnunet_trainer_dir, respan_nnunet_python

    return [
        str(respan_nnunet_python()),
        script,
        "--serve",
        "--model-dir",
        str(nnunet_trainer_dir()),
        "--fold",
        str(fold),
    ]


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = ""
    return env


class ResidentNnUNet:
    """Client for one long-lived nnU-Net process."""

    def __init__(self, command: list[str] | None = None) -> None:
        self.command = command if command is not None else _worker_command()
        self._proc: subprocess.Popen[str] | None = None
        self._sock: socket.socket | None = None
        self._buf = b""
        self._drain: threading.Thread | None = None
        self.last_message: dict[str, Any] = {}
        self.pid: int | None = None

    def start(self) -> ResidentNnUNet:
        """Start the worker and wait until the model is loaded."""
        if self._proc is not None:
            return self
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=_worker_env(),
        )
        assert self._proc.stdout is not None
        port = self._read_port()
        self.pid = self._proc.pid
        self._drain = threading.Thread(target=self._drain_stdout, name="nnunet-resident-log", daemon=True)
        self._drain.start()
        self._sock = socket.create_connection(("127.0.0.1", port), timeout=30)
        ready = self._recv(_READY_TIMEOUT_SEC)
        self.last_message = ready
        if ready.get("event") != "ready":
            message = str(ready.get("message", ready))
            self.close()
            raise ResidentError(message)
        print("nnU-Net model is loaded and will be reused for later highmag files.", flush=True)
        return self

    def predict(self, input_dir: str, output_dir: str) -> int:
        """Segment one prepared folder. The weights stay in the worker."""
        self._ensure_alive()
        try:
            return self._predict_once(input_dir, output_dir)
        except ResidentError:
            if self._proc is not None and self._proc.poll() is None:
                raise
            print("nnU-Net resident stopped. Reloading the model once.", flush=True)
            self.close()
            self.start()
            return self._predict_once(input_dir, output_dir)

    def close(self) -> None:
        """Ask the worker to exit and release the GPU."""
        sock = self._sock
        proc = self._proc
        self._sock = None
        self._proc = None
        if sock is not None:
            try:
                sock.sendall(b'{"cmd":"shutdown"}\n')
                sock.settimeout(30)
                sock.recv(4096)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        self._buf = b""

    def _predict_once(self, input_dir: str, output_dir: str) -> int:
        self._send({"cmd": "predict", "input_dir": input_dir, "output_dir": output_dir})
        message = self._recv(_PREDICT_TIMEOUT_SEC)
        self.last_message = message
        if message.get("event") == "error":
            raise ResidentError(str(message.get("message", "nnU-Net predict failed")))
        if message.get("event") != "done":
            raise ResidentError(f"unexpected resident reply: {message}")
        return int(message.get("returncode", 1))

    def _ensure_alive(self) -> None:
        if self._proc is None or self._sock is None:
            raise ResidentError("nnU-Net resident is not running")
        if self._proc.poll() is not None:
            raise ResidentError("nnU-Net resident exited")

    def _send(self, payload: dict[str, Any]) -> None:
        if self._sock is None:
            raise ResidentError("nnU-Net resident is not connected")
        data = (json.dumps(payload) + "\n").encode("utf-8")
        self._sock.sendall(data)

    def _recv(self, timeout: float) -> dict[str, Any]:
        if self._sock is None:
            raise ResidentError("nnU-Net resident is not connected")
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResidentError("timed out waiting for the nnU-Net resident")
            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout as exc:
                raise ResidentError("timed out waiting for the nnU-Net resident") from exc
            if not chunk:
                raise ResidentError("nnU-Net resident closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def _read_port(self) -> int:
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.monotonic() + _PORT_TIMEOUT_SEC
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if line == "" and self._proc.poll() is not None:
                raise ResidentError("nnU-Net resident exited before it published a port")
            text = line.strip()
            if text:
                print(text, flush=True)
            if text.startswith(PORT_PREFIX):
                return int(text[len(PORT_PREFIX):])
        raise ResidentError("timed out waiting for the nnU-Net resident port")

    def _drain_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.rstrip()
            if text:
                print(text, flush=True)


def _send_line(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def _read_line(conn: socket.socket, buf: bytearray) -> tuple[dict[str, Any] | None, bytearray]:
    while b"\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            return None, buf
        buf.extend(chunk)
    line, _, rest = buf.partition(b"\n")
    return json.loads(line.decode("utf-8")), bytearray(rest)


def _serve_loop(conn: socket.socket, predict_fn) -> None:
    buf = bytearray()
    while True:
        message, buf = _read_line(conn, buf)
        if message is None:
            return
        command = message.get("cmd")
        if command == "shutdown":
            _send_line(conn, {"event": "done", "returncode": 0})
            return
        if command != "predict":
            _send_line(conn, {"event": "error", "message": f"unknown command: {command}"})
            continue
        try:
            predict_fn(str(message["input_dir"]), str(message["output_dir"]))
        except Exception as exc:
            _send_line(conn, {"event": "error", "message": str(exc)})
            continue
        _send_line(conn, {"event": "done", "returncode": 0, "loads": 1})


def _accept_client() -> tuple[socket.socket, socket.socket]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])
    print(f"{PORT_PREFIX}{port}", flush=True)
    conn, _addr = server.accept()
    return server, conn


def serve_fake() -> None:
    """Protocol stand-in. Loads nothing and marks each output folder."""
    server, conn = _accept_client()
    try:
        _send_line(conn, {"event": "ready", "loads": 1})

        def _predict(input_dir: str, output_dir: str) -> None:
            del input_dir
            os.makedirs(output_dir, exist_ok=True)
            Path(output_dir, "resident_marker.txt").write_text("1", encoding="utf-8")

        _serve_loop(conn, _predict)
    finally:
        conn.close()
        server.close()


def serve_real(model_dir: str, fold: str) -> None:
    """Load nnU-Net once, then segment each folder that arrives on the socket."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    server, conn = _accept_client()
    try:
        try:
            if not torch.cuda.is_available():
                raise ResidentError("CUDA is not available in the nnU-Net environment")
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device("cuda"),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True,
            )
            use_fold: tuple[str | int, ...] = (fold,)
            predictor.initialize_from_trained_model_folder(
                model_dir,
                use_fold,
                checkpoint_name="checkpoint_final.pth",
            )
        except Exception as exc:
            _send_line(conn, {"event": "error", "message": str(exc)})
            return
        _send_line(conn, {"event": "ready", "loads": 1})

        def _predict(input_dir: str, output_dir: str) -> None:
            predictor.predict_from_files(
                input_dir,
                output_dir,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=3,
                num_processes_segmentation_export=3,
                num_parts=1,
                part_id=0,
            )

        _serve_loop(conn, _predict)
    finally:
        conn.close()
        server.close()


@contextmanager
def predict_hook(module: Any, resident: ResidentNnUNet) -> Iterator[None]:
    """Route ``module.run_nnunet_predict`` to the resident process."""
    original = module.run_nnunet_predict

    def _predict(
        nnunet_predict_bat: str,
        input_dir: str,
        output_dir: str,
        dataset_id: str,
        nnunet_type: str,
        settings: Any,
        logger: Any,
    ) -> int:
        del nnunet_predict_bat, dataset_id, nnunet_type
        if hasattr(module, "initialize_nnUnet"):
            module.initialize_nnUnet(settings, logger)
        logger.info("Using resident nnU-Net. The model stays loaded.")
        fold = str(getattr(settings, "nnunet_fold", "all"))
        return resident.predict(str(input_dir), str(output_dir), fold=fold)

    module.run_nnunet_predict = _predict
    try:
        yield
    finally:
        module.run_nnunet_predict = original


@contextmanager
def resident_nnunet_session(command: list[str] | None = None) -> Iterator[Any]:
    """Patch RESPAN so each new stack reuses one loaded model.

    The process starts on the first prediction, not when the watch loop begins,
    so an idle watcher does not hold the GPU before the first highmag arrives.
    """
    from RESPAN.Environment import sr

    holder: dict[str, ResidentNnUNet | None] = {"resident": None}

    class _Lazy:
        def predict(self, input_dir: str, output_dir: str, fold: str = "all") -> int:
            resident = holder["resident"]
            if resident is None:
                launch = command if command is not None else _worker_command(fold=fold)
                resident = ResidentNnUNet(launch).start()
                holder["resident"] = resident
            return resident.predict(input_dir, output_dir)

        def close(self) -> None:
            resident = holder["resident"]
            if resident is not None:
                resident.close()

    lazy = _Lazy()
    try:
        with predict_hook(sr, lazy):
            yield lazy  # type: ignore[misc]
    finally:
        lazy.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resident nnU-Net predictor.")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--fold", default="all")
    args = parser.parse_args()
    if args.fake:
        serve_fake()
        return
    if not args.serve:
        parser.error("pass --serve or --fake")
    serve_real(args.model_dir, args.fold)


if __name__ == "__main__":
    main()
