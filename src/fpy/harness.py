"""Client for the sequencer harness, which runs sequences on a real
``Svc::FpySequencer``.

The harness is a long-lived process speaking one JSON object per line in each
direction. It builds a fresh component per request, so runs do not affect each
other, and it is reused across a whole test session, so a run costs a pipe
round trip rather than a process spawn.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class HarnessError(RuntimeError):
    """The harness could not carry out a request."""


class HarnessAssertError(HarnessError):
    """The sequencer hit an FW_ASSERT."""


class HarnessCrashed(HarnessError):
    """The harness process died mid-request."""


@dataclass
class RunResult:
    """What running one sequence produced."""

    error_code: int = 0
    exit_code: int | None = None
    validation_failed: bool = False
    stack_size: int = 0
    statements_dispatched: int = 0
    final_state: str = ""
    sim_time_us: int = 0
    error: str = ""
    cmds: list[bytes] = field(default_factory=list)
    events: list[tuple[int, str]] = field(default_factory=list)
    serial: list[tuple[int, bytes]] = field(default_factory=list)


class Harness:
    """A harness process. Restarts itself if it dies."""

    def __init__(self, binary: str, read_timeout_s: float = 30.0):
        self.binary = binary
        self.read_timeout_s = read_timeout_s
        self._process: subprocess.Popen | None = None
        self._next_id = 0

    def _start(self):
        self._process = subprocess.Popen(
            [self.binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _request(self, payload: dict) -> dict:
        if self._process is None or self._process.poll() is not None:
            self._start()
        self._next_id += 1
        payload["id"] = self._next_id

        assert self._process.stdin is not None
        assert self._process.stdout is not None
        try:
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        except (BrokenPipeError, ValueError) as e:
            self._reap()
            raise HarnessCrashed(f"harness died writing request: {e}") from e

        if not line:
            detail = self._reap()
            raise HarnessCrashed(f"harness died running {payload.get('op')}: {detail}")
        return json.loads(line)

    def _reap(self) -> str:
        """Collect the dead process's stderr and forget it."""
        if self._process is None:
            return "no process"
        stderr = ""
        try:
            self._process.kill()
            _, stderr = self._process.communicate(timeout=5)
        except Exception:
            pass
        self._process = None
        return (stderr or "").strip()[-2000:]

    def close(self):
        if self._process is None:
            return
        try:
            self._request({"op": "quit"})
        except Exception:
            pass
        self._reap()

    def info(self) -> dict:
        """The harness's schema version and compile-time constants."""
        return self._request({"op": "ping"})

    def run(
        self,
        seq_path: str,
        cwd: str | None = None,
        args: bytes | None = None,
        tlm: dict[int, bytes] | None = None,
        prm: dict[int, bytes] | None = None,
        time_base: int = 0,
        time_context: int = 0,
        initial_time_us: int = 0,
        fail_opcodes: set[int] | None = None,
        seq_run_opcodes: set[int] | None = None,
        cmd_response: int = 0,
        disconnected_serial_ports: set[int] | None = None,
    ) -> RunResult:
        """Run the sequence at *seq_path*, resolved against *cwd*."""
        response = self._request(
            {
                "op": "run",
                "seq_path": seq_path,
                "cwd": cwd or "",
                "args_hex": (args or b"").hex(),
                "tlm": {str(k): v.hex() for k, v in (tlm or {}).items()},
                "prm": {str(k): v.hex() for k, v in (prm or {}).items()},
                "time": {
                    "base": time_base,
                    "context": time_context,
                    "initial_us": initial_time_us,
                },
                "fail_opcodes": sorted(fail_opcodes or ()),
                "seq_run_opcodes": sorted(seq_run_opcodes or ()),
                "cmd_response": cmd_response,
                "disconnected_serial_ports": sorted(disconnected_serial_ports or ()),
            }
        )

        if not response.get("ok", False):
            kind = response.get("error", "unknown")
            detail = response.get("detail", "")
            if kind == "assert":
                raise HarnessAssertError(detail)
            if kind in ("stalled", "dispatch_limit", "bad_cwd"):
                raise HarnessError(f"{kind}: {detail}")
            raise HarnessError(f"{kind}: {detail}")

        return RunResult(
            error_code=response["error_code"],
            exit_code=response["exit_code"],
            validation_failed=response["validation_failed"],
            stack_size=response["stack_size"],
            statements_dispatched=response["statements_dispatched"],
            final_state=response["final_state"],
            sim_time_us=response["sim_time_us"],
            error=response["error"],
            cmds=[bytes.fromhex(c) for c in response["cmds"]],
            events=[(sev, text) for sev, text in response["events"]],
            serial=[(port, bytes.fromhex(data)) for port, data in response["serial"]],
        )


def binary_path(repo_root: Path) -> Path:
    """Where the built harness lands."""
    return repo_root / "test" / "harness" / "build" / "bin" / "Linux" / "FpyHarness"
