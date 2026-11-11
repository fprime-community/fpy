"""Client for the C++ test harnesses that run sequences on the real
Svc::FpySequencer and Svc::WasmSequencer (see test/harness). Each harness
program is started once and reused: each request is one JSON line on its
stdin, each reply one JSON line on its stdout, and every request runs on a
brand-new sequencer instance inside the harness."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
_BUILD_ARTIFACTS = REPO_ROOT / "build-artifacts" / "Linux"
FPY_HARNESS_BINARY = _BUILD_ARTIFACTS / "FpySeqHarness" / "bin" / "FpySeqHarness"
WASM_HARNESS_BINARY = _BUILD_ARTIFACTS / "WasmSeqHarness" / "bin" / "WasmSeqHarness"


class HarnessError(Exception):
    """The harness itself failed: it could not run the sequence at all, gave
    a malformed reply, or crashed. Distinct from a sequence failing."""


def build_fpy_harness() -> None:
    """Builds the FpySequencer harness executable from the fprime submodule.
    The build is incremental, so this is cheap when nothing changed."""
    _build_deployment("FpySeqHarness")


def build_wasm_harness() -> None:
    """Builds the WasmSequencer harness executable from the fprime submodule.
    The build is incremental, so this is cheap when nothing changed."""
    _build_deployment("WasmSeqHarness")


def _build_deployment(target: str) -> None:
    """Builds and installs one of the harness deployments (see the root
    CMakeLists.txt), generating the shared build cache first if needed."""
    if not (REPO_ROOT / "test" / "fprime" / "CMakeLists.txt").exists():
        raise HarnessError(
            "the fprime submodule is not checked out. Run:\n"
            "  git submodule update --init test/fprime"
        )
    if not (REPO_ROOT / "build-fprime-automatic-native").exists():
        _run_fprime_util(["generate"])
    _run_fprime_util(["build", "--target", target])


def _run_fprime_util(args: list[str]) -> None:
    try:
        result = subprocess.run(
            ["fprime-util"] + args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise HarnessError(
            "fprime-util was not found. Are the harness build tools "
            "installed (uv sync installs them)?"
        )
    if result.returncode != 0:
        raise HarnessError(
            f"'fprime-util {' '.join(args)}' failed. Are the "
            "harness build tools installed (uv sync installs them)?\n"
            + result.stdout[-2000:]
            + result.stderr[-2000:]
        )


class SequencerHarness:
    """One running harness process."""

    def __init__(self, binary: Path):
        self._binary = binary
        self._process: subprocess.Popen | None = None
        self._stderr_file = None

    def run(self, request: dict) -> dict:
        """Sends one run request and returns the harness's reply."""
        if self._process is None or self._process.poll() is not None:
            self._start()
        try:
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
            reply = self._process.stdout.readline()
        except OSError as e:
            reply = ""
        if not reply:
            # The harness died (an assertion failure in the sequencer, most
            # likely). Report what it said and restart on the next run.
            stderr = self._read_stderr()
            self.close()
            raise HarnessError(
                f"harness process died while running {request.get('seqFile')}:\n{stderr}"
            )
        return json.loads(reply)

    def _start(self) -> None:
        if not self._binary.exists():
            raise HarnessError(
                f"harness binary {self._binary} does not exist; build it with "
                "build_fpy_harness() or build_wasm_harness()"
            )
        self._stderr_file = tempfile.TemporaryFile(mode="w+")
        self._process = subprocess.Popen(
            [str(self._binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
        )

    def _read_stderr(self) -> str:
        assert self._stderr_file is not None
        self._stderr_file.seek(0)
        return self._stderr_file.read()

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.stdin.close()
            self._process.wait(timeout=10)
        if self._stderr_file is not None:
            self._stderr_file.close()
        self._process = None
        self._stderr_file = None


_fpy_harness: SequencerHarness | None = None
_wasm_harness: SequencerHarness | None = None
# The first failed build, re-raised on later calls: retrying the build once
# it has failed only repeats the same slow failure.
_fpy_build_error: HarnessError | None = None
_wasm_build_error: HarnessError | None = None


def fpy_harness() -> SequencerHarness:
    """The shared harness for the fpy bytecode backend, building its binary
    on first use."""
    global _fpy_harness, _fpy_build_error
    if _fpy_build_error is not None:
        raise _fpy_build_error
    if _fpy_harness is None:
        try:
            build_fpy_harness()
        except HarnessError as e:
            _fpy_build_error = e
            raise
        _fpy_harness = SequencerHarness(FPY_HARNESS_BINARY)
    return _fpy_harness


def wasm_harness() -> SequencerHarness:
    """The shared harness for the LLVM/wasm backend, building its binary on
    first use."""
    global _wasm_harness, _wasm_build_error
    if _wasm_build_error is not None:
        raise _wasm_build_error
    if _wasm_harness is None:
        try:
            build_wasm_harness()
        except HarnessError as e:
            _wasm_build_error = e
            raise
        _wasm_harness = SequencerHarness(WASM_HARNESS_BINARY)
    return _wasm_harness


def close_all() -> None:
    """Stops any running harness processes (end of the test session)."""
    global _fpy_harness, _wasm_harness
    if _fpy_harness is not None:
        _fpy_harness.close()
        _fpy_harness = None
    if _wasm_harness is not None:
        _wasm_harness.close()
        _wasm_harness = None
