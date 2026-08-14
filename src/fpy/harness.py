"""Client for the C++ test harness that runs sequences on the real
Svc::FpySequencer (see test/harness). The harness program is started once and
reused: each request is one JSON line on its stdin, each reply one JSON line
on its stdout, and every request runs on a brand-new sequencer instance
inside the harness."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
FPY_HARNESS_BINARY = (
    REPO_ROOT / "build-artifacts" / "Linux" / "FpyHarness" / "bin" / "FpyHarness"
)
_WASM_HARNESS_ROOT = REPO_ROOT / "test" / "harness" / "wasm"
WASM_HARNESS_BINARY = (
    _WASM_HARNESS_ROOT
    / "build-artifacts"
    / "Linux"
    / "WasmSeqHarness"
    / "bin"
    / "WasmSeqHarness"
)


class HarnessError(Exception):
    """The harness itself failed: it could not run the sequence at all, gave
    a malformed reply, or crashed. Distinct from a sequence failing."""


def build_harness() -> None:
    """Builds the FpySequencer harness executable from the fprime submodule.
    The build is incremental, so this is cheap when nothing changed."""
    if not (REPO_ROOT / "test" / "fprime" / "CMakeLists.txt").exists():
        raise HarnessError(
            "the fprime submodule is not checked out. Run:\n"
            "  git submodule update --init test/fprime"
        )
    _fprime_util_build(REPO_ROOT)


def build_wasm_harness() -> None:
    """Builds the WasmSequencer harness executable from the fprime-wasm
    submodule, with the exit-code patch applied (test/harness/patches)."""
    if not (REPO_ROOT / "test" / "fprime-wasm" / "CMakeLists.txt").exists():
        raise HarnessError(
            "the fprime-wasm submodule is not checked out. Run:\n"
            "  git submodule update --init test/fprime-wasm"
        )
    _apply_wasm_patch()
    _fprime_util_build(_WASM_HARNESS_ROOT, build_args=["--all"])


def _apply_wasm_patch() -> None:
    """Applies the local WasmSequencer fixes to the fprime-wasm submodule if
    they are not applied yet (see test/harness/patches/README.md)."""
    patch = REPO_ROOT / "test" / "harness" / "patches" / "wasm-sequencer-fixes.patch"
    submodule = REPO_ROOT / "test" / "fprime-wasm"
    applied = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch)],
        cwd=submodule,
        capture_output=True,
    )
    if applied.returncode == 0:
        return
    result = subprocess.run(
        ["git", "apply", str(patch)], cwd=submodule, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise HarnessError(
            f"could not apply {patch.name} to test/fprime-wasm:\n{result.stderr}"
        )


def _fprime_util_build(deployment: Path, build_args: list[str] = None) -> None:
    if not (deployment / "build-fprime-automatic-native").exists():
        _run_fprime_util(deployment, ["generate"])
    _run_fprime_util(deployment, ["build"] + (build_args or []))


def _run_fprime_util(deployment: Path, args: list[str]) -> None:
    try:
        result = subprocess.run(
            ["fprime-util"] + args,
            cwd=deployment,
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
            f"'fprime-util {' '.join(args)}' in {deployment} failed. Are the "
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
                f"harness binary {self._binary} does not exist; build it with build_harness()"
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


def fpy_harness() -> SequencerHarness:
    """The shared harness for the fpy bytecode backend, building its binary
    on first use."""
    global _fpy_harness, _fpy_build_error
    if _fpy_build_error is not None:
        raise _fpy_build_error
    if _fpy_harness is None:
        try:
            build_harness()
        except HarnessError as e:
            _fpy_build_error = e
            raise
        _fpy_harness = SequencerHarness(FPY_HARNESS_BINARY)
    return _fpy_harness


def wasm_harness() -> SequencerHarness:
    """The shared harness for the LLVM/wasm backend."""
    global _wasm_harness
    if _wasm_harness is None:
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
