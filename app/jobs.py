"""Single-worker background job registry for the Flask app.

Jobs are executed in a background thread that calls canonical `scripts/*.py`
via `subprocess.run`, so the web process never duplicates training logic.

At most one job runs at a time (single lock).  Each job appends its stdout/
stderr lines to a ring buffer that the UI polls via `/api/jobs`.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Optional


_MAX_LOG_LINES = 400


@dataclass
class JobSpec:
    """A single queued/running job."""

    id: str
    kind: str                 # e.g. "train_lstm", "full_pipeline", "reset_lstm"
    label: str                # human-readable title shown in the UI
    status: str = "pending"   # pending | running | success | failed
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_LOG_LINES))
    error: Optional[str] = None
    progress: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "error": self.error,
            "log": list(self.log),
        }


class JobRegistry:
    """Thread-safe registry that executes at most one job at a time."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._lock = threading.Lock()
        self._current: Optional[JobSpec] = None
        self._history: Deque[JobSpec] = deque(maxlen=10)

    # --- public API -------------------------------------------------------

    def current(self) -> Optional[dict]:
        with self._lock:
            return self._current.to_dict() if self._current else None

    def recent(self, n: int = 5) -> list[dict]:
        with self._lock:
            items = list(self._history)[-n:]
            if self._current and self._current not in items:
                items.append(self._current)
            return [j.to_dict() for j in items]

    def submit(
        self,
        kind: str,
        label: str,
        runner: Callable[[JobSpec], None],
    ) -> tuple[bool, dict]:
        """Start a job if no other job is running. Returns (accepted, job_dict)."""
        with self._lock:
            if self._current and self._current.status == "running":
                return False, self._current.to_dict()

            job = JobSpec(id=uuid.uuid4().hex[:8], kind=kind, label=label)
            self._current = job

        t = threading.Thread(
            target=self._execute, args=(job, runner), daemon=True,
            name=f"job-{job.id}",
        )
        t.start()
        return True, job.to_dict()

    def run_subprocess(self, job: JobSpec, cmd: list[str]) -> int:
        """Stream stdout of a subprocess into the job log and return the exit code."""
        job.log.append(f"$ {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(self._project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                job.log.append(line)
                job.progress = line[:180]
        proc.wait()
        return int(proc.returncode)

    def python_script(self, job: JobSpec, script: str, *args: str) -> int:
        """Run `python -u scripts/<script> <args>` and stream output."""
        script_path = self._project_root / "scripts" / script
        cmd = [sys.executable, "-u", str(script_path), *args]
        return self.run_subprocess(job, cmd)

    # --- internal ---------------------------------------------------------

    def _execute(self, job: JobSpec, runner: Callable[[JobSpec], None]) -> None:
        job.status = "running"
        job.started_at = datetime.utcnow().isoformat()
        try:
            runner(job)
            job.status = "success"
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)
            job.log.append(f"[ERROR] {exc}")
        finally:
            job.finished_at = datetime.utcnow().isoformat()
            with self._lock:
                self._history.append(job)
