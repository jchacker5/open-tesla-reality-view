"""Single-worker FIFO job queue for studio builds (one GPU, one build at a time).

A job is a list of phases. A command phase runs as a subprocess in its own session (so cancel can kill
the whole process group: otrv -> colmap / brush) under a PTY (so Brush prints its progress bar). Output
is parsed by progress.Tracker and written to work/<name>/events.jsonl, which the server streams over SSE.

work/<name>/
  job.json         spec + state + the build's process-group id (survives restarts: a job that was running
                   becomes "interrupted", or "orphaned-running" while its process group is still alive)
  events.jsonl     {"id", "t", "type", ...} one per line; ids increase across resumes
  studio.log       raw output
  stages/<s>.done  {"seconds", "units"}: completed stages (skipped on resume, and the ETA prior)
"""

from __future__ import annotations

import json
import os
import pty
import re
import select
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import progress

TERMINAL = ("done", "failed", "cancelled", "interrupted")
# A job that was running when an earlier studio died and whose process group is still alive. Not terminal
# (it still holds work/<name> and the GPU); resume is refused until it's gone, and cancel/kill ends it.
ORPHANED = "orphaned-running"
INTERRUPTED_MSG = "the studio stopped while this build was running; resume to continue"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
KILL_GRACE = 5.0  # seconds between SIGTERM and SIGKILL


def valid_name(name: str) -> bool:
    return bool(NAME_RE.fullmatch(name)) and ".." not in name


def pgid_alive(pgid: int | None) -> bool:
    """True if any process is still in process group `pgid`."""
    if not pgid or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, but not ours
        return True
    return True


def kill_group(pgid: int, grace: float | None = None, proc: subprocess.Popen | None = None) -> None:
    """SIGTERM process group `pgid`, wait up to `grace` seconds for it to go, then SIGKILL what's left.
    Blocks; with `proc` (the group leader, our child) it is also reaped."""
    grace = KILL_GRACE if grace is None else grace
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    end = time.monotonic() + grace
    while time.monotonic() < end:
        if proc is not None:
            try:
                proc.wait(0.05)
            except subprocess.TimeoutExpired:
                continue
            proc = None  # leader reaped; grandchildren may linger
        elif not pgid_alive(pgid):
            return
        else:
            time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)  # also reaps grandchildren that ignored SIGTERM
    except (ProcessLookupError, PermissionError):
        pass
    if proc is not None:
        try:
            proc.wait(2)
        except subprocess.TimeoutExpired:
            pass


@dataclass
class Phase:
    """One step of a job. `run` is an argv (subprocess under a PTY) or a callable(job, log) run in-process.
    `stages` are the progress stages this phase covers; a phase is skipped when `skip(job)` is true."""

    name: str
    run: list[str] | Callable
    stages: tuple[str, ...]
    skip: Callable[[Job], bool] = lambda job: False
    required: bool = True  # a failed optional phase (e.g. bench) is logged, not fatal
    before: Callable | None = None  # in-process setup run before `run`, e.g. copying the footage


@dataclass
class Job:
    name: str
    dir: Path
    spec: dict
    state: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    stage: str | None = None
    frac: dict = field(default_factory=dict)
    stage_eta: float | None = None
    eta: float | None = None
    attempts: int = 0
    pgid: int | None = None  # process group of the running command (recorded in job.json)
    cancel_requested: bool = False
    interrupted: bool = False  # stopped by SIGTERM (studio shutdown, or someone else): resumable
    proc: subprocess.Popen | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.name,
            "name": self.name,
            "spec": self.spec,
            "state": self.state,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "error": self.error,
            "stage": self.stage,
            "frac": self.frac,
            "stage_eta": self.stage_eta,
            "eta": self.eta,
            "attempts": self.attempts,
            "pgid": self.pgid,
            "stages_done": [s for s in progress.STAGES if (self.dir / "stages" / f"{s}.done").exists()],
        }

    def save(self) -> None:
        d = self.to_dict()
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(self.dir / "job.json")

    @classmethod
    def load(cls, path: Path) -> Job | None:
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        job = cls(name=d["name"], dir=path.parent, spec=d.get("spec", {}))
        for k in ("state", "created", "started", "finished", "error", "stage", "frac", "attempts", "pgid"):
            if k in d and d[k] is not None:
                setattr(job, k, d[k])
        if job.state in ("running", ORPHANED) and pgid_alive(job.pgid):
            # an earlier studio died without killing its build, which is still running: don't start a
            # second one over the same work dir
            job.state = ORPHANED
            job.error = (
                f"a build from an earlier studio session is still running (process group {job.pgid}); "
                "kill it before resuming"
            )
        elif job.state in ("queued", "running", ORPHANED):  # the server stopped mid-build
            job.state, job.pgid = "interrupted", None
            job.error = INTERRUPTED_MSG
        return job


class EventLog:
    """Append-only events.jsonl with increasing ids (continues after the last id on resume)."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.next_id = last_id(path) + 1

    def emit(self, event: dict) -> dict:
        with self.lock:
            ev = {"id": self.next_id, "t": round(time.time(), 3), **event}
            self.next_id += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(ev) + "\n")
            return ev


def last_id(path: Path) -> int:
    if not path.exists():
        return 0
    last = 0
    with open(path, "rb") as f:
        for line in f:
            try:
                last = max(last, int(json.loads(line)["id"]))
            except (ValueError, KeyError, TypeError):
                continue
    return last


def read_events(path: Path, after: int = 0) -> list[dict]:
    """Events with id > after (for SSE replay via Last-Event-ID)."""
    out = []
    if not path.exists():
        return out
    with open(path, "rb") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except ValueError:
                continue  # a line being written right now
            if ev.get("id", 0) > after:
                out.append(ev)
    return out


class Runner:
    """FIFO queue + one worker thread. `plan(job)` returns the job's phases; `units(job)` returns
    (n_images, steps, map_total) for progress totals and ETAs; `probe(job, stage)` optionally returns
    (cur, total) for stages whose tools print no progress (frame extraction, masks)."""

    def __init__(
        self,
        work: Path,
        plan: Callable[[Job], list[Phase]],
        units: Callable[[Job], tuple[int, int, int | None]] = lambda job: (0, 0, None),
        probe: Callable[[Job, str], tuple[int, int] | None] | None = None,
        cwd: Path | None = None,
        env: dict | None = None,
        progress_interval: float = 0.5,
    ):
        self.work = Path(work)
        self.plan = plan
        self.units = units
        self.probe = probe
        self.cwd = cwd
        self.env = env
        self.progress_interval = progress_interval
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.cond = threading.Condition()
        self.current: Job | None = None
        self._stop = False
        self._thread: threading.Thread | None = None
        self.rates: dict[str, float] = {}
        self._logs: dict[str, EventLog] = {}
        self._load()

    # ---- lifecycle ----

    def _load(self) -> None:
        if not self.work.is_dir():
            return
        for p in sorted(self.work.glob("*/job.json")):
            job = Job.load(p)
            if job and job.name == p.parent.name:
                self.jobs[job.name] = job
                job.save()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="otrv-studio-worker", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 10) -> None:
        """Stop the worker and the running build: SIGTERM its process group, SIGKILL after KILL_GRACE
        seconds. The build is marked "interrupted" (resumable). Blocks until done; safe to call twice."""
        with self.cond:
            self._stop = True
            self.cond.notify_all()
        job = self.current
        if job:
            job.interrupted = True
            proc = job.proc
            if proc and proc.poll() is None:
                kill_group(proc.pid, proc=proc)
        if self._thread:
            self._thread.join(timeout)
        if job and job.state not in TERMINAL:  # the worker didn't get there in time (or is stuck in-process)
            job.proc = job.pgid = None
            self._set_state(job, "interrupted", INTERRUPTED_MSG)

    @property
    def stopping(self) -> bool:
        return self._stop

    # ---- API ----

    def events_path(self, name: str) -> Path:
        return self.work / name / "events.jsonl"

    def log(self, name: str) -> EventLog:
        """The one EventLog per job, so ids stay unique across threads."""
        with self.cond:
            if name not in self._logs:
                self._logs[name] = EventLog(self.events_path(name))
            return self._logs[name]

    def get(self, name: str) -> Job | None:
        job = self.jobs.get(name)
        if job and job.state == ORPHANED:
            self._reap_orphan(job)
        return job

    def _reap_orphan(self, job: Job) -> None:
        if job.state == ORPHANED and not pgid_alive(job.pgid):
            job.pgid = None
            self._set_state(job, "interrupted", "the orphaned build has stopped; resume to continue")

    def list(self) -> list[Job]:
        for job in list(self.jobs.values()):
            self._reap_orphan(job)
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def submit(self, name: str, spec: dict) -> Job:
        if not valid_name(name):
            raise ValueError(f"invalid name {name!r}")
        with self.cond:
            old = self.jobs.get(name)
            if old and old.state not in TERMINAL:
                raise ValueError(f"{name} is already {old.state}")
            job = Job(name=name, dir=self.work / name, spec=spec)
            job.dir.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(job.dir / "stages", ignore_errors=True)
            self.events_path(name).unlink(missing_ok=True)
            self._logs.pop(name, None)
            (job.dir / "studio.log").unlink(missing_ok=True)
            self.jobs[name] = job
            job.save()
            self.log(name).emit({"type": "state", "state": "queued"})
            self.queue.append(name)
            self.cond.notify_all()
        return job

    def resume(self, name: str) -> Job:
        with self.cond:
            job = self.jobs.get(name)
            if not job:
                raise KeyError(name)
            self._reap_orphan(job)
            if job.state == ORPHANED:
                raise ValueError(
                    f"{name} is still running from an earlier studio session (process group {job.pgid}); kill it first"
                )
            if job.state not in ("failed", "cancelled", "interrupted"):
                raise ValueError(f"{name} is {job.state}; only failed, cancelled or interrupted builds resume")
            job.state, job.error, job.cancel_requested, job.finished = "queued", None, False, None
            job.save()
            self.log(name).emit({"type": "state", "state": "queued", "resume": True})
            self.queue.append(name)
            self.cond.notify_all()
        return job

    def cancel(self, name: str) -> Job:
        with self.cond:
            job = self.jobs.get(name)
            if not job:
                raise KeyError(name)
            if job.state in TERMINAL:
                return job
            if job.state == ORPHANED:
                return self.kill_orphan(name)
            job.cancel_requested = True
            if job.state == "queued":
                if name in self.queue:
                    self.queue.remove(name)
                self._set_state(job, "cancelled")
                return job
            proc = job.proc
        if proc:
            self._kill(proc)
        return job

    def kill_orphan(self, name: str, wait: bool = False) -> Job:
        """Kill the process group of an orphaned-running job (SIGTERM, SIGKILL after KILL_GRACE s); the
        job becomes "interrupted" once the group is gone. Returns at once unless `wait`."""
        job = self.jobs.get(name)
        if not job:
            raise KeyError(name)
        if job.state != ORPHANED:
            raise ValueError(f"{name} is {job.state}, not orphaned")
        pgid = job.pgid

        def run():
            if pgid:
                kill_group(pgid)
            with self.cond:
                self._reap_orphan(job)

        if wait:
            run()
        else:
            threading.Thread(target=run, name=f"otrv-kill-{pgid}", daemon=True).start()
        return job

    def forget(self, name: str) -> None:
        with self.cond:
            job = self.jobs.get(name)
            if job and job.state not in TERMINAL:
                raise ValueError(f"{name} is {job.state}; cancel it first")
            self.jobs.pop(name, None)
            self._logs.pop(name, None)

    def busy(self, name: str) -> bool:
        job = self.jobs.get(name)
        return bool(job and job.state not in TERMINAL)

    # ---- worker ----

    def _set_state(self, job: Job, state: str, error: str | None = None) -> None:
        job.state = state
        if error:
            job.error = error
        if state in TERMINAL:
            job.finished = time.time()
            job.eta = 0.0 if state == "done" else None
        job.save()
        self.log(job.name).emit({"type": "state", "state": state, **({"error": error} if error else {})})

    @staticmethod
    def _kill(proc: subprocess.Popen, grace: float | None = None) -> None:
        """In the background: SIGTERM the process group, then SIGKILL it if still alive after `grace` s."""
        if proc.poll() is not None and not pgid_alive(proc.pid):
            return
        # start_new_session: the group id is the leader's pid, even after the leader has exited
        threading.Thread(target=kill_group, args=(proc.pid, grace), daemon=True).start()

    def _worker(self) -> None:
        while True:
            with self.cond:
                while not self.queue and not self._stop:
                    self.cond.wait()
                if self._stop:
                    return
                name = self.queue.popleft()
                job = self.jobs.get(name)
                if not job or job.cancel_requested or job.state != "queued":
                    continue
                self.current = job
            try:
                self._run(job)
            except Exception as e:  # never let one job kill the worker
                self._set_state(job, "failed", f"{type(e).__name__}: {e}")
            finally:
                job.proc = job.pgid = None
                self.current = None

    def _run(self, job: Job) -> None:
        log = self.log(job.name)
        self.rates = progress.history_rates(self.work)
        n_images, steps, map_total = self.units(job)
        done = self._stages_done(job)
        todo = [p for p in self.plan(job) if not (all(s in done for s in p.stages) or p.skip(job))]
        for phase in todo:  # a phase that runs again redoes all of its stages
            for s in phase.stages:
                done.discard(s)
                (job.dir / "stages" / f"{s}.done").unlink(missing_ok=True)
        first = next((s for s in progress.STAGES if s not in done), progress.STAGES[-1])
        tracker = progress.Tracker(n_images, map_total, start_stage=first)
        job.frac = {s: 1.0 for s in done}
        job.started, job.attempts = job.started or time.time(), job.attempts + 1
        self._set_state(job, "running")
        ctx = _Ctx(self, job, log, tracker, n_images, steps)
        job.interrupted = False
        try:
            for phase in todo:
                if job.cancel_requested or job.interrupted or self._stop:
                    break
                log.emit({"type": "phase", "phase": phase.name})
                ctx.emit(tracker.begin(phase.stages[0]))
                ok, err = ctx.run_phase(phase)
                if job.cancel_requested or ((job.interrupted or self._stop) and not ok):
                    break
                if ok:
                    ctx.emit(tracker.finish(phase.stages[-1]))
                elif phase.required:
                    self._set_state(job, "failed", err)
                    return
                else:
                    log.emit({"type": "log", "line": f"(optional step {phase.name} failed: {err})"})
                    ctx.emit(tracker.finish(phase.stages[-1]))
        finally:
            ctx.close()
        job.pgid = None
        if job.cancel_requested:
            self._set_state(job, "cancelled")
        elif job.interrupted or self._stop:
            self._set_state(job, "interrupted", INTERRUPTED_MSG if self._stop else _SIGTERM_MSG)
        else:
            job.stage = progress.STAGES[-1]
            self._set_state(job, "done")

    def _stages_done(self, job: Job) -> set[str]:
        d = job.dir / "stages"
        return {p.stem for p in d.glob("*.done")} if d.is_dir() else set()


class _Ctx:
    """Per-attempt state: turns tracker events into job fields, events.jsonl lines and stage markers."""

    def __init__(self, runner: Runner, job: Job, log: EventLog, tracker: progress.Tracker, n_images: int, steps: int):
        self.r, self.job, self.log, self.tracker = runner, job, log, tracker
        self.n_images, self.steps = n_images, steps
        self._last_progress = 0.0
        self._raw = open(job.dir / "studio.log", "a", buffering=1)  # noqa: SIM115 (closed by close())

    def emit(self, events: list[dict]) -> None:
        job, now = self.job, time.monotonic()
        for ev in events:
            if ev["type"] == "stage":
                job.stage, job.stage_eta = ev["stage"], None
                if ev["status"] == "done":
                    job.frac[ev["stage"]] = 1.0
                    d = job.dir / "stages"
                    d.mkdir(exist_ok=True)
                    n = progress.units(ev["stage"], self.n_images, self.steps)
                    (d / f"{ev['stage']}.done").write_text(json.dumps({"seconds": ev["seconds"], "units": n}))
                else:
                    job.frac.setdefault(ev["stage"], 0.0)
                job.eta = progress.eta_total(self.tracker, self.r.rates, self.n_images, self.steps)
                self.log.emit({**ev, "eta": job.eta})
                self._last_progress = 0.0
            elif ev["type"] == "progress":
                job.frac[ev["stage"]] = ev["frac"]
                job.stage_eta = ev["stage_eta"]
                if now - self._last_progress >= self.r.progress_interval or ev["frac"] >= 1.0:
                    self._last_progress = now
                    job.eta = progress.eta_total(self.tracker, self.r.rates, self.n_images, self.steps)
                    self.log.emit({**ev, "eta": job.eta})

    def line(self, text: str) -> None:
        self._raw.write(text + "\n")
        clean = progress.strip_ansi(text).strip()
        if not clean:
            return
        events = self.tracker.feed(clean)
        if not self.tracker.is_progress_line(clean):  # progress redraws go to the bars, not the log
            self.log.emit({"type": "log", "line": clean[:500]})
        self.emit(events)

    def tick(self) -> None:
        """Called about every 0.5 s while a phase runs: file-count progress for quiet stages, ETA refresh."""
        stage = self.tracker.stage
        if self.r.probe and stage:
            hit = self.r.probe(self.job, stage)
            if hit:
                self.emit(self.tracker.progress(stage, *hit))
        now = time.monotonic()
        if now - self._last_progress >= 2.0 and stage:
            self._last_progress = now
            self.job.eta = progress.eta_total(self.tracker, self.r.rates, self.n_images, self.steps)
            self.log.emit({"type": "eta", "stage": stage, "eta": self.job.eta})

    def close(self) -> None:
        self._raw.close()

    def _call(self, fn: Callable, phase: Phase) -> tuple[bool, str | None]:
        try:
            fn(self.job, self.line)
            return True, None
        except Exception as e:  # a setup error fails the job with its message, not a traceback
            self.line(f"error: {e}")
            return False, f"{phase.name}: {e}"

    def run_phase(self, phase: Phase) -> tuple[bool, str | None]:
        if phase.before:
            ok, err = self._call(phase.before, phase)
            if not ok or self.job.cancel_requested:
                return ok, err
        if callable(phase.run):
            return self._call(phase.run, phase)
        return self._exec(phase)

    def _exec(self, phase: Phase) -> tuple[bool, str | None]:
        master, slave = pty.openpty()
        _set_winsize(slave, 50, 200)
        env = {**os.environ, **(self.r.env or {}), "PYTHONUNBUFFERED": "1"}
        try:
            proc = subprocess.Popen(
                phase.run,
                stdin=subprocess.DEVNULL,
                stdout=slave,
                stderr=slave,
                cwd=self.r.cwd,
                env=env,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as e:
            os.close(master)
            os.close(slave)
            return False, f"{phase.name}: could not start {phase.run[0]}: {e}"
        os.close(slave)
        self.job.proc, self.job.pgid = proc, proc.pid  # start_new_session: pgid == pid
        self.job.save()  # so a restarted studio can tell whether this build outlived it
        if self.job.cancel_requested or self.r.stopping:  # cancel/stop raced with the start
            Runner._kill(proc)
        tail: list[str] = []
        buf = ""
        try:
            while True:
                ready, _, _ = select.select([master], [], [], 0.5)
                if ready:
                    try:
                        data = os.read(master, 65536)
                    except OSError:  # EIO: every writer closed the PTY (Linux)
                        data = b""
                    if not data:
                        break
                    lines, buf = progress.split_lines(buf + data.decode("utf-8", "replace"))
                    for ln in lines:
                        self.line(ln)
                        if ln.strip():
                            tail = (tail + [progress.strip_ansi(ln).strip()])[-5:]
                elif proc.poll() is not None:
                    break  # exited, and nothing left to read (or a detached grandchild holds the PTY)
                else:
                    self.tick()
        finally:
            os.close(master)
        if buf.strip():
            self.line(buf)
        rc = proc.wait()
        if rc == 0:
            return True, None
        if self.job.cancel_requested:
            return False, "cancelled"
        if self.r.stopping or rc in (-signal.SIGTERM, 128 + signal.SIGTERM):
            self.job.interrupted = True  # a graceful stop, not a crash: resumable
            return False, "interrupted"
        last = next((t for t in reversed(tail) if t and not self.tracker.is_progress_line(t)), "")
        return False, f"{phase.name} exited with code {rc}" + (f": {last}" if last else "")


_SIGTERM_MSG = "the build was stopped by SIGTERM; resume to continue"


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    try:
        import fcntl
        import struct
        import termios

        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except (ImportError, OSError):
        pass
