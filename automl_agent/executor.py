from __future__ import annotations

import math
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import pandas as pd
import psutil

from .contracts import ExecutionPolicy
from .util import atomic_json, file_hash, read_json, redact
from .diagnostics import RotatingOutput, snapshot, emit, artifact


STREAM_STAGING_DEFAULT = False
STREAM_STAGING_CELLS = 5_000_000


def stage_frame(path: Path, frame: pd.DataFrame, *, deadline=None, memory_mb=None, chunk_rows=128, streaming=None):
    if streaming is None:
        streaming = STREAM_STAGING_DEFAULT or frame.size > STREAM_STAGING_CELLS
    if not streaming:
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Frame staging deadline exhausted")
        data = frame.astype(object).where(frame.notna(), None)
        atomic_json(path, {"columns": list(frame), "index": list(map(str, frame.index)), "data": data.values.tolist(),
            "numeric": [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]})
        return
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

    def check():
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Frame staging deadline exhausted")
        if memory_mb is not None and psutil.Process().memory_info().rss > memory_mb * 1024**2:
            raise MemoryError("Frame staging memory limit exhausted")

    check()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            def write(value):
                json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write('{"columns":')
            write(list(frame))
            stream.write(',"numeric":')
            write([name for name in frame if pd.api.types.is_numeric_dtype(frame[name])])
            stream.write(',"index":[')
            for start in range(0, len(frame), chunk_rows):
                check()
                if start:
                    stream.write(',')
                chunk = json.dumps(list(map(str, frame.index[start:start + chunk_rows])), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                stream.write(chunk[1:-1])
            stream.write('],"data":[')
            step = max(1, min(chunk_rows, 16384 // max(1, len(frame.columns))))
            for start in range(0, len(frame), step):
                check()
                values = frame.iloc[start:start + step]
                data = values.astype(object).where(values.notna(), None)
                chunk = json.dumps(data.values.tolist(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                if start:
                    stream.write(',')
                stream.write(chunk[1:-1])
            stream.write(']}\n')
            stream.flush()
            os.fsync(stream.fileno())
        check()
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def terminate_tree(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


class Executor:
    def __init__(self, cpu, memory_mb, deadline, journal):
        physical = psutil.virtual_memory().total / 1024**2 * .85
        limits = [memory_mb, physical]
        for path in (Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')):
            try:
                value = path.read_text().strip()
                if value.isdigit():limits.append(int(value) / 1024**2 * .85)
            except OSError:pass
        cpu_limits = [cpu]
        if hasattr(os, 'sched_getaffinity'):
            cpu_limits.append(len(os.sched_getaffinity(0)))
        try:
            quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
            if quota != 'max':cpu_limits.append(max(1, int(quota) // int(period)))
        except (OSError, ValueError):
            try:
                quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
                period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
                if quota > 0:cpu_limits.append(max(1, quota // period))
            except (OSError, ValueError):pass
        self.cpu, self.memory_mb, self.deadline, self.journal = max(1, min(cpu_limits)), min(limits), deadline, journal

    def run(self, directory: Path, request: dict, seconds: float):
        directory.mkdir(parents=True, exist_ok=True)
        if "context" in request:
            request["context"]["worker_overhead_reserve"] = min(10, max(.1, seconds * .02))
            request["context"]["budget_seconds"] = min(request["context"].get("budget_seconds", seconds),
                max(.1, seconds - request["context"]["worker_overhead_reserve"]))
            request["context"]["deadline"] = min(self.deadline, time.time() + seconds)
        atomic_json(directory / "request.json", request)
        environment = {
            "PATH": os.defpath, "LANG": "en_US.UTF-8", "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
            "HOME": str(directory), "TMPDIR": str(directory / "tmp"), "XDG_CACHE_HOME": str(directory / "cache"),
            "MPLCONFIGDIR": str(directory / "matplotlib"), "JOBLIB_TEMP_FOLDER": str(directory / "tmp"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled",
            "AGENT_WORKER_CPUS": str(self.cpu),
            **{name: str(self.cpu) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        }
        (directory / "tmp").mkdir(exist_ok=True)
        end = min(self.deadline, time.time() + seconds)
        started, peak, reason = time.monotonic(), 0.0, None
        resource_reason = None
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen([sys.executable, "-I", str(Path(__file__).with_name("worker_entry.py")), str(directory / "request.json")],
                cwd=directory, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            output = RotatingOutput(process.stdout, directory)
            creation = psutil.Process(process.pid).create_time()
            receipt = {"pid": process.pid, "create_time": creation, "attempt": str(directory), "deadline": end}
            atomic_json(directory / "process.json", receipt)
            self.journal.put("active_worker", receipt)
            emit('worker_started', directory=str(directory), request=request, process=receipt)
            captured_trials = {}
            last_diagnostic_poll = 0
            try:
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        try:
                            tree = [parent, *parent.children(recursive=True)]
                        except (psutil.AccessDenied, PermissionError):
                            tree = [parent]
                            if not self.journal.get("process_tree_monitor_limit"):
                                self.journal.put("process_tree_monitor_limit", "Host disallows enumerating child processes; monitor worker RSS, worker process creation disabled")
                        memory = sum(p.memory_info().rss for p in tree if p.is_running()) / 1024**2
                        peak = max(peak, memory)
                        total = memory + psutil.Process().memory_info().rss / 1024**2
                        if total > self.memory_mb or psutil.virtual_memory().available < 256 * 1024**2:
                            resource_reason = resource_reason or "memory_limit"
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    if time.time() >= end:
                        resource_reason = resource_reason or "timeout"
                    if shutil.disk_usage(directory).free < 512 * 1024**2:
                        resource_reason = resource_reason or "disk_limit"
                    if resource_reason:
                        # A true resource stop takes precedence over a log pipe
                        # failure, and remains visible to the caller.
                        reason = resource_reason
                    elif output.error:
                        reason = "log_limit" if output.error == 'log_limit' else 'error'
                    if time.monotonic()-last_diagnostic_poll >= 1:
                        for history in [*directory.glob('diagnostic_trials.jsonl'), *directory.glob('flaml_*.jsonl')]:
                            stamp = (history.stat().st_size, history.stat().st_mtime_ns)
                            if captured_trials.get(history.name) != stamp:
                                artifact(history)
                                captured_trials[history.name] = stamp
                        last_diagnostic_poll = time.monotonic()
                    if reason:
                        terminate_tree(process)
                        break
                    time.sleep(0.05)
            except BaseException:
                terminate_tree(process)
                raise
            finally:
                # A library child may still hold the pipe; close the whole group
                # before draining so a verified result is not lost to a log error.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(0.05)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                output.finish()
                if output.error and reason is None:
                    reason = 'log_limit' if output.error == 'log_limit' else 'error'
                atomic_json(directory/'diagnostic_process_result.json', {'exit_code':process.returncode,
                    'signal':-process.returncode if process.returncode is not None and process.returncode<0 else None,
                    'reason':reason, 'log_error':output.error, 'log_bytes':output.total,
                    'peak_memory_mb':peak,'elapsed_seconds':time.monotonic()-started})
                emit('worker_finished', directory=str(directory), exit_code=process.returncode,
                    signal=-process.returncode if process.returncode is not None and process.returncode < 0 else None,
                    reason=reason, log_error=output.error, log_bytes=output.total, peak_memory_mb=peak,
                    seconds=time.monotonic()-started)
                snapshot(directory)
                self.journal.put("active_worker", None)
        path = directory / "worker_result.json"
        if reason in {"error", "log_limit"} and path.is_file():
            try:
                recovered = read_json(path)
                valid = recovered.get("status") == "completed" and all(
                    (directory / relative).resolve().is_relative_to(directory.resolve())
                    and (directory / relative).is_file()
                    and file_hash(directory / relative) == sha
                    for relative, sha in (recovered.get("hashes") or {}).items())
            except (OSError, ValueError):
                valid = False
            if valid:
                self.journal.event("verified_result_over_log_error", {"directory": str(directory), "reason": reason})
                reason = None
        if reason:
            return {"status": reason, "error": "External resource monitor stopped the candidate", "seconds": time.monotonic() - started, "peak_memory_mb": peak,
                "diagnostic_code":reason,"diagnostic_artifact":str(directory/'diagnostic_process_result.json')}
        if not path.is_file():
            tail = (directory / "worker.log").read_text(encoding="utf-8", errors="replace")[-5000:]
            return {"status": "error", "error": f"Worker exited {process.returncode} without a verified result", "log": redact(tail),
                "seconds": time.monotonic() - started, "peak_memory_mb": peak}
        result = read_json(path)
        for relative, expected in result.get("hashes", {}).items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file() or file_hash(path) != expected:
                return {"status": "error", "error": "Incomplete or corrupt worker artifacts", "seconds": time.monotonic() - started, "peak_memory_mb": peak}
        result["peak_memory_mb"] = peak
        result["wall_seconds"] = time.monotonic() - started
        return result

    def recover_worker(self):
        receipt = self.journal.get("active_worker")
        if not receipt:
            return
        try:
            process = psutil.Process(receipt["pid"])
            if abs(process.create_time() - receipt["create_time"]) < 0.01 and any(arg.startswith(receipt["attempt"] + "/") for arg in process.cmdline()):
                # Only a verified process belonging to this run may be terminated.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=1)
                except psutil.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
        self.journal.put("active_worker", None)

    def probe(self, workdir, source_train, controller, output, credentials):
        protected = workdir / "protected_probe.txt"
        protected.write_text("probe", encoding="utf-8")
        directory = workdir / "policy_probe"
        request = {"mode": "probe", "probe_paths": {"source_train": str(source_train), "other_labels": str(protected), "credentials_file": str(credentials)},
            "write_probe_paths": {"controller_write": str(controller), "output_write": str(output if output.exists() else protected)}}
        result = self.run(directory, request, min(15, max(1, self.deadline - time.time())))
        return ExecutionPolicy(writable_directories=[str(workdir / "attempts")], tests=result,
            limitations=["Python import and I/O checks are defense in depth, not OS isolation",
                "Native extensions and deliberately hostile Python are outside the limited-trust guarantee"])
