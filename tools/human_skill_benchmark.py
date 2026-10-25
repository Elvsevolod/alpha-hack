"""Equal-budget, archive-bound Qwen comparison. Development tooling only."""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnostic_3h as d
from tools.goal_quality_run import SharedPeriodGuard, decode_response, guest_report
from tools.visual_ab_benchmark import Allowance, Handler, response_deadline, finish_job


def validate_pair(base, skill):
    def files(path):
        with zipfile.ZipFile(path) as archive:
            tree = ast.parse(archive.read("agent_bundle.py"))
            embedded = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "FILES" for target in node.targets))
            outer = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                if name not in {"agent_bundle.py", "ml_knowledge_bundle.py"} and not name.startswith("mlkb_data_")}
            return embedded, outer
    a, outer_a = files(base)
    b, outer_b = files(skill)
    changed = {name for name in a.keys() | b.keys() if a.get(name) != b.get(name)}
    assert changed == {"automl_agent/human_practice.py", "automl_agent/research.py", "ml_knowledge/context.py"}, changed
    assert {name for name in outer_a.keys() | outer_b.keys() if outer_a.get(name) != outer_b.get(name)} == {"mlkb_context.py"}
    d.verify_archive(base)
    d.verify_archive(skill)
    return sorted(changed)


def run(args):
    from dotenv import load_dotenv
    changed = validate_pair(args.base, args.skill)
    args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as cleanup:
        load_dotenv(ROOT / ".env", override=False)
        key = d.limits.load_key(ROOT / ".env")
        before = d.limits.snapshot(d.limits.fetch_info(key), key, "before")
        d.write(args.output / "api_limits_before.json", before)
        if before["budget_exhausted"] or before["limits"].get("blocked") or before["remaining_budget"] is None or before["remaining_budget"] < args.cap:
            raise RuntimeError("Known unblocked account remainder below the newly authorized allowance")
        accounting = Allowance(args.ledger or args.output / "budget_ledger.json", before, args.cap)
        accounting.cap = accounting.state["initial_spend"] + args.cap
        accounting.arm_limit = (args.cap - accounting.reserve) / (4 * len(args.seeds))
        cleanup.callback(accounting.release)
        token = secrets.token_urlsafe(32)
        d.register_secret(key)
        d.register_secret(token)
        info = json.loads(d.docker("info", "--format", "{{json .}}").stdout)
        assert info["NCPU"] >= 2 * args.cpu and info["MemTotal"] >= (2 * args.memory_gib + 1) * d.GIB
        assert shutil.disk_usage(args.output).free >= 6 * d.GIB
        assert d.docker("network", "inspect", args.network, "--format", "{{.Internal}}").stdout.strip() == "true"
        image = d.docker("image", "inspect", d.IMAGE, "--format", "{{.Id}}").stdout.strip()
        jobs, waves = {}, []
        for repetition, seed in enumerate(args.seeds):
            for kind, source in (("regression", "Регрессия"), ("classification", "Многоклассовая классификация")):
                wave = []
                for version, archive in (("base", args.base), ("skill", args.skill)):
                    arm = f"{kind}_{seed}_{version}"
                    directory = args.output / arm / "task"
                    directory.mkdir(parents=True)
                    hashes = {}
                    for name in ("task.md", "train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv"):
                        original = ROOT / "data_for_participants" / source / name
                        shutil.copyfile(original, directory / name)
                        hashes[name] = d.digest(original)
                    with zipfile.ZipFile(archive) as package:
                        package.extractall(directory)
                    job = {"kind": arm, "task_kind": kind, "version": version, "seed": seed,
                        "directory": str(directory), "container": "alpha-skill-" + args.output.name[-8:].replace("_", "-") + "-" + arm.replace("_", "-"),
                        "input_hashes": hashes, "archive_sha256": d.digest(archive),
                        "started": False, "finished": False, "deadline_mono": None}
                    jobs[arm] = job
                    wave.append(job)
                waves.append(wave if (repetition + (kind == "classification")) % 2 == 0 else list(reversed(wave)))
        fetch = lambda: d.limits.snapshot(d.limits.fetch_info(key), key, "status")
        guard = SharedPeriodGuard(before, fetch, args.output, budget_cap=args.cap, accounting=accounting)
        guard.budget, guard.threshold = args.cap, args.cap - accounting.reserve
        def deadline(body, job):
            end, reserve, source = response_deadline(body, job)
            return min(end, guard.period_end), reserve, source + "+api_period"
        server = d.Gateway(("0.0.0.0", 0), guard, jobs, os.environ["OPENAI_BASE_URL"].rstrip("/"), key,
            deadline_reader=deadline, client_key=token, response_decoder=decode_response, accounting=accounting)
        server.RequestHandlerClass = Handler
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        relay = "alpha-skill-relay-" + args.output.name[-8:].replace("_", "-")
        d.docker("create", "--name", relay, "--network", args.network, "--cpus", ".25", "--memory", "192m",
            "-v", str(ROOT / "tools/visual_ab_relay.py") + ":/relay.py:ro", "--env", "GATEWAY_PORT=" + str(server.server_port),
            "--env", "ALLOWED_ARMS=" + ",".join(jobs),
            d.IMAGE, "python", "/relay.py")
        d.docker("network", "connect", "bridge", relay)
        d.docker("start", relay)
        def close_relay():
            d.docker("stop", "-t", "1", relay, check=False)
            d.docker("rm", relay, check=False)
        cleanup.callback(close_relay)
        done = threading.Event()
        def beat():
            while not done.is_set():
                for owner in ("observer", "gateway"):
                    d.write(args.output / (owner + "_heartbeat.json"), {"mono": time.monotonic(), "utc": d.utc()})
                done.wait(2)
        heartbeat = threading.Thread(target=beat, daemon=True)
        heartbeat.start()
        manifest = {"base_archive": str(args.base), "skill_archive": str(args.skill), "only_changed_sources": changed,
            "image_id": image, "network": args.network, "external_internet": False,
            "seconds_per_task": args.seconds, "cpu_per_task": args.cpu, "memory_gib_per_task": args.memory_gib,
            "max_calls": args.max_calls, "max_candidates": args.max_candidates, "research_focus_override": None,
            "model": "qwen/qwen3.8-27b", "api_cap": args.cap, "new_generation_stop_threshold": guard.threshold,
            "arm_request_admission_cost_limit": accounting.arm_limit, "inflight_reserve": accounting.reserve,
            "reserve_is_published_price_bound": False, "request_order": "FIFO serialized across concurrent paired arms",
            "accounting": "max(known_response_cost,account_counter_delta); close all on unknown cost",
            "seeds": args.seeds, "old_ledgers_reopened": False, "source_sha256": d.digest(Path(__file__)),
            "authorization": "Explicit instruction to compare v072 with skill; separate new allowance up to 120 units.",
            "shared_ledger": str(accounting.path), "relay_sha256": d.digest(ROOT / "tools/visual_ab_relay.py"),
            "quality_scope": "Previously viewed provided development tasks; matched closed controls per seed; short-budget comparison",
            "primary_endpoint": "Final selected model score on closed control, paired by task and seed",
            "promotion_rule": "No protocol failures; positive mean paired relative improvement with uncertainty excluding zero and no task mean regression over 1%; otherwise keep baseline; no universal superiority claim",
            "started_utc": d.utc(), "jobs": jobs}
        d.write(args.output / "MANIFEST.json", manifest)
        shutil.copyfile(__file__, args.output / "benchmark_source.py")
        d.write(args.output / "CONFIG.json", {"jobs": jobs})
        results, watchdog = [], None
        print(json.dumps({"status": "ready", "remaining_before": before["remaining_budget"], "api_cap": args.cap,
            "runs": len(jobs), "seconds_each": args.seconds, "arm_allowance": accounting.arm_limit}), flush=True)
        try:
            probe = d.docker("run", "--rm", "--network", args.network, "--cpus", ".25", "--memory", "192m",
                d.IMAGE, "python", "-c", "import urllib.request;print(urllib.request.urlopen('http://" + relay + ":18767/health',timeout=10).read().decode())")
            assert json.loads(probe.stdout)["alive"] is True
            probe_code = """import urllib.request,urllib.error,json,sys
for arm in sys.argv[1:]:
 try:
  urllib.request.urlopen(urllib.request.Request('http://RELAY:18767/'+arm+'/v1/chat/completions',data=b'{}',headers={'Content-Type':'application/json'}),timeout=10)
  raise RuntimeError('Unauthenticated model request was admitted')
 except urllib.error.HTTPError as error:
  body=json.loads(error.read())
  assert error.code==403 and body['error']['code']=='local_auth_required',(error.code,body)
print('All model routes reached the authenticated gateway without a paid call')
""".replace("RELAY", relay)
            d.docker("run", "--rm", "--network", args.network, "--cpus", ".25", "--memory", "192m",
                d.IMAGE, "python", "-c", probe_code, *jobs)
            d.write(args.output / "network_preflight.json", {"passed": True, "paid_requests": 0})
            with (args.output / "watchdog.log").open("w") as log:
                watchdog = subprocess.Popen([sys.executable, str(ROOT / "tools/diagnostic_3h.py"), "watchdog", "--output", str(args.output)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for wave in waves:
                if guard.closed:
                    break
                for job in wave:
                    environment = {**os.environ, "OPENAI_API_KEY": token,
                        "OPENAI_BASE_URL": f"http://{relay}:18767/{job['kind']}/v1"}
                    command = ["docker", "create", "--name", job["container"], "--network", args.network,
                        "--cpus", str(args.cpu), "--memory", str(args.memory_gib) + "g", "--memory-swap", str(args.memory_gib) + "g",
                        "-v", job["directory"] + ":/task", "-w", "/task", "--env", "PYTHONDONTWRITEBYTECODE=1",
                        "--env", "OPENAI_API_KEY", "--env", "OPENAI_BASE_URL", d.IMAGE,
                        "python", "solution.py", "task.md", "--time-limit", str(args.seconds), "--cpu", str(args.cpu),
                        "--memory-mb", str(args.memory_gib * 1024), "--seed", str(job["seed"]),
                        "--max-calls", str(args.max_calls), "--max-candidates", str(args.max_candidates), "--run-id", "diagnostic"]
                    created = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
                    if created.returncode:
                        raise RuntimeError("Container creation failed: " + created.stderr[:300])
                    job.update(started=True, entry_mono=time.monotonic(), entry_utc=d.utc())
                    job["deadline_mono"] = job["entry_mono"] + args.seconds + 30
                    d.write(args.output / "CONFIG.json", {"jobs": jobs})
                    d.docker("start", job["container"])
                while not all(job["finished"] for job in wave):
                    progress = {}
                    for job in wave:
                        if job["finished"]:
                            continue
                        state = json.loads(d.docker("inspect", "--format", "{{json .State}}", job["container"]).stdout)
                        if not state["Running"]:
                            job.update(finished=True, docker_state=state, finished_utc=d.utc(), finished_mono=time.monotonic())
                            logs = d.docker("logs", job["container"], check=False)
                            (args.output / job["kind"] / "container.log").write_text(logs.stdout + logs.stderr)
                            results.append(finish_job(job, args.output))
                            # Immutable ZIPs retain the exact corpus. Completed jobs
                            # need their journal/models, not another expanded copy.
                            cache = Path(job["directory"]) / ".ml_knowledge_runtime"
                            if cache.is_dir():
                                shutil.rmtree(cache)
                        else:
                            progress[job["kind"]] = guest_report(job)
                    d.write(args.output / "CONFIG.json", {"jobs": jobs})
                    d.write(args.output / "PROGRESS.json", {"utc": d.utc(), "jobs": progress,
                        "known_response_cost": accounting.state["reported_cost"], "gate_closed": guard.reason,
                        "arm_spend": {arm: accounting.arm_spend(arm) for arm in jobs}})
                    if watchdog.poll() is not None and not all(job["finished"] for job in wave):
                        raise RuntimeError("Independent watchdog exited")
                    time.sleep(5)
                d.write(args.output / "REPORT.json", {"results": results, "jobs": jobs, "gate_reason": guard.reason,
                    "all_finished": len(results) == len(jobs), "quality_advantage_established": False})
                print(json.dumps({"status": "pair_finished", "pair": wave[0]["task_kind"], "seed": wave[0]["seed"],
                    "known_cost": accounting.state["reported_cost"], "gate_reason": guard.reason}), flush=True)
        finally:
            for job in jobs.values():
                if job["started"] and not job["finished"]:
                    d.docker("kill", job["container"], check=False)
                if job["started"]:
                    d.docker("rm", job["container"], check=False)
            d.write(args.output / "RUN_FINISHED", {"utc": d.utc()})
            done.set()
            if watchdog and watchdog.poll() is None:
                watchdog.terminate()
                watchdog.wait(timeout=10)
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            heartbeat.join(timeout=5)
            after = d.limits.snapshot(d.limits.fetch_info(key), key, "after")
            after["comparison"] = d.limits.compare(before, after)
            d.write(args.output / "api_limits_after.json", after)
            print(json.dumps({"remaining_before": before["remaining_budget"], "remaining_after": after["remaining_budget"],
                "counter_increase": after["comparison"].get("increase"), "known_cost": accounting.state["reported_cost"]}), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--cap", type=float, default=120)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--memory-gib", type=float, default=3)
    parser.add_argument("--max-calls", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    args = parser.parse_args()
    args.base, args.skill, args.output = args.base.resolve(), args.skill.resolve(), args.output.resolve()
    args.network = "alpha-v010-llm"
    raise SystemExit(run(args))
