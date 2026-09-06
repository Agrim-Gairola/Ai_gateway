"""
run_queue.py
============
Cap-aware, idempotent, resumable job runner for the PCR experiment backlog.

Problem it solves: Groq (daily token cap), Gemini (500 req/day) and Mistral
(per-minute) rate limits mean a full test sweep can't finish in one sitting --
you hit a wall, wait ~a day, retry. This script turns the backlog into a queue:
each run does whatever isn't currently rate-limited, marks capped jobs
"deferred" (not failed), and a daily scheduled task re-runs it until the queue
is empty -- then removes its own scheduled task.

    python -m pcr_experiment.run_queue            # run one pass
    python -m pcr_experiment.run_queue --status   # just print the board
    python -m pcr_experiment.run_queue --reset-deferred   # deferred -> pending

Job list:   pcr_experiment/queue.json        (edit to add work)
State:      pcr_experiment/queue_state.json  (auto)
Log:        pcr_experiment/exports/queue.log
"""
import os
import re
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / "queue.json"
STATE = ROOT / "queue_state.json"
LOG = ROOT / "exports" / "queue.log"
PY = str((ROOT.parent / ".venv" / "Scripts" / "python.exe"))
JOB_TIMEOUT = 95 * 60          # seconds per job
MAX_ATTEMPTS = 4              # non-cap failures beyond this -> dead
PCR3_TAG = "allgroq_gptoss20_qwen36_qwen38"

CAP_MARKERS = [
    "tokens per day", "(TPD)", "requests per day", "RequestsPerDay",
    "PerDayPerProject", "per day (RPD)", "exceeded your current quota",
    "generate_content_free_tier_requests", "Rate limit reached",
    "rate_limit_exceeded", "resource_exhausted", "quota",
    "rate limit exceeded", '"type":"rate_limited"', "rate_limited",
    "429",  # any 5x-retry failure that reached here already survived transient
            # 429s inside call_model's own backoff -- if it still failed, the
            # limit is real (per-minute or per-day), so defer, don't burn attempts
]


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def log(msg):
    LOG.parent.mkdir(exist_ok=True)
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_json(p, default):
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_state(st):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


# --------------------------------------------------------------------------- #
# completion checks -- a job is "done" the moment its output exists, so a
# re-run after a crash costs nothing.
# --------------------------------------------------------------------------- #
def _ckpt_len(path):
    try:
        with open(path, encoding="utf-8") as f:
            return len(json.load(f).get("results", []))
    except Exception:
        return 0


def is_complete(job):
    k = job["kind"]
    if k == "pcr2":
        return _ckpt_len(ROOT / "checkpoints" / f"{job['pair']}_{job['benchmark']}.json") >= job["n"]
    if k == "pcr3":
        return _ckpt_len(ROOT / "checkpoints" / f"pcr3_{PCR3_TAG}_{job['benchmark']}.json") >= job["n"]
    if k == "solo_sweep":
        sweep = load_json(ROOT / "exports" / "solo_sweep.json", {})
        need = job["n"]
        for label, d in sweep.items():
            for bm in job["benchmarks"]:
                cell = d.get("cells", {}).get(f"{bm}_n{need}")
                if not (isinstance(cell, dict) and "accuracy" in cell):
                    return False
        # also: at least one candidate present (guard against empty cache)
        return bool(sweep)
    if k == "rebuild":
        return False  # always re-run; it's free and keeps exports fresh
    return False


def cmd_for(job):
    k = job["kind"]
    if k == "pcr2":
        return [PY, "-u", "-m", "pcr_experiment.pcr_runner",
                "--pair", job["pair"], "--benchmark", job["benchmark"], "--n", str(job["n"])]
    if k == "pcr3":
        return [PY, "-u", "-m", "pcr_experiment.pcr3_runner",
                "--benchmark", job["benchmark"], "--n", str(job["n"])]
    if k == "solo_sweep":
        return [PY, "-u", "-m", "pcr_experiment.solo_sweep",
                "--n", str(job["n"]), "--benchmarks", *job["benchmarks"], "--retry-errors"]
    if k == "rebuild":
        return None  # handled inline
    raise ValueError(k)


def job_id(job):
    k = job["kind"]
    if k == "pcr2":
        return f"pcr2:{job['pair']}:{job['benchmark']}"
    if k == "pcr3":
        return f"pcr3:{job['benchmark']}"
    if k == "solo_sweep":
        return f"solo_sweep:{'+'.join(job['benchmarks'])}:n{job['n']}"
    return k


def run_rebuild():
    for mod in ("pcr_experiment.ablation_study", "pcr_experiment.trade_off_figures",
                "pcr_experiment.make_figures"):
        args = [PY, "-m", mod] + (["--export"] if mod.endswith("ablation_study") else [])
        try:
            subprocess.run(args, cwd=str(ROOT.parent), timeout=1200,
                           capture_output=True, text=True)
        except Exception as e:
            log(f"  rebuild {mod} failed: {e}")


def run_job(job):
    """Returns 'done' | 'deferred' | 'error'."""
    cmd = cmd_for(job)
    if cmd is None:
        run_rebuild()
        return "done"
    try:
        p = subprocess.run(cmd, cwd=str(ROOT.parent), timeout=JOB_TIMEOUT,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {JOB_TIMEOUT}s")
        return "deferred"        # timeouts are usually backoff storms -> try later
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-4:])
    if is_complete(job):
        log(f"  ok (rc={p.returncode})")
        return "done"
    capped = any(m.lower() in out.lower() for m in CAP_MARKERS)
    failed_call = "call failed after" in out or p.returncode != 0
    if capped and failed_call:
        log(f"  DEFERRED (rate-limited). tail: {tail[:300]}")
        return "deferred"
    log(f"  ERROR (rc={p.returncode}). tail: {tail[:400]}")
    return "error"


def board(jobs, st):
    rows = []
    for j in jobs:
        s = st.get(job_id(j), {})
        rows.append((job_id(j), s.get("status", "pending"), s.get("attempts", 0)))
    w = max(len(r[0]) for r in rows) + 2
    print("\n" + "=" * (w + 24))
    print("QUEUE STATUS")
    print("=" * (w + 24))
    for jid, status, att in rows:
        print(f"  {jid:<{w}} {status:<10} {('attempts=' + str(att)) if att else ''}")
    from collections import Counter
    c = Counter(r[1] for r in rows)
    print("-" * (w + 24))
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    return c


def notify_done(summary):
    """Best-effort desktop notification when the whole queue finishes.
    Tries a BurntToast toast, then a WinRT toast, then a MessageBox popup."""
    title = "PCR queue finished"
    body = summary.replace('"', "'")
    ps_burnt = (f'Import-Module BurntToast -ErrorAction Stop; '
                f'New-BurntToastNotification -Text "{title}", "{body}"')
    ps_winrt = (
        '[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null; '
        '$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent('
        '[Windows.UI.Notifications.ToastTemplateType]::ToastText02); '
        f'$t.GetElementsByTagName("text")[0].AppendChild($t.CreateTextNode("{title}")) > $null; '
        f'$t.GetElementsByTagName("text")[1].AppendChild($t.CreateTextNode("{body}")) > $null; '
        '[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('
        '"Microsoft.WindowsPowerShell").Show([Windows.UI.Notifications.ToastNotification]::new($t))')
    ps_box = (f'Add-Type -AssemblyName System.Windows.Forms; '
              f'[System.Windows.Forms.MessageBox]::Show("{body}", "{title}") > $null')
    for ps in (ps_burnt, ps_winrt, ps_box):
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                log(f"  desktop notification sent ({ps.split(';')[0][:40]}...)")
                return
        except Exception:
            continue
    log("  (could not raise a desktop notification -- see queue_DONE.flag / QUEUE_SUMMARY.txt)")


def maybe_self_remove(all_settled, jobs, st):
    if not all_settled:
        return
    done = [job_id(j) for j in jobs if st.get(job_id(j), {}).get("status") == "done"]
    dead = [job_id(j) for j in jobs if st.get(job_id(j), {}).get("status") == "dead"]
    summary = (f"{len(done)} done, {len(dead)} dead. "
               + (f"Dead: {', '.join(dead)}. " if dead else "")
               + "Rebuilt: ablation_study.json, trade_off + family figures.")
    (ROOT / "queue_DONE.flag").write_text(now() + "\n" + summary + "\n", encoding="utf-8")
    (ROOT / "QUEUE_SUMMARY.txt").write_text(
        f"PCR test queue finished {now()}\n\n{summary}\n\n"
        f"done:\n  " + "\n  ".join(done) + "\n\n"
        f"dead (needs a look):\n  " + ("\n  ".join(dead) if dead else "(none)") + "\n\n"
        f"Next: re-read ABLATION_STUDY.md sections 4a / 7 / 8 -- the pending cells "
        f"(PCR-3 hard tier, pair10/pair11 head-to-head) are now filled.\n",
        encoding="utf-8")
    log("QUEUE EMPTY -- all jobs done or dead. " + summary)
    # regenerate analysis + paper-ready number dump (FINAL_NUMBERS.md)
    try:
        subprocess.run([PY, "-m", "pcr_experiment.finalize"],
                       cwd=str(ROOT.parent), timeout=2400, capture_output=True, text=True)
        log("  finalize.py done -> pcr_experiment/FINAL_NUMBERS.md")
    except Exception as e:
        log(f"  finalize.py failed: {e} -- run 'python -m pcr_experiment.finalize' by hand")
    notify_done(summary)
    # remove the recurring scheduled task if it exists (Windows)
    try:
        r = subprocess.run(["schtasks", "/delete", "/tn", "PCR_daily_queue", "/f"],
                           capture_output=True, text=True)
        log(f"  schtasks delete PCR_daily_queue -> rc={r.returncode} {r.stdout.strip()}{r.stderr.strip()}")
    except Exception as e:
        log(f"  (could not auto-remove scheduled task: {e}) -- delete PCR_daily_queue manually")


LOCK = ROOT / "queue.lock"
LOCK_STALE_SEC = 7 * 3600


def acquire_lock():
    # atomic create-or-fail; only a stale lock (>LOCK_STALE_SEC) may be stolen
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {now()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            age = time.time() - LOCK.stat().st_mtime
        except FileNotFoundError:
            return acquire_lock()  # race: it vanished, retry once
        if age < LOCK_STALE_SEC:
            log(f"another run_queue is active (lock {int(age)}s old) -- exiting")
            return False
        log(f"stale lock ({int(age)}s) -- taking over")
        LOCK.unlink(missing_ok=True)
        return acquire_lock()


def release_lock():
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="print the board and exit")
    ap.add_argument("--reset-deferred", action="store_true", help="deferred/error -> pending, then exit")
    a = ap.parse_args()

    jobs = load_json(QUEUE, [])
    if not jobs:
        print(f"No jobs in {QUEUE}. Nothing to do."); return
    st = load_json(STATE, {})

    if a.status:
        board(jobs, st); return

    if not (a.reset_deferred or acquire_lock()):
        return
    try:
        _main_pass(a, jobs, st)
    finally:
        release_lock()


def _main_pass(a, jobs, st):
    if a.reset_deferred:
        for jid, s in st.items():
            if s.get("status") in ("deferred", "error"):
                s["status"] = "pending"; s["attempts"] = 0
        save_state(st); print("deferred/error -> pending"); board(jobs, st); return

    log(f"=== queue pass start ({len(jobs)} jobs) ===")
    for job in jobs:
        jid = job_id(job)
        s = st.setdefault(jid, {"status": "pending", "attempts": 0})
        if s["status"] in ("done", "dead"):
            continue
        if job["kind"] != "rebuild" and is_complete(job):
            s["status"] = "done"; save_state(st); log(f"{jid}: already complete -> done"); continue
        log(f"{jid}: running (status was {s['status']}, attempts {s['attempts']})")
        res = run_job(job)
        if res == "done":
            s["status"] = "done"
        elif res == "deferred":
            s["status"] = "deferred"                     # attempts NOT incremented for cap waits
        else:
            s["attempts"] += 1
            s["status"] = "dead" if s["attempts"] >= MAX_ATTEMPTS else "error"
        s["last_run"] = now()
        save_state(st)
        time.sleep(5)

    # final rebuild so exports reflect whatever landed this pass
    run_rebuild()
    log("rebuilt exports/figures")

    c = board(jobs, st)
    settled = all(st.get(job_id(j), {}).get("status") in ("done", "dead")
                  for j in jobs if j["kind"] != "rebuild")
    maybe_self_remove(settled, jobs, st)
    log(f"=== queue pass done. {dict(c)} ===")


if __name__ == "__main__":
    main()
