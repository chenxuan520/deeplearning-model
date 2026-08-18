#!/usr/bin/env python3
"""Conservative unattended stop judge for the long AlphaZero grind.

The trainer is allowed to stop only when all three conditions hold on one
frozen candidate:
  1. no best-model promotion for at least 30 completed iterations;
  2. policy loss has no meaningful downward trend for 10 rolling windows;
  3. three consecutive 10-game-per-color full gauntlets have black 100%
     and white >=80% on every level 1-7.

The monitor itself never changes training data or model parameters.  It only
reads logs/checkpoints, launches evaluation on CPUs 56-63, and stops the exact
`alphazero train` process after all conditions pass.
"""

import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
TRAIN_LOG = RUNTIME / "train.log"
MONITOR_LOG = RUNTIME / "plateau_monitor.log"
STATE_PATH = RUNTIME / "plateau_state.json"
CHECK_INTERVAL_SEC = 300
GAUNTLET_TIMEOUT_SEC = 6 * 60 * 60


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_log(message: str) -> None:
    line = f"[{now()}] {message}"
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_train_events():
    events = []
    if not TRAIN_LOG.exists():
        return events
    with TRAIN_LOG.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def linear_slope(values):
    n = len(values)
    if n < 2:
        return float("-inf")
    xm = (n - 1) / 2.0
    ym = sum(values) / n
    denom = sum((i - xm) ** 2 for i in range(n))
    return sum((i - xm) * (v - ym) for i, v in enumerate(values)) / denom


def loss_plateau(train_rows):
    """Require ten consecutive 5-step rolling means with no real descent.

    15 completed iterations provide ten transitions between rolling means.
    A decrease larger than 0.005 in any transition means learning continues.
    The overall 15-point slope must also be >= -0.001 loss/iteration.
    This intentionally errs toward training longer, never stopping early.
    """
    if len(train_rows) < 15:
        return False, {"reason": "need_15_losses"}
    losses = [row[1] for row in train_rows[-15:]]
    means = [sum(losses[i : i + 5]) / 5.0 for i in range(11)]
    deltas = [means[i + 1] - means[i] for i in range(10)]
    slope = linear_slope(losses)
    ok = all(delta >= -0.005 for delta in deltas) and slope >= -0.001
    return ok, {
        "last15": [round(v, 5) for v in losses],
        "rolling5_first": round(means[0], 5),
        "rolling5_last": round(means[-1], 5),
        "min_delta": round(min(deltas), 6),
        "slope": round(slope, 6),
    }


def exact_train_pids():
    pids = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline = entry / "cmdline"
        try:
            cmd = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if "./bin/alphazero train " in cmd:
            pids.append(int(entry.name))
    return pids


def parse_gauntlet(output: str):
    rows = []
    pat = re.compile(
        r"^\s*(\d+)\s*\|\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*"
        r"\([^)]*\)\s*\|\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*"
        r"\([^)]*\)\s*$"
    )
    for line in output.splitlines():
        m = pat.match(line)
        if m:
            rows.append(tuple(map(int, m.groups())))
    passed = len(rows) == 7 and all(
        black_wins == 10 and black_losses == 0 and black_draws == 0
        and white_wins >= 8
        for _, black_wins, black_losses, black_draws,
        white_wins, white_losses, white_draws in rows
    )
    return passed, rows


def run_gauntlet(candidate: Path, index: int):
    cmd = [
        "taskset", "-c", "56-63", "./bin/alphazero", "gauntlet",
        "--model", str(candidate),
        "--levels", "1,2,3,4,5,6,7",
        "--games", "10",
        "--workers", "8",
        "--sims", "600",
        "--seed", str(9000 + index),
    ]
    write_log(f"full-spectrum window {index}/3 start: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=GAUNTLET_TIMEOUT_SEC,
        check=False,
    )
    output_path = RUNTIME / f"plateau_gauntlet_{candidate.stem}_{index}.log"
    output_path.write_text(result.stdout, encoding="utf-8")
    passed, rows = parse_gauntlet(result.stdout)
    write_log(
        f"full-spectrum window {index}/3 done: rc={result.returncode} "
        f"passed={passed} rows={rows}"
    )
    return result.returncode == 0 and passed


def save_state(state):
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def evaluate_status():
    events = load_train_events()
    train_by_iter = {}
    promoted_iters = []
    for event in events:
        if event.get("phase") == "train_done":
            train_by_iter[int(event["iter"])] = float(event["policy_loss"])
        if event.get("phase") == "gate" and event.get("result") == "promoted":
            promoted_iters.append(int(event["iter"]))
    train_rows = sorted(train_by_iter.items())
    if not train_rows:
        return None
    latest_iter = train_rows[-1][0]
    last_promotion = max(promoted_iters) if promoted_iters else 0
    no_promotion_rounds = latest_iter - last_promotion
    plateau, plateau_detail = loss_plateau(train_rows)
    return {
        "latest_iter": latest_iter,
        "last_promotion_iter": last_promotion,
        "no_promotion_rounds": no_promotion_rounds,
        "condition_no_promotion": no_promotion_rounds >= 30,
        "condition_loss_plateau": plateau,
        "loss_detail": plateau_detail,
        "latest_policy_loss": train_rows[-1][1],
    }


def main():
    write_log("plateau monitor started")
    last_logged_iter = -1
    evaluated_iters = set()
    while True:
        status = evaluate_status()
        if status is None:
            write_log("no train_done events yet")
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        latest_iter = status["latest_iter"]
        if latest_iter != last_logged_iter:
            write_log(
                "status " + json.dumps(status, ensure_ascii=False, sort_keys=True)
            )
            save_state({**status, "updated_at": now(), "full_spectrum_streak": 0})
            last_logged_iter = latest_iter

        first_two = (
            status["condition_no_promotion"]
            and status["condition_loss_plateau"]
        )
        if first_two and latest_iter not in evaluated_iters:
            evaluated_iters.add(latest_iter)
            latest = RUNTIME / "latest.net"
            candidate = RUNTIME / f"plateau_candidate_iter{latest_iter}.net"
            # Give the trainer's checkpoint rename time to settle.
            time.sleep(10)
            shutil.copy2(latest, candidate)
            streak = 0
            for index in range(1, 4):
                try:
                    passed = run_gauntlet(candidate, index)
                except (subprocess.TimeoutExpired, OSError) as exc:
                    write_log(f"full-spectrum window {index}/3 failed: {exc}")
                    passed = False
                if not passed:
                    break
                streak += 1
            state = {**status, "updated_at": now(), "full_spectrum_streak": streak,
                     "candidate": str(candidate)}
            save_state(state)
            if streak == 3:
                shutil.copy2(candidate, RUNTIME / "champion_plateau.net")
                (RUNTIME / "PLATEAU_REACHED").write_text(
                    json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                write_log("ALL THREE CONDITIONS PASSED; stopping exact trainer processes")
                for pid in exact_train_pids():
                    os.kill(pid, signal.SIGTERM)
                return

        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
