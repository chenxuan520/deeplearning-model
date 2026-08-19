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
import struct
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
TRAIN_LOG = RUNTIME / "train.log"
MONITOR_LOG = RUNTIME / "plateau_monitor.log"
STATE_PATH = RUNTIME / "plateau_state.json"
REQUEST_PATH = RUNTIME / "PLATEAU_CHECK_REQUEST"
PAUSED_PATH = RUNTIME / "PLATEAU_PAUSED"
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


def parse_gauntlet(output: str, expected_levels):
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
    row_levels = [row[0] for row in rows]
    passed = row_levels == list(expected_levels) and all(
        black_wins == 10 and black_losses == 0 and black_draws == 0
        and white_wins >= 8
        for _, black_wins, black_losses, black_draws,
        white_wins, white_losses, white_draws in rows
    )
    return passed, rows


def run_gauntlet_stage(candidate: Path, index: int, levels, label: str):
    cmd = [
        "taskset", "-c", "56-63", "./bin/alphazero", "gauntlet",
        "--model", str(candidate),
        "--levels", ",".join(map(str, levels)),
        "--games", "10",
        "--workers", "8",
        "--sims", "600",
        "--seed", str(9000 + index),
    ]
    write_log(
        f"full-spectrum window {index}/3 stage={label} start: {' '.join(cmd)}"
    )
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=GAUNTLET_TIMEOUT_SEC,
        check=False,
    )
    output_path = RUNTIME / (
        f"plateau_gauntlet_{candidate.stem}_{index}_{label}.log"
    )
    output_path.write_text(result.stdout, encoding="utf-8")
    passed, rows = parse_gauntlet(result.stdout, levels)
    write_log(
        f"full-spectrum window {index}/3 stage={label} done: "
        f"rc={result.returncode} passed={passed} rows={rows}"
    )
    return result.returncode == 0 and passed


def run_full_spectrum_window(candidate: Path, index: int):
    # L7 is a 200-simulation MCTS opponent and dominates wall time.  Keep the
    # hard condition unchanged, but fail fast on levels 1-6 before paying the
    # L7 cost.  A passing window still covers every level 1-7 with the original
    # 10-games-per-color and 600-simulation model budget.
    if not run_gauntlet_stage(candidate, index, range(1, 7), "l1-l6"):
        write_log(
            f"full-spectrum window {index}/3 short-circuited before L7"
        )
        return False
    return run_gauntlet_stage(candidate, index, [7], "l7")


def save_state(state):
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def request_pause(trigger_iteration):
    request_id = uuid.uuid4().hex
    temporary = REQUEST_PATH.with_suffix(".tmp")
    temporary.write_text(f"{request_id} {trigger_iteration}\n", encoding="utf-8")
    os.replace(temporary, REQUEST_PATH)
    return request_id


def read_handshake(path):
    try:
        parts = path.read_text(encoding="utf-8").split()
        if len(parts) != 2:
            return None
        return parts[0], int(parts[1])
    except (OSError, ValueError):
        return None


def wait_for_paused_iteration(request_id, timeout_sec=2 * 60 * 60):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        paused = read_handshake(PAUSED_PATH)
        if paused is not None and paused[0] == request_id:
            return paused[1]
        request = read_handshake(REQUEST_PATH)
        if request is None or request[0] != request_id:
            return None
        time.sleep(10)
    return None


def handshake_is_current(request_id, paused_iteration=None):
    request = read_handshake(REQUEST_PATH)
    paused = read_handshake(PAUSED_PATH)
    return (request is not None and request[0] == request_id
            and paused is not None and paused[0] == request_id
            and (paused_iteration is None or paused[1] == paused_iteration))


def cancel_pause_request(request_id):
    request = read_handshake(REQUEST_PATH)
    if request is not None and request[0] == request_id:
        try:
            REQUEST_PATH.unlink()
        except FileNotFoundError:
            pass


def publish_plateau_reached(request_id, iteration):
    target = RUNTIME / "PLATEAU_REACHED"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(f"{request_id} {iteration}\n", encoding="utf-8")
    os.replace(temporary, target)


def checkpoint_generations():
    pointer = RUNTIME / "latest.current"
    if pointer.exists():
        try:
            parts = pointer.read_text(encoding="utf-8").split()
            if len(parts) != 3:
                return None
            values = [int(part) for part in parts]
            if values[0] < 0 or values[1] < -1 or values[2] < -1:
                return None
            return tuple(values)
        except (OSError, ValueError):
            return None
    if (RUNTIME / "latest.versioned").exists():
        return None
    # Legacy migration path. New trainers publish latest.current on their
    # first completed iteration; aliases are never authoritative afterward.
    path = RUNTIME / "latest.state"
    try:
        data = path.read_bytes()
        if len(data) < 4:
            return None
        return struct.unpack("i", data[:4])[0], -1, -1
    except OSError:
        return None


def checkpoint_model_path(generation):
    pointer = RUNTIME / "latest.current"
    if pointer.exists():
        return RUNTIME / f"checkpoint.latest.{generation}.net"
    return RUNTIME / "latest.net"


def evaluate_status():
    events = load_train_events()
    train_by_iter = {}
    promoted_iters = []
    completed_gate_iters = set()
    completed_iters = set()
    for event in events:
        if event.get("phase") == "train_done":
            train_by_iter[int(event["iter"])] = float(event["policy_loss"])
        if event.get("phase") == "gate" and event.get("result") == "promoted":
            promoted_iters.append(int(event["iter"]))
        if event.get("phase") == "gate" and (
                "challenger_wins" in event
                or event.get("result") in ("init_best", "promoted")):
            completed_gate_iters.add(int(event["iter"]))
        if event.get("phase") == "iteration_complete":
            completed_iters.add(int(event["iter"]))
    train_rows = sorted(
        (iteration, loss) for iteration, loss in train_by_iter.items()
        if iteration in completed_iters
    )
    if not train_rows:
        return None
    latest_iter = train_rows[-1][0]
    checkpoint_tuple = checkpoint_generations()
    pointer_best = checkpoint_tuple[1] if checkpoint_tuple is not None else -1
    last_promotion = max(promoted_iters + ([pointer_best] if pointer_best >= 0 else [])) if (promoted_iters or pointer_best >= 0) else 0
    no_promotion_rounds = sum(
        1 for iteration, _ in train_rows if iteration > last_promotion
    )
    plateau, plateau_detail = loss_plateau(train_rows)
    iteration_complete = latest_iter in completed_iters
    gate_complete = latest_iter in completed_gate_iters
    checkpoint_iter = checkpoint_tuple[0] if checkpoint_tuple is not None else None
    return {
        "latest_iter": latest_iter,
        "last_promotion_iter": last_promotion,
        "no_promotion_rounds": no_promotion_rounds,
        "condition_no_promotion": no_promotion_rounds >= 30,
        "condition_loss_plateau": plateau,
        "loss_detail": plateau_detail,
        "latest_policy_loss": train_rows[-1][1],
        "condition_iteration_complete": iteration_complete,
        "condition_gate_complete": gate_complete,
        "checkpoint_iter": checkpoint_iter,
        "checkpoint_best_generation": pointer_best,
        "condition_checkpoint_current": checkpoint_iter == latest_iter,
    }


def evaluate_paused_candidate(status, trigger_iteration):
    """Pause the trainer, evaluate one frozen candidate, always release pause.

    Returns True only after publishing a nonce-matched PLATEAU_REACHED marker.
    Every other return/exception removes this request so the trainer resumes.
    """
    request_id = request_pause(trigger_iteration)
    reached_published = False
    write_log(
        f"plateau pause requested id={request_id} "
        f"from trigger iter {trigger_iteration}"
    )
    try:
        paused_iter = wait_for_paused_iteration(request_id)
        if paused_iter is None:
            write_log("trainer did not acknowledge plateau pause in time")
            return False

        paused_status = evaluate_status()
        paused_valid = (
            paused_status is not None
            and paused_status["latest_iter"] == paused_iter
            and paused_status["condition_no_promotion"]
            and paused_status["condition_loss_plateau"]
            and paused_status["last_promotion_iter"]
                == status["last_promotion_iter"]
            and paused_status["condition_iteration_complete"]
            and paused_status["condition_checkpoint_current"]
        )
        if not paused_valid:
            write_log("conditions changed before trainer pause; resume training")
            return False

        latest = checkpoint_model_path(paused_iter)
        if not latest.exists():
            write_log(f"versioned checkpoint missing: {latest}")
            return False
        candidate = RUNTIME / f"plateau_candidate_iter{paused_iter}.net"
        shutil.copy2(latest, candidate)
        streak = 0
        for index in range(1, 4):
            try:
                passed = run_full_spectrum_window(candidate, index)
            except (subprocess.TimeoutExpired, OSError) as exc:
                write_log(f"full-spectrum window {index}/3 failed: {exc}")
                passed = False
            if not passed:
                break
            streak += 1

        fresh_status = evaluate_status()
        still_valid = (
            fresh_status is not None
            and fresh_status["latest_iter"] == paused_iter
            and fresh_status["condition_no_promotion"]
            and fresh_status["condition_loss_plateau"]
            and fresh_status["last_promotion_iter"]
                == status["last_promotion_iter"]
            and fresh_status["condition_iteration_complete"]
            and fresh_status["condition_checkpoint_current"]
            and handshake_is_current(request_id, paused_iter)
        )
        state = {**(fresh_status or status), "updated_at": now(),
                 "full_spectrum_streak": streak,
                 "candidate": str(candidate),
                 "candidate_iteration": paused_iter,
                 "conditions_rechecked_after_gauntlet": still_valid}
        save_state(state)
        if streak == 3 and still_valid:
            shutil.copy2(candidate, RUNTIME / "champion_plateau.net")
            publish_plateau_reached(request_id, paused_iter)
            reached_published = True
            write_log(
                "ALL THREE CONDITIONS PASSED; PLATEAU_REACHED written; "
                "paused trainer will exit cleanly"
            )
            return True
        if streak == 3 and not still_valid:
            write_log(
                "three gauntlet windows passed but training conditions "
                "changed during evaluation; trainer remains alive"
            )
        return False
    except Exception as exc:  # fail open: training must resume
        write_log(f"plateau evaluation aborted safely: {type(exc).__name__}: {exc}")
        return False
    finally:
        if not reached_published:
            cancel_pause_request(request_id)


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
            # train_done is logged before latest checkpoint and gate.  Never
            # freeze/stop on that intermediate state: wait until the exact
            # iteration is durable and its most recent scheduled gate ended.
            # Evaluate only immutable per-gate snapshots after the trainer's
            # durable iteration_complete marker. This avoids copying a mutable
            # latest.net while the next iteration writes it.
            if (not status["condition_iteration_complete"]
                    or not status["condition_gate_complete"]):
                time.sleep(CHECK_INTERVAL_SEC)
                continue
            evaluated_iters.add(latest_iter)
            if evaluate_paused_candidate(status, latest_iter):
                return

        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
