#!/usr/bin/env python3
"""Scientific loss fitting/plotting for the long AlphaZero run.

Run with `.venv-analysis/bin/python tools/analyze_loss_scipy.py`.
The script never touches training state; it reads runtime/train.log only.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress, t


ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runtime" / "train.log"
OUT_PNG = ROOT / "runtime" / "policy_loss_analysis.png"
OUT_SVG = ROOT / "runtime" / "policy_loss_analysis_scipy.svg"
OUT_JSON = ROOT / "runtime" / "policy_loss_fit_scipy.json"


def load_history():
    by_iter = {}
    promotions = []
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("phase") == "train_done":
            by_iter[int(event["iter"])] = event
        if event.get("phase") == "gate" and event.get("result") == "promoted":
            promotions.append(int(event["iter"]))
    rows = [by_iter[i] for i in sorted(by_iter)]
    return rows, sorted(set(promotions))


def rolling_mean(values, window):
    kernel = np.ones(window) / window
    prefix = np.convolve(values, kernel, mode="valid")
    head = np.array([np.mean(values[:i]) for i in range(1, window)])
    return np.concatenate([head, prefix])


def r2_rmse(actual, predicted):
    residual = actual - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 - ss_res / ss_tot, float(np.sqrt(np.mean(residual**2)))


def fit_asymptotes(iterations, losses, fit_start=100):
    mask = iterations >= fit_start
    x = iterations[mask]
    y = losses[mask]
    t = x - fit_start + 1.0
    ymin = float(np.min(y))

    def exp_model(t_, floor, amplitude, rate):
        return floor + amplitude * np.exp(-rate * t_)

    def power_model(t_, floor, amplitude, exponent):
        return floor + amplitude * np.power(t_, -exponent)

    bounds = ([0.0, 0.0, 1e-7], [ymin - 1e-4, 10.0, 5.0])
    exp_params, _ = curve_fit(
        exp_model,
        t,
        y,
        p0=[max(0.0, ymin - 0.25), max(0.05, y[0] - ymin + 0.25), 0.005],
        bounds=bounds,
        maxfev=200000,
    )
    power_params, _ = curve_fit(
        power_model,
        t,
        y,
        p0=[max(0.0, ymin - 0.25), max(0.05, y[0] - ymin + 0.25), 0.25],
        bounds=bounds,
        maxfev=200000,
    )
    exp_pred = exp_model(t, *exp_params)
    power_pred = power_model(t, *power_params)
    exp_r2, exp_rmse = r2_rmse(y, exp_pred)
    power_r2, power_rmse = r2_rmse(y, power_pred)
    return {
        "fit_start": fit_start,
        "exp": {
            "params": exp_params,
            "r2": exp_r2,
            "rmse": exp_rmse,
            "predict": lambda iterations_: exp_model(
                np.asarray(iterations_) - fit_start + 1.0, *exp_params
            ),
        },
        "power": {
            "params": power_params,
            "r2": power_r2,
            "rmse": power_rmse,
            "predict": lambda iterations_: power_model(
                np.asarray(iterations_) - fit_start + 1.0, *power_params
            ),
        },
    }


def linear_fit(iterations, losses, window):
    x = iterations[-window:]
    y = losses[-window:]
    result = linregress(x, y)
    slope = result.slope
    intercept = result.intercept
    pred = slope * x + intercept
    r2, rmse = r2_rmse(y, pred)
    critical = float(t.ppf(0.975, df=len(x) - 2))
    ci_low = float(slope - critical * result.stderr)
    ci_high = float(slope + critical * result.stderr)
    return {
        "window": window,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "rmse": rmse,
        "pvalue": float(result.pvalue),
        "slope_stderr": float(result.stderr),
        "slope_ci95": [ci_low, ci_high],
        "predict": lambda iterations_: slope * np.asarray(iterations_) + intercept,
    }


def serializable(model):
    out = {key: value for key, value in model.items() if key != "predict"}
    if "params" in out:
        out["params"] = [float(v) for v in out["params"]]
    return out


def main():
    rows, promotions = load_history()
    iterations = np.array([int(row["iter"]) for row in rows], dtype=float)
    policy = np.array([float(row["policy_loss"]) for row in rows])
    value = np.array([float(row["value_loss"]) for row in rows])
    rolling5 = rolling_mean(policy, 5)
    rolling15 = rolling_mean(policy, 15)
    asymptotes = fit_asymptotes(iterations, policy, fit_start=100)
    linear40 = linear_fit(iterations, policy, min(40, len(rows)))
    linear80 = linear_fit(iterations, policy, min(80, len(rows)))

    current = int(iterations[-1])
    future = np.array([current + 15, current + 30, 400], dtype=float)
    report = {
        "points": len(rows),
        "latest_iteration": current,
        "latest_policy_loss": float(policy[-1]),
        "minimum_policy_loss": float(np.min(policy)),
        "rolling5_latest": float(rolling5[-1]),
        "rolling15_latest": float(rolling15[-1]),
        "fit_start": asymptotes["fit_start"],
        "exponential": serializable(asymptotes["exp"]),
        "power": serializable(asymptotes["power"]),
        "recent_linear_40": serializable(linear40),
        "recent_linear_80": serializable(linear80),
        "forecasts": {
            str(int(x)): {
                "exponential": float(asymptotes["exp"]["predict"]([x])[0]),
                "power": float(asymptotes["power"]["predict"]([x])[0]),
                "linear40": float(linear40["predict"]([x])[0]),
                "linear80": float(linear80["predict"]([x])[0]),
            }
            for x in future
        },
        "warning": (
            "The training recipe changed over time and loss is noisy. "
            "Fits summarize observed trends; plateau monitor remains the stop authority."
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2.3, 1.4, 1.2]}
    )
    fig.suptitle(
        f"AlphaZero Gomoku training loss — {len(rows)} complete iterations",
        fontsize=18,
    )

    # Full run.
    ax1.scatter(iterations, policy, s=12, alpha=0.38, color="#38bdf8", label="policy loss")
    ax1.plot(iterations, rolling5, lw=2.3, color="#fbbf24", label="rolling mean (5)")
    ax1.plot(iterations, rolling15, lw=1.5, color="#fb7185", alpha=0.9, label="rolling mean (15)")
    for gate in promotions:
        ax1.axvline(gate, color="#22c55e", alpha=0.18, lw=1)
    ax1.set_ylabel("policy loss")
    ax1.set_title("Full history (green vertical lines = best-model promotions)")
    ax1.grid(alpha=0.18)
    ax1.legend(ncol=3)

    # Stable-stage asymptote comparison.
    fit_start = asymptotes["fit_start"]
    stable_mask = iterations >= fit_start
    fit_x = np.linspace(fit_start, max(400, current + 30), 500)
    ax2.scatter(iterations[stable_mask], policy[stable_mask], s=13, alpha=0.35, color="#38bdf8", label="observed")
    ax2.plot(fit_x, asymptotes["exp"]["predict"](fit_x), color="#34d399", lw=2.2,
             label=f"exp floor={asymptotes['exp']['params'][0]:.3f}, R2={asymptotes['exp']['r2']:.3f}")
    ax2.plot(fit_x, asymptotes["power"]["predict"](fit_x), color="#a78bfa", lw=2.2, ls="--",
             label=f"power floor={asymptotes['power']['params'][0]:.3f}, R2={asymptotes['power']['r2']:.3f}")
    ax2.axvline(current, color="#94a3b8", ls=":", lw=1)
    ax2.set_xlim(fit_start, max(400, current + 30))
    ax2.set_ylabel("policy loss")
    ax2.set_title("Constrained asymptotic fits (iter 100+) — diagnostic, not a stop rule")
    ax2.grid(alpha=0.18)
    ax2.legend()

    # Recent plateau view.
    recent_start = max(int(iterations[0]), current - 100)
    recent_mask = iterations >= recent_start
    line_x = np.linspace(current - 80, current + 30, 300)
    ax3.scatter(iterations[recent_mask], policy[recent_mask], s=20, alpha=0.48, color="#38bdf8", label="observed")
    ax3.plot(iterations[recent_mask], rolling5[recent_mask], color="#fbbf24", lw=2.2, label="rolling mean (5)")
    ax3.plot(line_x, linear40["predict"](line_x), color="#fb7185", lw=2.1,
             label=f"linear 40: {linear40['slope']:.6f}/iter, R2={linear40['r2']:.3f}")
    ax3.plot(line_x, linear80["predict"](line_x), color="#60a5fa", lw=1.8, ls="--",
             label=f"linear 80: {linear80['slope']:.6f}/iter, R2={linear80['r2']:.3f}")
    ax3.axvline(current, color="#94a3b8", ls=":", lw=1)
    ax3.set_xlim(recent_start, current + 30)
    ax3.set_xlabel("training iteration")
    ax3.set_ylabel("policy loss")
    ax3.set_title("Recent stage and linear trend")
    ax3.grid(alpha=0.18)
    ax3.legend(ncol=2)

    text = (
        f"Latest: iter {current}, policy={policy[-1]:.4f}, rolling5={rolling5[-1]:.4f} | "
        f"linear40 slope={linear40['slope']:.6f}/iter | "
        "Training recipe changed over time; extrapolation uncertainty is high."
    )
    fig.text(0.5, 0.012, text, ha="center", fontsize=10, color="#cbd5e1")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(OUT_PNG, dpi=160)
    fig.savefig(OUT_SVG)
    print(json.dumps(report, indent=2))
    print(OUT_PNG)


if __name__ == "__main__":
    main()
