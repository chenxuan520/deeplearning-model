#!/usr/bin/env python3
"""Plot and fit AlphaZero policy-loss history using only the stdlib."""

import csv
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runtime" / "train.log"
OUT_SVG = ROOT / "runtime" / "policy_loss_analysis.svg"
OUT_CSV = ROOT / "runtime" / "policy_loss_history.csv"
OUT_JSON = ROOT / "runtime" / "policy_loss_fit.json"


def load_history():
    by_iter = {}
    promotions = []
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("phase") == "train_done":
            by_iter[int(event["iter"])] = {
                "iteration": int(event["iter"]),
                "step": int(event["step"]),
                "policy_loss": float(event["policy_loss"]),
                "value_loss": float(event["value_loss"]),
            }
        if event.get("phase") == "gate" and event.get("result") == "promoted":
            promotions.append(int(event["iter"]))
    return [by_iter[k] for k in sorted(by_iter)], sorted(set(promotions))


def rolling_mean(values, window):
    out = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def linear_fit(xs, ys):
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom
    intercept = ym - slope * xm
    pred = [intercept + slope * x for x in xs]
    return {
        "slope": slope,
        "intercept": intercept,
        "rmse": rmse(ys, pred),
        "r2": r_squared(ys, pred),
        "predict": lambda x: intercept + slope * x,
    }


def r_squared(actual, predicted):
    mean = sum(actual) / len(actual)
    total = sum((v - mean) ** 2 for v in actual)
    residual = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return 1.0 - residual / total if total > 0 else 0.0


def rmse(actual, predicted):
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def asymptotic_fit(xs, ys, mode):
    """Fit y=c+a*x^-b or y=c+a*exp(-k*x), searching c."""
    minimum = min(ys)
    best = None
    # The floor must remain below every observed value.  Searching it directly
    # makes the transformed regression stable and dependency-free.
    for index in range(500):
        floor = (minimum - 0.002) * index / 500.0
        transformed = [math.log(max(y - floor, 1e-12)) for y in ys]
        axis = [math.log(x) if mode == "power" else x for x in xs]
        fit = linear_fit(axis, transformed)
        rate = -fit["slope"]
        if rate <= 0:
            continue
        amplitude = math.exp(fit["intercept"])
        if mode == "power":
            pred = [floor + amplitude * x ** (-rate) for x in xs]
        else:
            pred = [floor + amplitude * math.exp(-rate * x) for x in xs]
        score = rmse(ys, pred)
        if best is None or score < best["rmse"]:
            best = {
                "mode": mode,
                "floor": floor,
                "amplitude": amplitude,
                "rate": rate,
                "rmse": score,
                "r2": r_squared(ys, pred),
            }
    if mode == "power":
        best["predict"] = lambda x: best["floor"] + best["amplitude"] * x ** (-best["rate"])
    else:
        best["predict"] = lambda x: best["floor"] + best["amplitude"] * math.exp(-best["rate"] * x)
    return best


def points_path(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def make_svg(rows, promotions, rolling, power, exponential, recent):
    width, height = 1400, 900
    left, right = 90, 35
    plot_w = width - left - right
    panels = [
        {"top": 70, "height": 470, "start": rows[0]["iteration"], "title": "Full policy-loss history"},
        {"top": 630, "height": 210, "start": max(rows[0]["iteration"], rows[-1]["iteration"] - 80), "title": "Recent 80 iterations (plateau view)"},
    ]
    max_iter = rows[-1]["iteration"]
    raw_y = [row["policy_loss"] for row in rows]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#cbd5e1}.title{font-size:25px;font-weight:bold;fill:#f8fafc}.sub{font-size:15px;fill:#94a3b8}.axis{stroke:#64748b;stroke-width:1}.grid{stroke:#334155;stroke-width:1}.raw{fill:#38bdf8;opacity:.42}.roll{fill:none;stroke:#fbbf24;stroke-width:4}.power{fill:none;stroke:#a78bfa;stroke-width:3;stroke-dasharray:10 7}.expo{fill:none;stroke:#34d399;stroke-width:2;stroke-dasharray:4 6}.recent{fill:none;stroke:#fb7185;stroke-width:3}.gate{stroke:#22c55e;stroke-width:1;opacity:.28}</style>',
        '<text x="90" y="34" class="title">AlphaZero Gomoku policy-loss curve and fitted trends</text>',
        f'<text x="90" y="56" class="sub">{len(rows)} iterations | latest iter {max_iter} loss {raw_y[-1]:.4f} | gate promotions shown in green</text>',
    ]

    for panel_index, panel in enumerate(panels):
        subset = [r for r in rows if r["iteration"] >= panel["start"]]
        xmin = panel["start"]
        xmax = max_iter + (30 if panel_index == 1 else 0)
        if panel_index == 0:
            ymin = max(0.0, min(raw_y) - 0.15)
            ymax = max(raw_y) + 0.15
        else:
            values = [r["policy_loss"] for r in subset]
            ymin = min(values) - 0.06
            ymax = max(values) + 0.06
        top, ph = panel["top"], panel["height"]

        def sx(x):
            return left + (x - xmin) / (xmax - xmin) * plot_w

        def sy(y):
            return top + ph - (y - ymin) / (ymax - ymin) * ph

        svg.append(f'<text x="{left}" y="{top - 15}" class="title" font-size="19">{html.escape(panel["title"])}</text>')
        for i in range(6):
            value = ymin + (ymax - ymin) * i / 5
            y = sy(value)
            svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" class="grid"/>')
            svg.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-size="13">{value:.2f}</text>')
        for i in range(7):
            value = xmin + (xmax - xmin) * i / 6
            x = sx(value)
            svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+ph}" class="grid"/>')
            svg.append(f'<text x="{x:.2f}" y="{top+ph+23}" text-anchor="middle" font-size="13">{value:.0f}</text>')
        for gate in promotions:
            if xmin <= gate <= xmax:
                svg.append(f'<line x1="{sx(gate):.2f}" y1="{top}" x2="{sx(gate):.2f}" y2="{top+ph}" class="gate"/>')
        for row in subset:
            svg.append(f'<circle cx="{sx(row["iteration"]):.2f}" cy="{sy(row["policy_loss"]):.2f}" r="3.2" class="raw"/>')
        roll_points = [(sx(row["iteration"]), sy(rolling[i])) for i, row in enumerate(rows) if row["iteration"] >= xmin]
        svg.append(f'<polyline points="{points_path(roll_points)}" class="roll"/>')
        if panel_index == 0:
            fit_xs = range(max(100, xmin), max_iter + 1)
            p_points = [(sx(x), sy(power["predict"](x))) for x in fit_xs]
            e_points = [(sx(x), sy(exponential["predict"](x))) for x in fit_xs]
            svg.append(f'<polyline points="{points_path(p_points)}" class="power"/>')
            svg.append(f'<polyline points="{points_path(e_points)}" class="expo"/>')
        else:
            line_xs = range(max(panel["start"], max_iter - 40), xmax + 1)
            line_points = [(sx(x), sy(recent["predict"](x))) for x in line_xs]
            svg.append(f'<polyline points="{points_path(line_points)}" class="recent"/>')
        svg.append(f'<line x1="{left}" y1="{top+ph}" x2="{left+plot_w}" y2="{top+ph}" class="axis"/>')
        svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" class="axis"/>')

    svg.extend([
        '<circle cx="100" cy="580" r="5" fill="#38bdf8" opacity=".55"/><text x="115" y="585" font-size="14">raw loss</text>',
        '<line x1="220" y1="580" x2="265" y2="580" class="roll"/><text x="275" y="585" font-size="14">rolling mean (5)</text>',
        '<line x1="430" y1="580" x2="475" y2="580" class="power"/><text x="485" y="585" font-size="14">power asymptote (iter 100+)</text>',
        '<line x1="720" y1="580" x2="765" y2="580" class="expo"/><text x="775" y="585" font-size="14">exponential asymptote</text>',
        '<line x1="1010" y1="580" x2="1055" y2="580" class="recent"/><text x="1065" y="585" font-size="14">recent linear fit</text>',
        f'<text x="90" y="875" class="sub">Power floor={power["floor"]:.4f}, exponent={power["rate"]:.4f}, R2={power["r2"]:.3f}, RMSE={power["rmse"]:.4f} | '
        f'Exp floor={exponential["floor"]:.4f}, k={exponential["rate"]:.5f}, R2={exponential["r2"]:.3f} | '
        f'Recent40 slope={recent["slope"]:.6f}/iter, R2={recent["r2"]:.3f}</text>',
        '</svg>',
    ])
    return "\n".join(svg)


def serializable(fit):
    return {k: v for k, v in fit.items() if k != "predict"}


def main():
    rows, promotions = load_history()
    if len(rows) < 40:
        raise SystemExit("not enough loss history")
    xs = [r["iteration"] for r in rows]
    losses = [r["policy_loss"] for r in rows]
    rolling = rolling_mean(losses, 5)

    stable_rows = [r for r in rows if r["iteration"] >= 100]
    stable_x = [r["iteration"] for r in stable_rows]
    stable_y = [r["policy_loss"] for r in stable_rows]
    power = asymptotic_fit(stable_x, stable_y, "power")
    exponential = asymptotic_fit(stable_x, stable_y, "exponential")
    recent_rows = rows[-40:]
    recent = linear_fit(
        [r["iteration"] for r in recent_rows],
        [r["policy_loss"] for r in recent_rows],
    )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "step", "policy_loss", "value_loss", "rolling5"])
        writer.writeheader()
        for row, roll in zip(rows, rolling):
            writer.writerow({**row, "rolling5": f"{roll:.6f}"})

    report = {
        "points": len(rows),
        "latest_iteration": rows[-1]["iteration"],
        "latest_policy_loss": rows[-1]["policy_loss"],
        "minimum_policy_loss": min(losses),
        "rolling5_latest": rolling[-1],
        "fit_range": {"start": 100, "end": rows[-1]["iteration"]},
        "power_asymptote": serializable(power),
        "exponential_asymptote": serializable(exponential),
        "recent_linear_40": serializable(recent),
        "forecasts": {
            str(x): {
                "power": power["predict"](x),
                "exponential": exponential["predict"](x),
                "recent_linear": recent["predict"](x),
            }
            for x in (rows[-1]["iteration"] + 15, rows[-1]["iteration"] + 30, 400)
        },
        "warning": "Training recipe changed over time; fits describe the observed curve and are not stop criteria.",
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_SVG.write_text(make_svg(rows, promotions, rolling, power, exponential, recent), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(OUT_SVG)


if __name__ == "__main__":
    main()
