#!/usr/bin/env python3
"""
Benchmark de fit-workers para el solver concurrente Ridge.

Salida:
  - scripts/worker_benchmark/results/fit_workers_benchmark.csv
  - scripts/worker_benchmark/results/fit_workers_time_ram.png
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
import statistics
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    print("Falta matplotlib. Instala con: pip3 install matplotlib")
    raise


METRIC_RE_TEST = re.compile(
    r"Test\s+-> MAE: ([0-9.]+) \| RMSE: ([0-9.]+) \| R2: ([0-9.]+)"
)
FIT_RE = re.compile(r"\[profile\] end\s+fit_linear_regression\s+elapsed=\s*([0-9.]+)s")


def parse_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def get_peak_rss_kb(pid: int) -> int:
    try:
        p = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
        val = p.stdout.strip()
        return int(val) if val else 0
    except Exception:
        return 0


def run_one(cmd: list[str]) -> tuple[float, int, str, int]:
    start = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    peak_kb = 0
    out = []
    while True:
        if p.stdout is not None:
            line = p.stdout.readline()
            if line:
                out.append(line)
        peak_kb = max(peak_kb, get_peak_rss_kb(p.pid))
        if p.poll() is not None:
            break
    if p.stdout is not None:
        rest = p.stdout.read()
        if rest:
            out.append(rest)
    return time.perf_counter() - start, peak_kb, "".join(out), p.returncode


def build_binary(repo_root: Path, bin_path: Path) -> None:
    p = subprocess.run(
        ["go", "build", "-o", str(bin_path), "./cmd/train-linear"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError("No se pudo compilar train-linear")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fit-workers (Ridge concurrente)")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--input", default="aqs_final_3M.csv")
    parser.add_argument("--train-year-end", type=int, default=2020)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-features", required=True)
    parser.add_argument("--cat-features", required=True)
    parser.add_argument("--fit-workers", default="4,8,12,16,20,24,28,32")
    parser.add_argument("--parser-workers", type=int, default=2)
    parser.add_argument("--encoder-workers", type=int, default=2)
    parser.add_argument("--raw-buffer", type=int, default=4)
    parser.add_argument("--parsed-buffer", type=int, default=4)
    parser.add_argument("--encoded-buffer", type=int, default=3)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats debe ser >= 1")

    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / "scripts" / "worker_benchmark" / "results"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "train-linear-fit-bench"
    raw_csv_path = out_dir / "fit_workers_benchmark_raw.csv"
    summary_csv_path = out_dir / "fit_workers_benchmark.csv"

    build_binary(repo_root, bin_path)

    raw_rows = []
    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fit_workers",
                "repeat",
                "seconds",
                "fit_seconds",
                "peak_rss_mb",
                "test_mae",
                "test_rmse",
                "test_r2",
                "status",
            ],
        )
        writer.writeheader()

        for fw in parse_list(args.fit_workers):
            for repeat in range(1, args.repeats + 1):
                print(f"Ejecutando fit-workers={fw} repeat={repeat}/{args.repeats} ...")
                cmd = [
                    str(bin_path),
                    "-input", args.input,
                    "-train-year-end", str(args.train_year_end),
                    "-max-rows", str(args.max_rows),
                    "-num-features", args.num_features,
                    "-cat-features", args.cat_features,
                    "-parser-workers", str(args.parser_workers),
                    "-encoder-workers", str(args.encoder_workers),
                    "-raw-buffer", str(args.raw_buffer),
                    "-parsed-buffer", str(args.parsed_buffer),
                    "-encoded-buffer", str(args.encoded_buffer),
                    "-solver", "ridge",
                    "-fit-workers", str(fw),
                    "-ridge-lambda", str(args.ridge_lambda),
                    "-profile",
                ]
                seconds, peak_kb, out, rc = run_one(cmd)
                metric = METRIC_RE_TEST.search(out)
                fit = FIT_RE.search(out)
                row = {
                    "fit_workers": fw,
                    "repeat": repeat,
                    "seconds": f"{seconds:.4f}",
                    "fit_seconds": fit.group(1) if fit else "",
                    "peak_rss_mb": f"{peak_kb / 1024.0:.2f}",
                    "test_mae": metric.group(1) if metric else "",
                    "test_rmse": metric.group(2) if metric else "",
                    "test_r2": metric.group(3) if metric else "",
                    "status": "ok" if rc == 0 else f"error_{rc}",
                }
                raw_rows.append(row)
                writer.writerow(row)
                logs_dir.mkdir(parents=True, exist_ok=True)
                (logs_dir / f"fit_workers_{fw}_r{repeat}.log").write_text(out, encoding="utf-8")

    def mean(vals: list[float]) -> float:
        return statistics.fmean(vals) if vals else float("nan")

    def stdev(vals: list[float]) -> float:
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    summary_rows = []
    for fw in parse_list(args.fit_workers):
        group = [
            r for r in raw_rows
            if int(r["fit_workers"]) == fw and r["status"] == "ok" and r["fit_seconds"]
        ]
        if not group:
            continue
        seconds_vals = [float(r["seconds"]) for r in group]
        fit_vals = [float(r["fit_seconds"]) for r in group]
        ram_vals = [float(r["peak_rss_mb"]) for r in group]
        mae_vals = [float(r["test_mae"]) for r in group if r["test_mae"]]
        rmse_vals = [float(r["test_rmse"]) for r in group if r["test_rmse"]]
        r2_vals = [float(r["test_r2"]) for r in group if r["test_r2"]]
        summary_rows.append({
            "fit_workers": fw,
            "runs": len(group),
            "seconds_mean": mean(seconds_vals),
            "seconds_median": statistics.median(seconds_vals),
            "seconds_std": stdev(seconds_vals),
            "seconds_min": min(seconds_vals),
            "fit_seconds_mean": mean(fit_vals),
            "fit_seconds_median": statistics.median(fit_vals),
            "fit_seconds_std": stdev(fit_vals),
            "fit_seconds_min": min(fit_vals),
            "peak_rss_mb_mean": mean(ram_vals),
            "peak_rss_mb_median": statistics.median(ram_vals),
            "peak_rss_mb_std": stdev(ram_vals),
            "test_mae_mean": mean(mae_vals),
            "test_rmse_mean": mean(rmse_vals),
            "test_r2_mean": mean(r2_vals),
        })

    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "fit_workers",
            "runs",
            "seconds_mean",
            "seconds_median",
            "seconds_std",
            "seconds_min",
            "fit_seconds_mean",
            "fit_seconds_median",
            "fit_seconds_std",
            "fit_seconds_min",
            "peak_rss_mb_mean",
            "peak_rss_mb_median",
            "peak_rss_mb_std",
            "test_mae_mean",
            "test_rmse_mean",
            "test_r2_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in r.items()
            })

    xs = [int(r["fit_workers"]) for r in summary_rows]
    total = [float(r["seconds_mean"]) for r in summary_rows]
    total_err = [float(r["seconds_std"]) for r in summary_rows]
    fit = [float(r["fit_seconds_mean"]) for r in summary_rows]
    fit_err = [float(r["fit_seconds_std"]) for r in summary_rows]
    ram = [float(r["peak_rss_mb_mean"]) for r in summary_rows]
    ram_err = [float(r["peak_rss_mb_std"]) for r in summary_rows]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.errorbar(xs, total, yerr=total_err, marker="o", capsize=3, label="Tiempo total promedio (s)")
    ax1.errorbar(xs, fit, yerr=fit_err, marker="o", capsize=3, label="Fit regression promedio (s)")
    ax1.set_xlabel("fit-workers")
    ax1.set_ylabel("Tiempo (s)")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.errorbar(xs, ram, yerr=ram_err, marker="s", capsize=3, color="tab:red", label="RAM pico promedio (MB)")
    ax2.set_ylabel("RAM pico (MB)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title(f"Efecto de fit-workers en tiempo y RAM (promedio de {args.repeats} corridas)")
    fig.tight_layout()
    fig.savefig(out_dir / "fit_workers_time_ram.png", dpi=160)
    plt.close(fig)

    print(f"CSV resumen: {summary_csv_path}")
    print(f"CSV crudo: {raw_csv_path}")
    print(f"Grafico: {out_dir / 'fit_workers_time_ram.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
