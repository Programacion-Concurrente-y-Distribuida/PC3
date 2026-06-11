#!/usr/bin/env python3
"""
Benchmark de parser/encoder workers para train-linear.

Salida:
  - scripts/worker_benchmark/results/worker_benchmark.csv
  - scripts/worker_benchmark/results/time_heatmap.png
  - scripts/worker_benchmark/results/ram_heatmap.png
  - scripts/worker_benchmark/results/pareto_time_vs_ram.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as exc:
    print("Falta matplotlib/numpy. Instala con: pip3 install matplotlib numpy")
    raise


METRIC_RE_TRAIN = re.compile(
    r"Train -> MAE: ([0-9.]+) \| RMSE: ([0-9.]+) \| R2: ([0-9.]+)"
)
METRIC_RE_TEST = re.compile(
    r"Test\s+-> MAE: ([0-9.]+) \| RMSE: ([0-9.]+) \| R2: ([0-9.]+)"
)


def parse_list(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("La lista no puede estar vacia")
    return out


def get_peak_rss_kb(pid: int) -> int:
    """
    Lee RSS KB del proceso.
    Si falla momentaneamente, retorna 0 para esa muestra.
    """
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        val = proc.stdout.strip()
        return int(val) if val else 0
    except Exception:
        return 0


def run_one(cmd: list[str], poll_ms: int = 80) -> tuple[float, int, str, int]:
    """
    Ejecuta comando, mide tiempo wall y pico RSS del proceso principal.
    """
    start = time.perf_counter()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    peak_kb = 0
    output_lines = []
    poll_s = max(poll_ms, 20) / 1000.0

    while True:
        if p.stdout is not None:
            line = p.stdout.readline()
            if line:
                output_lines.append(line)
        rss = get_peak_rss_kb(p.pid)
        if rss > peak_kb:
            peak_kb = rss

        if p.poll() is not None:
            break
        time.sleep(poll_s)

    if p.stdout is not None:
        remaining = p.stdout.read()
        if remaining:
            output_lines.append(remaining)

    elapsed = time.perf_counter() - start
    return elapsed, peak_kb, "".join(output_lines), p.returncode


def extract_metrics(out: str) -> dict[str, float | str]:
    mt = METRIC_RE_TRAIN.search(out)
    ms = METRIC_RE_TEST.search(out)
    if not mt or not ms:
        return {
            "train_mae": "",
            "train_rmse": "",
            "train_r2": "",
            "test_mae": "",
            "test_rmse": "",
            "test_r2": "",
        }
    return {
        "train_mae": float(mt.group(1)),
        "train_rmse": float(mt.group(2)),
        "train_r2": float(mt.group(3)),
        "test_mae": float(ms.group(1)),
        "test_rmse": float(ms.group(2)),
        "test_r2": float(ms.group(3)),
    }


def build_binary(repo_root: Path, bin_path: Path) -> None:
    cmd = [
        "go",
        "build",
        "-o",
        str(bin_path),
        "./cmd/train-linear",
    ]
    p = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError("No se pudo compilar train-linear")


def save_heatmap(
    matrix: np.ndarray,
    row_labels: list[int],
    col_labels: list[int],
    title: str,
    cbar_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Encoder workers")
    ax.set_ylabel("Parser workers")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = "NA" if math.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_scatter(rows: list[dict[str, str]], out_path: Path) -> None:
    x = []
    y = []
    lbl = []
    for r in rows:
        try:
            x.append(float(r["seconds"]))
            y.append(float(r["peak_rss_mb"]))
            lbl.append(f"P{r['parser_workers']}-E{r['encoder_workers']}")
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=45)
    for xi, yi, t in zip(x, y, lbl):
        ax.annotate(t, (xi, yi), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Tiempo total (s)")
    ax.set_ylabel("RAM pico (MB)")
    ax.set_title("Pareto Tiempo vs RAM")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark parser/encoder workers")
    parser.add_argument("--repo-root", default=".", help="Ruta root del repo")
    parser.add_argument("--input", default="aqs_final_3M.csv")
    parser.add_argument("--train-year-end", type=int, default=2020)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-features", required=True)
    parser.add_argument("--cat-features", required=True)
    parser.add_argument("--parser-workers", default="2,4,6,8,10,12")
    parser.add_argument("--encoder-workers", default="2,4,6,8,10")
    parser.add_argument("--raw-buffer", default=0, type=int, help="0 => auto (2*parser)")
    parser.add_argument("--parsed-buffer", default=0, type=int, help="0 => auto (2*encoder)")
    parser.add_argument("--encoded-buffer", default=3, type=int)
    parser.add_argument("--solver", default="ridge")
    parser.add_argument("--fit-workers", default=4, type=int)
    parser.add_argument("--ridge-lambda", default=1.0, type=float)
    parser.add_argument("--poll-ms", default=80, type=int)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / "scripts" / "worker_benchmark" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "train-linear-bench"
    csv_path = out_dir / "worker_benchmark.csv"

    parser_ws = parse_list(args.parser_workers)
    encoder_ws = parse_list(args.encoder_workers)

    print("Compilando binario...")
    build_binary(repo_root, bin_path)

    rows = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "parser_workers",
                "encoder_workers",
                "raw_buffer",
                "parsed_buffer",
                "encoded_buffer",
                "seconds",
                "peak_rss_mb",
                "train_mae",
                "train_rmse",
                "train_r2",
                "test_mae",
                "test_rmse",
                "test_r2",
                "status",
            ],
        )
        w.writeheader()

        for p in parser_ws:
            for e in encoder_ws:
                raw_b = args.raw_buffer if args.raw_buffer > 0 else (2 * p)
                parsed_b = args.parsed_buffer if args.parsed_buffer > 0 else (2 * e)
                cmd = [
                    str(bin_path),
                    "-input",
                    args.input,
                    "-train-year-end",
                    str(args.train_year_end),
                    "-max-rows",
                    str(args.max_rows),
                    "-num-features",
                    args.num_features,
                    "-cat-features",
                    args.cat_features,
                    "-parser-workers",
                    str(p),
                    "-encoder-workers",
                    str(e),
                    "-raw-buffer",
                    str(raw_b),
                    "-parsed-buffer",
                    str(parsed_b),
                    "-encoded-buffer",
                    str(args.encoded_buffer),
                    "-solver",
                    args.solver,
                    "-fit-workers",
                    str(args.fit_workers),
                    "-ridge-lambda",
                    str(args.ridge_lambda),
                ]
                print(f"Ejecutando P={p} E={e} ...")
                secs, peak_kb, out, rc = run_one(cmd, poll_ms=args.poll_ms)
                metrics = extract_metrics(out)

                row = {
                    "parser_workers": p,
                    "encoder_workers": e,
                    "raw_buffer": raw_b,
                    "parsed_buffer": parsed_b,
                    "encoded_buffer": args.encoded_buffer,
                    "seconds": f"{secs:.4f}",
                    "peak_rss_mb": f"{peak_kb / 1024.0:.2f}",
                    "train_mae": metrics["train_mae"],
                    "train_rmse": metrics["train_rmse"],
                    "train_r2": metrics["train_r2"],
                    "test_mae": metrics["test_mae"],
                    "test_rmse": metrics["test_rmse"],
                    "test_r2": metrics["test_r2"],
                    "status": "ok" if rc == 0 else f"error_{rc}",
                }
                w.writerow(row)
                rows.append({k: str(v) for k, v in row.items()})

                # Guarda output crudo por combinación para auditoría.
                out_log = out_dir / f"run_p{p}_e{e}.log"
                out_log.write_text(out, encoding="utf-8")

    # Plot data
    time_matrix = np.full((len(parser_ws), len(encoder_ws)), np.nan)
    ram_matrix = np.full((len(parser_ws), len(encoder_ws)), np.nan)

    idx_p = {p: i for i, p in enumerate(parser_ws)}
    idx_e = {e: j for j, e in enumerate(encoder_ws)}

    for r in rows:
        if not r["status"].startswith("ok"):
            continue
        i = idx_p[int(r["parser_workers"])]
        j = idx_e[int(r["encoder_workers"])]
        time_matrix[i, j] = float(r["seconds"])
        ram_matrix[i, j] = float(r["peak_rss_mb"])

    save_heatmap(
        time_matrix,
        parser_ws,
        encoder_ws,
        "Tiempo total (s) por combinación de workers",
        "segundos",
        out_dir / "time_heatmap.png",
    )
    save_heatmap(
        ram_matrix,
        parser_ws,
        encoder_ws,
        "RAM pico (MB) por combinación de workers",
        "MB",
        out_dir / "ram_heatmap.png",
    )
    save_scatter(rows, out_dir / "pareto_time_vs_ram.png")

    print("\nBenchmark terminado.")
    print(f"CSV: {csv_path}")
    print(f"Graficos: {out_dir / 'time_heatmap.png'}")
    print(f"          {out_dir / 'ram_heatmap.png'}")
    print(f"          {out_dir / 'pareto_time_vs_ram.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
