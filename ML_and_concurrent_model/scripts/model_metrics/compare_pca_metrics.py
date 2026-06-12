#!/usr/bin/env python3
"""
Ejecuta train-linear sin PCA y con PCA, extrae metricas y genera un analisis comparativo.

Salidas:
  - scripts/model_metrics/results/model_metrics_raw.csv
  - scripts/model_metrics/results/model_metrics_summary.csv
  - scripts/model_metrics/results/model_metrics_analysis.md
  - scripts/model_metrics/results/model_metrics_comparison.png
  - scripts/model_metrics/results/logs/*.log
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    print("Falta matplotlib. Instala con: pip3 install matplotlib")
    raise


METRIC_RE_TRAIN = re.compile(
    r"Train -> MAE: ([0-9.]+) \| RMSE: ([0-9.]+) \| R2: ([0-9.]+)"
)
METRIC_RE_TEST = re.compile(
    r"Test\s+-> MAE: ([0-9.]+) \| RMSE: ([0-9.]+) \| R2: ([0-9.]+)"
)
PCA_RE = re.compile(r"PCA: activado \| componentes=([0-9]+) \| varianza explicada=([0-9.]+)")
PROFILE_RE = re.compile(r"\[profile\] end\s+([a-zA-Z0-9_]+)\s+elapsed=\s*([0-9.]+)s")


def parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("La lista de varianzas PCA no puede estar vacia")
    for value in values:
        if value <= 0 or value > 1:
            raise ValueError("--pca-variances debe contener valores en el rango (0, 1]")
    return values


def get_peak_rss_kb(pid: int) -> int:
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        value = proc.stdout.strip()
        return int(value) if value else 0
    except Exception:
        return 0


def run_one(cmd: list[str], poll_ms: int = 80) -> tuple[float, int, str, int]:
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    peak_kb = 0
    output = []
    poll_s = max(poll_ms, 20) / 1000.0

    while True:
        if proc.stdout is not None:
            line = proc.stdout.readline()
            if line:
                output.append(line)
        peak_kb = max(peak_kb, get_peak_rss_kb(proc.pid))
        if proc.poll() is not None:
            break
        time.sleep(poll_s)

    if proc.stdout is not None:
        rest = proc.stdout.read()
        if rest:
            output.append(rest)

    return time.perf_counter() - start, peak_kb, "".join(output), proc.returncode


def build_binary(repo_root: Path, bin_path: Path) -> None:
    proc = subprocess.run(
        ["go", "build", "-o", str(bin_path), "./cmd/train-linear"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("No se pudo compilar train-linear")


def extract_profile_seconds(output: str, stage: str) -> str:
    for name, seconds in PROFILE_RE.findall(output):
        if name == stage:
            return seconds
    return ""


def extract_metrics(output: str) -> dict[str, str]:
    train = METRIC_RE_TRAIN.search(output)
    test = METRIC_RE_TEST.search(output)
    pca = PCA_RE.search(output)
    result = {
        "train_mae": train.group(1) if train else "",
        "train_rmse": train.group(2) if train else "",
        "train_r2": train.group(3) if train else "",
        "test_mae": test.group(1) if test else "",
        "test_rmse": test.group(2) if test else "",
        "test_r2": test.group(3) if test else "",
        "pca_components": pca.group(1) if pca else "",
        "pca_explained_variance": pca.group(2) if pca else "",
        "fit_seconds": extract_profile_seconds(output, "fit_linear_regression"),
        "pca_seconds": extract_profile_seconds(output, "pca"),
    }
    return result


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    scenario_names = []
    for row in raw_rows:
        if row["scenario"] not in scenario_names:
            scenario_names.append(row["scenario"])

    summary = []
    for scenario in scenario_names:
        group = [r for r in raw_rows if r["scenario"] == scenario and r["status"] == "ok"]
        if not group:
            continue

        def vals(key: str) -> list[float]:
            return [float(r[key]) for r in group if r[key] != ""]

        row: dict[str, str] = {
            "scenario": scenario,
            "use_pca": group[0]["use_pca"],
            "pca_variance_goal": group[0]["pca_variance_goal"],
            "runs": str(len(group)),
        }
        for key in [
            "seconds",
            "peak_rss_mb",
            "train_mae",
            "train_rmse",
            "train_r2",
            "test_mae",
            "test_rmse",
            "test_r2",
            "fit_seconds",
            "pca_seconds",
            "pca_components",
            "pca_explained_variance",
        ]:
            numbers = vals(key)
            row[f"{key}_mean"] = f"{mean(numbers):.6f}" if numbers else ""
            row[f"{key}_std"] = f"{stdev(numbers):.6f}" if numbers else ""
        summary.append(row)
    return summary


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_delta(summary: list[dict[str, str]], scenario: str, metric: str) -> float | None:
    base = next((r for r in summary if r["scenario"] == "sin_pca"), None)
    current = next((r for r in summary if r["scenario"] == scenario), None)
    if not base or not current or not base.get(metric) or not current.get(metric):
        return None
    return float(current[metric]) - float(base[metric])


def write_analysis(path: Path, summary: list[dict[str, str]]) -> None:
    lines = [
        "# Analisis de metricas: PCA vs sin PCA",
        "",
        "## Criterio de lectura",
        "",
        "- En MAE y RMSE, menor es mejor.",
        "- En R2, mayor es mejor.",
        "- Si PCA mantiene metricas similares con menos componentes, puede ser util para reducir dimensionalidad.",
        "- Si PCA empeora MAE/RMSE o baja R2 de forma clara, conviene conservar el baseline sin PCA.",
        "",
        "## Resumen",
        "",
        "| escenario | runs | MAE test | RMSE test | R2 test | segundos | RAM MB | componentes PCA | varianza PCA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary:
        lines.append(
            "| {scenario} | {runs} | {test_mae_mean} | {test_rmse_mean} | {test_r2_mean} | "
            "{seconds_mean} | {peak_rss_mb_mean} | {pca_components_mean} | {pca_explained_variance_mean} |".format(
                **row
            )
        )

    lines.extend(["", "## Comparacion contra baseline sin PCA", ""])
    for row in summary:
        if row["scenario"] == "sin_pca":
            continue
        mae_delta = metric_delta(summary, row["scenario"], "test_mae_mean")
        rmse_delta = metric_delta(summary, row["scenario"], "test_rmse_mean")
        r2_delta = metric_delta(summary, row["scenario"], "test_r2_mean")
        lines.append(
            f"- `{row['scenario']}`: delta MAE={mae_delta:+.6f}, "
            f"delta RMSE={rmse_delta:+.6f}, delta R2={r2_delta:+.6f}."
        )

    if len(summary) > 1:
        best_mae = min(summary, key=lambda r: float(r["test_mae_mean"]))
        best_rmse = min(summary, key=lambda r: float(r["test_rmse_mean"]))
        best_r2 = max(summary, key=lambda r: float(r["test_r2_mean"]))
        lines.extend(
            [
                "",
                "## Lectura rapida",
                "",
                f"- Mejor MAE test: `{best_mae['scenario']}` ({best_mae['test_mae_mean']}).",
                f"- Mejor RMSE test: `{best_rmse['scenario']}` ({best_rmse['test_rmse_mean']}).",
                f"- Mejor R2 test: `{best_r2['scenario']}` ({best_r2['test_r2_mean']}).",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(path: Path, summary: list[dict[str, str]]) -> None:
    labels = [r["scenario"] for r in summary]
    mae = [float(r["test_mae_mean"]) for r in summary]
    rmse = [float(r["test_rmse_mean"]) for r in summary]
    r2 = [float(r["test_r2_mean"]) for r in summary]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, values, title, ylabel in [
        (axes[0], mae, "MAE test", "MAE"),
        (axes[1], rmse, "RMSE test", "RMSE"),
        (axes[2], r2, "R2 test", "R2"),
    ]:
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_command(args: argparse.Namespace, bin_path: Path, use_pca: bool, variance: float | None) -> list[str]:
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
        str(args.parser_workers),
        "-encoder-workers",
        str(args.encoder_workers),
        "-raw-buffer",
        str(args.raw_buffer),
        "-parsed-buffer",
        str(args.parsed_buffer),
        "-encoded-buffer",
        str(args.encoded_buffer),
        "-solver",
        args.solver,
        "-fit-workers",
        str(args.fit_workers),
        "-ridge-lambda",
        str(args.ridge_lambda),
        "-profile",
    ]
    if use_pca:
        cmd.extend(["-use-pca", "-pca-variance", str(variance)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara metricas del modelo con PCA y sin PCA")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--input", default="aqs_final_3M.csv")
    parser.add_argument("--train-year-end", type=int, default=2020)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-features", required=True)
    parser.add_argument("--cat-features", required=True)
    parser.add_argument("--parser-workers", type=int, default=2)
    parser.add_argument("--encoder-workers", type=int, default=2)
    parser.add_argument("--raw-buffer", type=int, default=4)
    parser.add_argument("--parsed-buffer", type=int, default=4)
    parser.add_argument("--encoded-buffer", type=int, default=3)
    parser.add_argument("--solver", default="ridge", choices=["ridge", "normal", "svd"])
    parser.add_argument("--fit-workers", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--pca-variances", default="0.95")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats debe ser >= 1")

    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / "scripts" / "model_metrics" / "results"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    bin_path = out_dir / "train-linear-model-metrics"
    build_binary(repo_root, bin_path)

    scenarios: list[tuple[str, bool, float | None]] = [("sin_pca", False, None)]
    scenarios.extend((f"pca_{int(v * 100)}", True, v) for v in parse_float_list(args.pca_variances))

    raw_rows: list[dict[str, str]] = []
    for scenario, use_pca, variance in scenarios:
        for repeat in range(1, args.repeats + 1):
            print(f"Ejecutando {scenario} repeat={repeat}/{args.repeats} ...")
            cmd = build_command(args, bin_path, use_pca, variance)
            seconds, peak_kb, output, rc = run_one(cmd)
            log_path = logs_dir / f"{scenario}_r{repeat}.log"
            log_path.write_text(output, encoding="utf-8")

            row = {
                "scenario": scenario,
                "repeat": str(repeat),
                "use_pca": "true" if use_pca else "false",
                "pca_variance_goal": "" if variance is None else str(variance),
                "seconds": f"{seconds:.6f}",
                "peak_rss_mb": f"{peak_kb / 1024.0:.6f}",
                "status": "ok" if rc == 0 else f"error_{rc}",
                "log_path": str(log_path),
            }
            row.update(extract_metrics(output))
            raw_rows.append(row)

    raw_fields = [
        "scenario",
        "repeat",
        "use_pca",
        "pca_variance_goal",
        "seconds",
        "peak_rss_mb",
        "train_mae",
        "train_rmse",
        "train_r2",
        "test_mae",
        "test_rmse",
        "test_r2",
        "fit_seconds",
        "pca_seconds",
        "pca_components",
        "pca_explained_variance",
        "status",
        "log_path",
    ]
    raw_csv = out_dir / "model_metrics_raw.csv"
    write_csv(raw_csv, raw_rows, raw_fields)

    summary = summarize(raw_rows)
    summary_fields = list(summary[0].keys()) if summary else ["scenario"]
    summary_csv = out_dir / "model_metrics_summary.csv"
    write_csv(summary_csv, summary, summary_fields)

    analysis_path = out_dir / "model_metrics_analysis.md"
    if summary:
        write_analysis(analysis_path, summary)
        save_plot(out_dir / "model_metrics_comparison.png", summary)

    print(f"CSV crudo: {raw_csv}")
    print(f"CSV resumen: {summary_csv}")
    print(f"Analisis: {analysis_path}")
    print(f"Grafico: {out_dir / 'model_metrics_comparison.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
