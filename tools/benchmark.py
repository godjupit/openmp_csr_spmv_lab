#!/usr/bin/env python3
"""Compile and compare CSR SpMV implementations using independent runs."""

import argparse
import csv
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]

def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=positive_int, default=1000,
                        help="independent process runs per implementation (default: 1000)")
    parser.add_argument("--repeats", type=positive_int, default=1,
                        help="timed SpMV calls per process (default: 1)")
    parser.add_argument("--threads", type=positive_int, default=8)
    parser.add_argument("--strategies", nargs="+",
                        choices=["serial", "thread", "omp1", "omp2", "omp3", "omp4"],
                        default=["serial", "thread", "omp1", "omp2", "omp3", "omp4"])
    parser.add_argument("--output", type=Path, default=ROOT / "output/benchmark_results")
    parser.add_argument("matrix_args", nargs=argparse.REMAINDER,
                        help="additional matrix options after --")
    args = parser.parse_args()
    extra = args.matrix_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    if any(arg.split("=", 1)[0] in {"--only", "--threads", "--repeats", "--no-baseline", "--help", "-h"}
           for arg in extra):
        parser.error("pass only matrix options after --")
    strategies = list(dict.fromkeys(["serial"] + args.strategies))
    source = ROOT / "csr_spmv_lab.cpp"
    args.output.mkdir(parents=True, exist_ok=True)
    samples = {name: [] for name in strategies}
    env = dict(os.environ, OMP_DYNAMIC="FALSE")

    with tempfile.TemporaryDirectory(prefix="csr-benchmark-") as build:
        binary = Path(build) / "csr_spmv_lab"
        subprocess.run([os.environ.get("CXX", "g++"), "-O2", "-std=c++17", "-fopenmp", "-pthread",
                        str(source), "-o", str(binary)], check=True)
        print(f"{args.runs} runs per strategy, {args.repeats} timed calls per run, "
              f"{args.threads} threads", flush=True)
        with (args.output / "raw.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["round", "strategy", "threads", "repeats", "time_ms"])
            for round_index in range(args.runs):
                # Rotate execution order to reduce systematic order bias.
                offset = round_index % len(strategies)
                order = strategies[offset:] + strategies[:offset]
                for name in order:
                    result = subprocess.run(
                        [str(binary), "--verbose", "--no-baseline", "--only", name, "--threads", str(args.threads),
                         "--repeats", str(args.repeats)] + extra,
                        capture_output=True, text=True, check=True, env=env)
                    match = re.search(r"\btime_ms=([0-9.]+)", result.stdout)
                    if not match or "All correctness checks passed." not in result.stdout:
                        raise RuntimeError(f"Invalid benchmark output:\n{result.stdout}")
                    elapsed = float(match.group(1))
                    samples[name].append(elapsed)
                    writer.writerow([round_index + 1, name, args.threads, args.repeats, elapsed])
                file.flush()
                if (round_index + 1) % max(1, args.runs // 20) == 0:
                    print(f"Completed {round_index + 1}/{args.runs} rounds", flush=True)

    baseline_mean = statistics.mean(samples["serial"])
    baseline_median = statistics.median(samples["serial"])
    print(f"\n{'strategy':<12} {'mean/ms':>12} {'median/ms':>12} "
          f"{'mean speedup':>14} {'median speedup':>15}")
    with (args.output / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["strategy", "runs", "mean_ms", "median_ms",
                         "mean_speedup", "median_speedup"])
        for name in strategies:
            mean = statistics.mean(samples[name])
            median = statistics.median(samples[name])
            mean_speedup = baseline_mean / mean if mean > 0 else float("nan")
            median_speedup = baseline_median / median if median > 0 else float("nan")
            writer.writerow([name, args.runs, mean, median, mean_speedup, median_speedup])
            print(f"{name:<12} {mean:12.6f} {median:12.6f} "
                  f"{mean_speedup:13.3f}x {median_speedup:14.3f}x")
    print(f"\nSaved raw.csv and summary.csv to {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stderr or str(error))
