#!/usr/bin/env python3
"""Compile and compare guided minimum chunks with the same CSR input and serial baseline."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = [1, 4, 8, 16, 32, 64, 128]


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=positive, default=4)
    parser.add_argument('--runs', type=positive, default=10)
    parser.add_argument('--repeats', type=positive, default=5000)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/guided_chunks')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('choose a new output directory')
    output.mkdir(parents=True)
    compiler = os.environ.get('CXX', '/opt/homebrew/opt/llvm/bin/clang++')
    binary = output / 'guided_chunks'
    command = [compiler, '-O2', '-std=c++17', '-fopenmp', '-pthread',
               str(ROOT / 'tools/guided_chunks.cpp'), '-o', str(binary)]
    subprocess.run(command, check=True)
    for name in ['csr_spmv_lab.cpp', 'tools/guided_chunks.cpp', 'tools/sweep_guided_chunks.py']:
        (output / Path(name).name).write_bytes((ROOT / name).read_bytes())
    env = dict(os.environ, OMP_DYNAMIC='FALSE')
    metadata = dict(vars(args), output=str(output), build=command, warmup=1, chunks=CHUNKS,
                    source_sha256=hashlib.sha256((ROOT / 'csr_spmv_lab.cpp').read_bytes()).hexdigest(),
                    matrix=dict(rows=10000, cols=100000, normal_nnz=32, long_rows=32,
                                long_nnz=20000, seed=20260926),
                    environment={k: v for k, v in env.items() if k.startswith(('OMP_', 'KMP_'))})
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    with (output / 'environment.txt').open('w') as file:
        for cmd in [[compiler, '--version'], ['sw_vers'],
                    ['sysctl', 'machdep.cpu.brand_string', 'hw.physicalcpu', 'hw.logicalcpu'],
                    ['otool', '-L', str(binary)]]:
            file.write(repr(cmd) + '\n' + subprocess.check_output(cmd, text=True) + '\n')
    samples = {chunk: [] for chunk in [0] + CHUNKS}
    rng = random.Random(20261017)
    with (output / 'raw.csv').open('w', newline='') as raw, (output / 'stdout.jsonl').open('w') as log:
        writer = csv.writer(raw)
        writer.writerow(['round', 'chunk', 'threads', 'warmup', 'repeats', 'time_ms'])
        for index in range(args.runs):
            order = list(samples)
            rng.shuffle(order)
            for chunk in order:
                command = [str(binary), str(chunk), '--threads', str(args.threads), '--repeats', str(args.repeats)]
                result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
                row = json.loads(result.stdout)
                assert row['correct'] and row['chunk'] == chunk and row['time_ms'] > 0
                samples[chunk].append(row['time_ms'])
                writer.writerow([index + 1, chunk, args.threads, 1, args.repeats, row['time_ms']])
                log.write(json.dumps(dict(command=command, round=index + 1, **row)) + '\n')
            raw.flush()
            log.flush()
            print(f'完成 {index + 1}/{args.runs} 轮', flush=True)
    serial = statistics.median(samples[0])
    summary = []
    for chunk, times in samples.items():
        median = statistics.median(times)
        summary.append(dict(chunk=chunk, threads=args.threads, runs=args.runs, repeats=args.repeats,
                            median_ms=median, min_ms=min(times), max_ms=max(times),
                            speedup=serial / median,
                            efficiency=serial / median / args.threads if chunk else ''))
    with (output / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [f'guided chunk 参数实验：{args.threads} 线程，串行基线 {serial:.6f} ms', '',
             '| 最小 chunk | 中位耗时/ms | 最小–最大/ms | 加速比 | 并行效率 |',
             '|---:|---:|---:|---:|---:|']
    for row in summary[1:]:
        lines.append(f"| {row['chunk']} | {row['median_ms']:.6f} | {row['min_ms']:.6f}–{row['max_ms']:.6f} | "
                     f"{row['speedup']:.3f}× | {row['efficiency']:.2%} |")
    report = '\n'.join(lines) + '\n'
    (output / 'summary.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
