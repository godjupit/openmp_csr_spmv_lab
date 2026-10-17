#!/usr/bin/env python3
"""Compare the four CSR OpenMP strategies across thread count and matrix size."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/parameter_results')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('choose a new output directory to preserve previous measurements')
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / 'csr_spmv_lab.cpp'
    compiler = os.environ.get('CXX', '/opt/homebrew/opt/llvm/bin/clang++')
    binary = output / 'csr_spmv_lab'
    build = [compiler, '-O2', '-std=c++17', '-fopenmp', '-pthread', str(source), '-o', str(binary)]
    subprocess.run(build, check=True)
    (output / 'source.cpp').write_bytes(source.read_bytes())
    (output / 'source.sha256').write_text(hashlib.sha256(source.read_bytes()).hexdigest() + '\n')
    env = dict(os.environ, OMP_DYNAMIC='FALSE')
    metadata = {'build': build, 'runs': 10, 'repeats': 500, 'seed': 20260926,
                'cols': 100000, 'normal_nnz': 32, 'long_nnz': 20000,
                'runtime_environment': {k: v for k, v in env.items() if k.startswith(('OMP_', 'KMP_'))}}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    with (output / 'environment.txt').open('w') as f:
        for command in [['sw_vers'], ['sysctl', 'machdep.cpu.brand_string', 'hw.physicalcpu', 'hw.logicalcpu'],
                        [compiler, '--version'], ['otool', '-L', str(binary)]]:
            f.write(repr(command) + '\n' + subprocess.check_output(command, text=True) + '\n')
    # The 10000-row, 8-thread baseline belongs to both sweeps and is measured once.
    configs = [(10000, 32, t) for t in [1, 2, 4, 8, 10]] + [(2500, 8, 8), (40000, 128, 8)]
    random.Random(20261017).shuffle(configs)
    strategies = ['serial', 'omp1', 'omp2', 'omp3', 'omp4']
    results = []
    with (output / 'raw.csv').open('w', newline='') as raw_file, (output / 'stdout.txt').open('w') as logs:
        writer = csv.writer(raw_file)
        writer.writerow(['rows', 'cols', 'nnz', 'long_rows', 'threads', 'round', 'strategy', 'repeats', 'time_ms'])
        for row_count, long_rows, thread_count in configs:
            samples = {name: [] for name in strategies}
            print(f'Start rows={row_count}, long_rows={long_rows}, threads={thread_count}', flush=True)
            for run in range(10):
                offset = run % len(strategies)
                for name in strategies[offset:] + strategies[:offset]:
                    command = [str(binary), '--verbose', '--no-baseline', '--only', name, '--rows', str(row_count), '--cols', '100000',
                               '--normal-nnz', '32', '--long-rows', str(long_rows), '--long-nnz', '20000',
                               '--threads', str(thread_count), '--seed', '20260926', '--repeats', '500']
                    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
                    logs.write(repr(command) + '\n' + result.stdout + result.stderr + '\n')
                    logs.flush()
                    if ('All correctness checks passed.' not in result.stdout or
                            'OpenMP enabled' not in result.stdout):
                        raise RuntimeError(f'{name}: invalid benchmark output')
                    match = re.search(r'\btime_ms=([0-9.]+)', result.stdout)
                    nnz = int(re.search(r'\bnnz=(\d+)', result.stdout)[1])
                    elapsed = float(match[1])
                    samples[name].append(elapsed)
                    writer.writerow([row_count, 100000, nnz, long_rows, thread_count, run + 1, name, 500, elapsed])
                raw_file.flush()
            baseline = statistics.median(samples['serial'])
            for name, times in samples.items():
                median = statistics.median(times)
                speedup = baseline / median
                results.append({'rows': row_count, 'cols': 100000, 'nnz': nnz, 'long_rows': long_rows,
                                'threads': thread_count, 'strategy': name, 'runs': 10, 'repeats': 500,
                                'median_ms': median, 'min_ms': min(times), 'max_ms': max(times),
                                'mean_ms': statistics.mean(times), 'stdev_ms': statistics.stdev(times),
                                'speedup': speedup, 'efficiency': speedup / thread_count if name != 'serial' else ''})
                print(f'  {name}: {median:.6f} ms, {speedup:.3f}x', flush=True)
            with (output / 'summary.csv').open('w', newline='') as file:
                summary = csv.DictWriter(file, fieldnames=list(results[0]))
                summary.writeheader(); summary.writerows(results)
    print(f'All {len(configs) * 10 * len(strategies)} runs passed correctness checks. Saved to {output}', flush=True)


if __name__ == '__main__':
    main()
