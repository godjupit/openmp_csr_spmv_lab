#!/usr/bin/env python3
"""Compare guided,16 scalar and SIMD, keeping the scheduling parameter fixed."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'csr_spmv_lab.cpp')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/simd_ablation')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5000)
    args = parser.parse_args()
    if min(args.threads, args.runs, args.repeats) < 1:
        parser.error('counts must be positive')
    source = args.source.resolve()
    # The production SIMD strategy must have the parameter this experiment controls.
    body = source.read_text().split('static void spmv_omp_student_2', 1)[1].split('static void spmv_omp_student_3', 1)[0]
    if 'schedule(guided, 16)' not in body or 'omp simd reduction(+:sum)' not in body:
        parser.error('this ablation requires production omp2 = guided,16 + SIMD')
    output = args.output.resolve()
    if output.exists():
        parser.error('choose a new output directory')
    output.mkdir(parents=True)
    compiler = os.environ.get('CXX', '/opt/homebrew/opt/llvm/bin/clang++')
    flags = ['-O2', '-std=c++17', '-fopenmp', '-pthread']
    commands = [[compiler] + flags + [str(source), '-o', str(output / 'simd')],
                [compiler] + flags + ['-DCSR_SOURCE_PATH="' + str(source) + '"',
                                     str(ROOT / 'tools/guided_chunks.cpp'), '-o', str(output / 'scalar')]]
    for command in commands:
        subprocess.run(command, check=True)
    (output / 'source.cpp').write_bytes(source.read_bytes())
    (output / 'guided_chunks.cpp').write_bytes((ROOT / 'tools/guided_chunks.cpp').read_bytes())
    (output / 'metadata.json').write_text(json.dumps(dict(build=commands, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        threads=args.threads, runs=args.runs, repeats=args.repeats, warmup=1, schedule='guided,16'), indent=2))
    env = dict(os.environ, OMP_DYNAMIC='FALSE')
    samples = {'scalar': [], 'simd': []}
    with (output / 'raw.csv').open('w', newline='') as raw, (output / 'stdout.jsonl').open('w') as log:
        writer = csv.writer(raw)
        writer.writerow(['round', 'mode', 'warmup', 'repeats', 'time_ms'])
        for index in range(args.runs):
            for mode in (['scalar', 'simd'] if index % 2 == 0 else ['simd', 'scalar']):
                command = [str(output / mode)]
                command += ['16'] if mode == 'scalar' else ['--only', 'omp2', '--verbose', '--no-baseline']
                command += ['--threads', str(args.threads), '--repeats', str(args.repeats)]
                result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
                if mode == 'scalar':
                    data = json.loads(result.stdout)
                    assert data['correct']
                    elapsed = data['time_ms']
                else:
                    assert 'All correctness checks passed.' in result.stdout
                    elapsed = float(re.search(r'\btime_ms=([0-9.]+)', result.stdout)[1])
                samples[mode].append(elapsed)
                writer.writerow([index + 1, mode, 1, args.repeats, elapsed])
                log.write(json.dumps(dict(command=command, stdout=result.stdout)) + '\n')
            raw.flush()
            log.flush()
    with (output / 'summary.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mode', 'median_ms', 'min_ms', 'max_ms'])
        for mode, times in samples.items():
            writer.writerow([mode, statistics.median(times), min(times), max(times)])
    scalar = statistics.median(samples['scalar'])
    simd = statistics.median(samples['simd'])
    print(f'guided,16 scalar: {scalar:.6f} ms; SIMD: {simd:.6f} ms; reduction: {1-simd/scalar:.2%}')


if __name__ == '__main__':
    main()
