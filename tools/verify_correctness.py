#!/usr/bin/env python3
"""Compile the submission source and verify every strategy on eight configurations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'csr_spmv_lab.cpp')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/correctness_final')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('choose a new output directory')
    output.mkdir(parents=True)
    source = args.source.resolve()
    compiler = os.environ.get('CXX', '/opt/homebrew/opt/llvm/bin/clang++')
    binary = output / 'csr_spmv_lab'
    subprocess.run([compiler, '-O2', '-std=c++17', '-fopenmp', '-pthread', str(source), '-o', str(binary)], check=True)
    (output / 'source.cpp').write_bytes(source.read_bytes())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    print('CSR SpMV - correctness verification', flush=True)
    print('UTC: ' + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    print('Source SHA256: ' + digest[:16], flush=True)
    print('Reference: serial; checked: std::thread, omp1, omp2, omp3', flush=True)
    print('Full output comparison: absolute/relative tolerance = 1e-10', flush=True)
    print('Each strategy: 1 warm-up, correctness check, 10 timed calls.', flush=True)
    print('These short-run timings are not used for performance conclusions.\n', flush=True)
    cases = [(f'default matrix, {p:2d} thread(s)', ['--threads', str(p)]) for p in [1, 2, 4, 8, 10]]
    cases += [
        ('65 rows / block tail', ['--threads', '4', '--rows', '65', '--cols', '100', '--long-rows', '0', '--long-nnz', '0']),
        ('empty rows / zero nnz', ['--threads', '4', '--rows', '10', '--cols', '100', '--normal-nnz', '0', '--long-rows', '0', '--long-nnz', '0']),
        ('single long row', ['--threads', '4', '--rows', '1', '--cols', '100', '--long-rows', '1', '--long-nnz', '100'])]
    results = []
    for index, (label, extra) in enumerate(cases, 1):
        command = [str(binary), '--verbose', '--repeats', '10'] + extra
        result = subprocess.run(command, env=dict(os.environ, OMP_DYNAMIC='FALSE'), capture_output=True, text=True, check=True)
        (output / f'case_{index}.txt').write_text(result.stdout + result.stderr)
        if 'All correctness checks passed.' not in result.stdout or len(re.findall(r'^\| omp[123] ', result.stdout, re.M)) != 3:
            raise RuntimeError(f'Incomplete correctness output: {label}')
        results.append(dict(case=label, command=command, passed=True))
        print(f'{index:2d}. {label:<35} PASS', flush=True)
    print(f'\nALL {len(results)} CONFIGURATIONS PASSED (4 parallel implementations each).', flush=True)
    print('Every strategy compared all y[row] entries against the serial result.', flush=True)
    print('Full per-case logs and source snapshot saved beside this result.', flush=True)
    (output / 'results.json').write_text(json.dumps(dict(source_sha256=digest, warmup=1, repeats=10, cases=results), indent=2))


if __name__ == '__main__':
    main()
