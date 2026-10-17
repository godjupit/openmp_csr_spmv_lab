#!/usr/bin/env python3
"""Equal-work CPU and streaming-capacity benchmark for this Apple Silicon Mac."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/cpu_capacity_results')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error('choose a new output directory')
    out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    compiler = os.environ.get('CXX', '/opt/homebrew/opt/llvm/bin/clang++')
    binary = out / 'cpu_capacity'
    command = [compiler, '-O2', '-std=c++17', '-fopenmp', '-pthread',
               str(root / 'tools/cpu_capacity.cpp'), '-o', str(binary)]
    subprocess.run(command, check=True)
    for name in ['cpu_capacity.cpp', 'benchmark_cpu_capacity.py', 'report_cpu_capacity.py']:
        (out / name).write_bytes((root / 'tools' / name).read_bytes())
    env = dict(os.environ, OMP_DYNAMIC='FALSE')
    metadata = {'build': command, 'runs': 10, 'jobs': 120, 'iterations_per_job': 1000000,
                'compute_flops': 3840000000, 'flops_per_iteration': 32,
                'stream_elements': 12000000, 'stream_passes': 12, 'stream_logical_bytes': 3456000000,
                'environment': {k:v for k,v in env.items() if k.startswith(('OMP_', 'KMP_'))}}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    with (out / 'environment.txt').open('w') as f:
        for cmd in [['sw_vers'], [compiler, '--version'], ['sysctl', 'machdep.cpu.brand_string',
                    'hw.physicalcpu', 'hw.logicalcpu', 'hw.perflevel0.name', 'hw.perflevel0.physicalcpu',
                    'hw.perflevel1.name', 'hw.perflevel1.physicalcpu'], ['otool', '-L', str(binary)]]:
            f.write(repr(cmd) + '\n' + subprocess.check_output(cmd, text=True) + '\n')
    configs = [(mode,p) for mode in ['compute','stream'] for p in [1,2,4,6,8,10]]
    rng = random.Random(20261017)
    samples = []
    with (out / 'raw.jsonl').open('w') as f:
        for run in range(10):
            order = list(configs); rng.shuffle(order)
            for mode,p in order:
                result = subprocess.run([str(binary), '--mode', mode, '--threads', str(p)],
                                        env=env, capture_output=True, text=True, check=True)
                sample = json.loads(result.stdout)
                assert sample['correct'] and sample['threads']==p and sample['elapsed_ms']>0
                assert all(t['units']==sample['workers'][0]['units'] for t in sample['workers'])
                sample['run'] = run+1; samples.append(sample)
                f.write(json.dumps(sample)+'\n');f.flush()
            print(f'Completed {run+1}/10 rounds (compute + stream, 1/2/4/6/8/10 threads)',flush=True)
    summary = []
    for mode in ['compute','stream']:
        baseline = statistics.median(s['elapsed_ms'] for s in samples if s['mode']==mode and s['threads']==1)
        for p in [1,2,4,6,8,10]:
            selected = [s for s in samples if s['mode']==mode and s['threads']==p]
            times = [s['elapsed_ms'] for s in selected]
            elapsed = statistics.median(times)
            rate = selected[0]['work']/(elapsed*1e6)
            summary.append(dict(mode=mode,threads=p,median_ms=elapsed,min_ms=min(times),max_ms=max(times),
                                throughput=rate,unit='GFLOP/s' if mode=='compute' else 'logical GB/s',
                                speedup=baseline/elapsed,efficiency=baseline/elapsed/p,
                                average_busy_cores=statistics.median(sum(t['cpu_ms'] for t in s['workers'])/s['elapsed_ms'] for s in selected),
                                time_imbalance=statistics.median(max(t['active_ms'] for t in s['workers'])/min(t['active_ms'] for t in s['workers']) for s in selected)))
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    with (out/'workers.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['mode','threads','run','tid','units','active_ms','cpu_ms','cpu_samples'])
        for s in samples:
            for t in s['workers']:
                writer.writerow([s['mode'],s['threads'],s['run'],t['tid'],t['units'],t['active_ms'],t['cpu_ms'],json.dumps(t['cpu_samples'])])
    subprocess.run([sys.executable, str(root / 'tools/report_cpu_capacity.py'), str(out)], check=True)
    print(f'All {len(samples)} runs validated. Saved to {out}',flush=True)


if __name__=='__main__':main()
