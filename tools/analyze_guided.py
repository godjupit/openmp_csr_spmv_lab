#!/usr/bin/env python3
"""Build and measure guided scheduling, preserving raw per-thread measurements."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=positive, default=4)
    parser.add_argument('--runs', type=positive, default=10)
    parser.add_argument('--repeats', type=positive, default=500)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/guided_analysis')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('choose a new output directory to preserve existing results')
    output.mkdir(parents=True)
    compiler = os.environ.get('CXX', '/opt/homebrew/opt/llvm/bin/clang++')
    build = [compiler, '-O2', '-std=c++17', '-fopenmp', '-pthread',
             str(ROOT / 'tools/analyze_guided.cpp'), '-o', str(output / 'analyze_guided')]
    subprocess.run(build, check=True)
    for name in ['csr_spmv_lab.cpp', 'tools/analyze_guided.cpp', 'tools/analyze_guided.py']:
        (output / Path(name).name).write_bytes((ROOT / name).read_bytes())
    env = dict(os.environ, OMP_DYNAMIC='FALSE')
    metadata = dict(vars(args), output=str(output), build=build, warmup=1,
                    matrix=dict(rows=10000, cols=100000, normal_nnz=32,
                                long_rows=32, long_nnz=20000, seed=20260926),
                    environment={k: v for k, v in env.items() if k.startswith(('OMP_', 'KMP_'))})
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (output / 'environment.txt').write_text(subprocess.check_output([compiler, '--version'], text=True)
        + subprocess.check_output(['sw_vers'], text=True)
        + subprocess.check_output(['sysctl', 'machdep.cpu.brand_string'], text=True))
    runs = []
    with (output / 'raw.jsonl').open('w') as raw:
        for index in range(args.runs):
            result = subprocess.run([str(output / 'analyze_guided'), '--threads', str(args.threads),
                                     '--repeats', str(args.repeats)],
                                    env=env, capture_output=True, text=True, check=True)
            sample = json.loads(result.stdout)
            assert sample['correct'] and len(sample['threads']) == args.threads
            for key, total in [('rows', 10000), ('long_rows', 32), ('nnz', 958976)]:
                assert sum(t[key] for t in sample['threads']) == total * args.repeats
            assert all(t['work_ms'] >= 0 and t['barrier_ms'] >= 0 for t in sample['threads'])
            sample['run'] = index + 1
            runs.append(sample)
            raw.write(json.dumps(sample) + '\n')
            raw.flush()
    # Arithmetic means keep workload totals additive across threads and runs.
    summary = []
    for tid in range(args.threads):
        samples = [r['threads'][tid] for r in runs]
        row = dict(tid=tid)
        for key in ['rows', 'long_rows', 'nnz']:
            row[key] = statistics.mean(s[key] / args.repeats for s in samples)
        row['share'] = row['nnz'] / 958976
        for key in ['work_ms', 'barrier_ms']:
            row[key] = statistics.mean(s[key] for s in samples)
        summary.append(row)
    with (output / 'threads.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    serial = statistics.median(r['serial_ms'] for r in runs)
    guided = statistics.median(r['guided_ms'] for r in runs)
    lines = [f'# guided {args.threads} 线程分析', '',
             f'{args.runs} 轮，每轮各预热 1 次、正式计时 {args.repeats} 次；正确性检查通过。',
             f'原始串行 {serial:.6f} ms，guided {guided:.6f} ms（各轮平均耗时的中位数）；'
             f'加速比 {serial/guided:.3f}×，并行效率 {serial/guided/args.threads:.2%}。', '',
             '| 线程 | 平均行数 | 平均长行数 | 平均非零元素数 | 工作量占比 | 工作阶段/ms | 屏障阶段/ms |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['tid']} | {s['rows']:.1f} | {s['long_rows']:.2f} | {s['nnz']:.1f} | "
                     f"{s['share']:.2%} | {s['work_ms']:.6f} | {s['barrier_ms']:.6f} |")
    lines += ['', '表内为所有正式调用的算术平均值。guided 每次动态分配行，平均长行数可以为小数。',
              '非零元素数近似表示乘加工作量；不代表 CPU 利用率。',
              '逐线程时间来自单独插桩的常驻线程组：工作阶段包含计算、guided 调度及系统抢占；'
              '屏障阶段包含等待和同步。二者为墙钟时间，不能相加为原始策略总耗时。',
              '原始策略整体计时不含逐线程统计。诊断循环另预热 1 次并检查正确性，再统计正式调用。',
              '当前未绑定核心，这些结果不能确定线程运行在 P 核还是 E 核。', '',
              '原始数据：`raw.jsonl`；逐线程数据：`threads.csv`；参数与源码快照均保存在本目录。', '']
    report = '\n'.join(lines)
    (output / 'report.md').write_text(report)
    print(report)
    print(f'输出目录：{output}')


if __name__ == '__main__':
    main()
