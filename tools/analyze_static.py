#!/usr/bin/env python3
"""Analyze static row ownership, work counts and timings for a given thread count."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]

def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=positive_int, default=8)
    parser.add_argument('--runs', type=positive_int, default=10)
    parser.add_argument('--repeats', type=positive_int, default=500)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/static_analysis')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('choose a new output directory; existing results are preserved')
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    compiler = os.environ.get('CXX', '/opt/homebrew/opt/llvm/bin/clang++')
    command = [compiler, '-O2', '-std=c++17', '-fopenmp', '-pthread',
               str(root / 'tools/analyze_static.cpp'), '-o', str(output / 'analyze_static')]
    subprocess.run(command, check=True)
    env = dict(os.environ, OMP_DYNAMIC='FALSE')
    metadata = {'build': command, 'runs': args.runs, 'repeats': args.repeats, 'threads': args.threads,
                'rows': 10000, 'cols': 100000, 'normal_nnz': 32, 'long_rows': 32,
                'long_nnz': 20000, 'seed': 20260926,
                'environment': {k: v for k, v in env.items() if k.startswith(('OMP_', 'KMP_'))}}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (output / 'source.cpp').write_bytes((root / 'csr_spmv_lab.cpp').read_bytes())
    for name in ['analyze_static.cpp', 'analyze_static.py']:
        (output / name).write_bytes((root / 'tools' / name).read_bytes())
    with (output / 'environment.txt').open('w') as file:
        for cmd in [[compiler, '--version'], ['sw_vers'],
                    ['sysctl', 'machdep.cpu.brand_string', 'hw.physicalcpu', 'hw.logicalcpu']]:
            file.write(repr(cmd) + '\n' + subprocess.check_output(cmd, text=True) + '\n')
    runs = []
    with (output / 'raw.jsonl').open('w') as logs:
        for index in range(args.runs):
            result = subprocess.run([str(output / 'analyze_static'), '--threads', str(args.threads), '--repeats', str(args.repeats)],
                                    capture_output=True, text=True, check=True, env=env)
            sample = json.loads(result.stdout)
            assert sample['actual_threads'] == args.threads and sample['correct']
            assert sum(t['rows'] for t in sample['threads']) == 10000
            assert sum(t['long_rows'] for t in sample['threads']) == 32
            assert sum(t['nnz'] for t in sample['threads']) == sample['matrix_nnz'] == 958976
            assert all(t['compute_ms'] >= 0 and t['barrier_ms'] >= 0 for t in sample['threads'])
            sample['run'] = index + 1
            runs.append(sample)
            logs.write(json.dumps(sample) + '\n'); logs.flush()
            print(f'Completed {index + 1}/{args.runs}', flush=True)
    serial = statistics.median(r['serial_ms'] for r in runs)
    parallel = statistics.median(r['static_ms'] for r in runs)
    summary = []
    for tid in range(args.threads):
        samples = [r['threads'][tid] for r in runs]
        first = samples[0]
        keys = ['rows', 'first_row', 'last_row', 'long_rows', 'nnz']
        assert all(all(s[k] == first[k] for k in keys) for s in samples)
        summary.append(dict(tid=tid, **{k: first[k] for k in keys},
                            nnz_share=first['nnz'] / 958976,
                            compute_ms=statistics.median(s['compute_ms'] for s in samples),
                            barrier_ms=statistics.median(s['barrier_ms'] for s in samples)))
    with (output / 'threads.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    with (output / 'timing.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['strategy', 'median_ms', 'min_ms', 'max_ms', 'speedup', 'efficiency'])
        for name, key in [('serial', 'serial_ms'), ('omp4_static', 'static_ms')]:
            values = [r[key] for r in runs]; median = statistics.median(values)
            writer.writerow([name, median, min(values), max(values), serial / median,
                             serial / median / args.threads if name != 'serial' else ''])
    heavy = max(summary, key=lambda t: t['nnz'])
    light = min(summary, key=lambda t: t['nnz'])
    imbalance = heavy['nnz'] / (958976 / args.threads)
    row_counts = [t['rows'] for t in summary]
    row_description = (f'各线程均分到 {row_counts[0]} 行' if len(set(row_counts)) == 1
                       else f'各线程分到 {min(row_counts)}–{max(row_counts)} 行')
    lines = [f'# static 策略的 {args.threads} 线程运行分析', '',
             f'默认矩阵：10000×100000，958976 个非零元素，32 个长行各 20000 个，普通行各 32 个，种子 20260926。',
             f'编译选项：`-O2 -std=c++17 -fopenmp -pthread`；{args.threads} 线程，未显式绑定核心。',
             f'独立启动 {args.runs} 次进程，每次预热后计时 {args.repeats} 次调用，全部正确性检查通过。', '',
             f'原始串行中位耗时 **{serial:.6f} ms**，原始 static 中位耗时 **{parallel:.6f} ms**；',
             f'加速比 **{serial / parallel:.3f}×**，效率 **{serial / parallel / args.threads:.2%}**。', '',
             '| 线程 | 行范围（含两端） | 行数 | 长行数 | 非零元素数 | 工作量占比 | 计算/ms | 末尾屏障阶段/ms |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for t in summary:
        lines.append(f"| {t['tid']} | {t['first_row']}–{t['last_row']} | {t['rows']} | {t['long_rows']} | "
                     f"{t['nnz']} | {t['nnz_share']:.2%} | {t['compute_ms']:.6f} | {t['barrier_ms']:.6f} |")
    lines += ['',
              f"{row_description}，最重线程 {heavy['tid']} 处理 {heavy['nnz']} 个非零元素，",
              f"最轻线程 {light['tid']} 处理 {light['nnz']} 个，二者工作量相差 {heavy['nnz'] / light['nnz']:.2f} 倍。",
              f'最重线程工作量为平均值的 **{imbalance:.2f} 倍**，说明按行均分不能保证计算量均衡。', '',
              '非零元素数近似表示乘加工作量。线程计算时间是离开起始屏障后、结束行循环前的墙钟耗时，',
              '含可能的调度延迟，不是纯 CPU 时间；各列取各轮单次平均值的中位数。',
              '原始 static 总耗时调用现有策略 4 测量，不包含逐线程统计。线程计时来自单独诊断循环，',
              '使用常驻线程组、计时器和显式屏障，包含插桩影响，不能相加后当作原始总耗时。',
              '末尾屏障阶段包含等待、同步和可能的调度延迟，不是纯空闲时间。',
              '本机性能核和能效核混合，实际时间还受核心调度与访存影响，不能仅用非零元素数解释全部差异。', '',
              '原始数据：[raw.jsonl](raw.jsonl)；逐线程汇总：[threads.csv](threads.csv)；',
              '原始实现计时：[timing.csv](timing.csv)。代码和分析使用 AI 辅助。', '']
    report = '\n'.join(lines)
    (output / 'report.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
