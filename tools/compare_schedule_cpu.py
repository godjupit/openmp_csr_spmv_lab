#!/usr/bin/env python3
"""Compare process CPU time and wall time for static and dynamic row scheduling."""
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
    parser.add_argument('--breakdown', action='store_true', help='estimate computation and overhead with calibrated per-row CPU timers')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/schedule_cpu')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('choose a new output directory')
    output.mkdir(parents=True)
    compiler = os.environ.get('CXX', '/opt/homebrew/opt/llvm/bin/clang++')
    binary = output / 'compare_schedule_cpu'
    build = [compiler, '-O2', '-std=c++17', '-fopenmp', '-pthread',
             str(ROOT / 'tools/compare_schedule_cpu.cpp'), '-o', str(binary)]
    subprocess.run(build, check=True)
    for name in ['csr_spmv_lab.cpp', 'tools/compare_schedule_cpu.cpp', 'tools/compare_schedule_cpu.py']:
        (output / Path(name).name).write_bytes((ROOT / name).read_bytes())
    env = dict(os.environ, OMP_DYNAMIC='FALSE')
    env.pop('CSR_CPU_BREAKDOWN', None)
    if args.breakdown:
        env['CSR_CPU_BREAKDOWN'] = '1'
    metadata = dict(vars(args), output=str(output), build=build, warmup=1,
                    cpu_clock='CLOCK_PROCESS_CPUTIME_ID',
                    environment={k: v for k, v in env.items() if k.startswith(('OMP_', 'KMP_'))})
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (output / 'environment.txt').write_text(subprocess.check_output([compiler, '--version'], text=True)
        + subprocess.check_output(['sw_vers'], text=True)
        + subprocess.check_output(['sysctl', 'machdep.cpu.brand_string'], text=True))
    strategies = [('serial', 'serial'), ('static', 'omp1'), ('dynamic,1', 'omp4')]
    samples = {name: [] for name, _ in strategies}
    with (output / 'raw.jsonl').open('w') as raw:
        for run in range(args.runs):
            offset = run % len(strategies)
            for name, selector in strategies[offset:] + strategies[:offset]:
                command = [str(binary), '--only', selector, '--threads', str(args.threads),
                           '--repeats', str(args.repeats)]
                result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
                sample = json.loads(result.stdout)
                assert sample['correct'] and sample['cpu_ms'] > 0 and sample['wall_ms'] > 0
                if args.breakdown and name != 'serial':
                    assert sample['rows_timed'] == 10000 * args.repeats
                    assert abs(sample['total_cpu_est_ms'] - sample['compute_cpu_est_ms']
                               - sample['overhead_cpu_est_ms']) < 1e-7
                sample.update(strategy=name, run=run + 1)
                samples[name].append(sample)
                raw.write(json.dumps(sample) + '\n')
                raw.flush()
    summary = []
    serial = statistics.median(s['wall_ms'] for s in samples['serial'])
    for name, _ in strategies:
        values = samples[name]
        wall = statistics.median(s['wall_ms'] for s in values)
        cpu = statistics.median(s['cpu_ms'] for s in values)
        # Use the median of paired ratios rather than a ratio of separate medians.
        cores = statistics.median(s['cpu_ms'] / s['wall_ms'] for s in values)
        summary.append(dict(strategy=name, wall_ms=wall, cpu_ms=cpu,
                            average_cores=cores, speedup=serial / wall))
    with (output / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [f'# static / dynamic CPU 时间（{args.threads} 线程）', '',
             f'{args.runs} 轮交替顺序运行；每次预热 1 次、检查正确性，再计时 {args.repeats} 次。全部检查通过。',
             '默认矩阵、固定种子；两种并行内核只改变调度方式。各指标为各轮单次平均值的中位数。', '',
             '| 策略 | 整体时间/ms | 累计 CPU 时间/ms | 平均占用核数 | 加速比 |',
             '|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['strategy']} | {row['wall_ms']:.6f} | {row['cpu_ms']:.6f} | "
                     f"{row['average_cores']:.2f} | {row['speedup']:.3f}× |")
    lines += ['', 'CPU 时间使用进程 CPU 时钟，累计所有线程实际消耗的 CPU 时间，包含乘加、调度、原子操作及忙等，'
              '不包含休眠等待；未拆出纯计算与开销。CPU 时间不能减去墙钟时间来计算开销。',
              '计时区间不包含矩阵构造、预热、正确性检查和输出。CPU 时钟仅在整批调用前后各读取一次。',
              '平均占用核数为每轮 CPU 时间/墙钟时间的中位数，衡量 CPU 占用，不等同于有效计算效率。',
              '未绑定核心；P/E 核调度、频率变化会影响结果。', '',
              '原始数据 `raw.jsonl`；汇总 `summary.csv`；本目录保留源码及环境快照。', '']
    report = '\n'.join(lines)
    if args.breakdown:
        # Use arithmetic means so total = computation + overhead remains additive.
        fields = ['cpu_ms', 'body_cpu_raw_ms', 'timer_interval_bias_ms', 'timer_cost_ms',
                  'total_cpu_est_ms', 'compute_cpu_est_ms', 'overhead_cpu_est_ms']
        breakdown_rows = []
        detail = ['', '## CPU 拆分估计（所有正式调用的算术平均）', '',
                  '| 策略 | 实测累计 CPU/ms（含插桩） | 校准后累计 CPU/ms | 计算 CPU 估计/ms | 开销 CPU 估计/ms |',
                  '|---|---:|---:|---:|---:|']
        for name in ['static', 'dynamic,1']:
            row = dict(strategy=name, **{key: statistics.mean(s[key] for s in samples[name]) for key in fields})
            overheads = [s['overhead_cpu_est_ms'] for s in samples[name]]
            row['overhead_min_ms'] = min(overheads)
            row['overhead_max_ms'] = max(overheads)
            row['negative_overhead_runs'] = sum(value < 0 for value in overheads)
            breakdown_rows.append(row)
            detail.append(f"| {name} | {row['cpu_ms']:.6f} | {row['total_cpu_est_ms']:.6f} | "
                          f"{row['compute_cpu_est_ms']:.6f} | {row['overhead_cpu_est_ms']:.6f} |")
        detail += ['', '每行乘加前后读取线程 CPU 时钟；两种策略使用相同的逐行计时粒度。',
                   '先在各工作线程测空计时间隔及整对计时成本：计算估计=行内时间总和−空区间偏差；'
                   '校准后累计 CPU=进程累计 CPU−计时器成本；开销估计=校准后累计 CPU−计算估计。',
                   '实测累计 CPU 时间来自插桩版本，不能用来报告原始策略性能。',
                   '校准仅扣除直接计时成本，无法消除插桩改变任务领取、线程竞争及缓存状态的影响。'
                   '计数、记录、并行启动、忙等也包含在剩余开销中，不能认定全部是调度开销。',
                   '这些是诊断版本的近似拆分，不能移用于未插桩程序；负估计值不截断，若出现说明校准误差过大。', '']
        for row in breakdown_rows:
            detail.append(f"{row['strategy']} 的各轮开销估计范围：{row['overhead_min_ms']:.6f}–"
                          f"{row['overhead_max_ms']:.6f} ms；负值 {row['negative_overhead_runs']}/{args.runs} 轮。"
                          + (' 开销接近校准误差，无法可靠量化。' if row['negative_overhead_runs'] else ''))
        with (output / 'breakdown.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(breakdown_rows[0]))
            writer.writeheader()
            writer.writerows(breakdown_rows)
        report = (f'# static / dynamic CPU 拆分估计（{args.threads} 线程）\n\n'
                  f'{args.runs} 轮，每轮预热 1 次并检查正确性，再计时 {args.repeats} 次；全部通过。\n'
                  '默认矩阵、固定种子。两种并行版本只改变调度方式，均对每行计时。\n'
                  + '\n'.join(detail)
                  + '\n\n原始数据 `raw.jsonl`；校准与拆分 `breakdown.csv`；本目录保留源码及环境快照。\n')
    (output / 'report.md').write_text(report)
    print(report)
    print(f'输出目录：{output}')


if __name__ == '__main__':
    main()
