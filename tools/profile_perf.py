#!/usr/bin/env python3
"""PDF section 3.5: compare serial, OpenMP 1 and std::thread with Linux perf."""
import argparse
import csv
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=positive_int, default=8)
    parser.add_argument('--runs', type=positive_int, default=10,
                        help='independent perf repetitions (PDF requires at least 10)')
    parser.add_argument('--repeats', type=positive_int, default=1000,
                        help='SpMV calls per process, excluding one warm-up')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/perf_omp1_results')
    args = parser.parse_args()
    if not 10 <= args.runs <= 100:
        parser.error('use 10 to 100 repetitions (PDF minimum 10; perf stat maximum 100)')
    if platform.system() != 'Linux':
        parser.error('Linux perf is required. Run this script in Linux or WSL, not macOS.')
    compiler = os.environ.get('CXX', 'g++')
    for command in (compiler, 'perf', 'lscpu'):
        if not shutil.which(command):
            parser.error(f'{command} is unavailable; install it in the Linux environment first')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Keep previous measurements intact, and avoid mixing incomplete and complete runs.
    if any(output.iterdir()):
        parser.error(f'{output} is not empty; choose a new --output directory')
    root = Path(__file__).resolve().parents[1]
    source = root / 'csr_spmv_lab.cpp'
    binary = output / 'csr_spmv_lab'
    shutil.copy2(source, output / source.name)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE',
               OMP_PROC_BIND='close', OMP_PLACES='cores', LC_ALL='C')

    def run(command):
        return subprocess.run(command, env=env, capture_output=True, text=True)

    for name, command in [
        ('lscpu.txt', ['lscpu']), ('uname.txt', ['uname', '-a']),
        ('perf-version.txt', ['perf', '--version']),
        ('perf-list.txt', ['perf', 'list']), ('compiler.txt', [compiler, '--version']),
    ]:
        result = run(command)
        (output / name).write_text(result.stdout + result.stderr)
        result.check_returncode()
    build = [compiler, '-O2', '-g', '-std=c++17', '-fopenmp', '-pthread',
             str(source), '-o', str(binary)]
    result = run(build)
    (output / 'build.txt').write_text(repr(build) + '\n' + result.stdout + result.stderr)
    result.check_returncode()
    runtime = run(['ldd', str(binary)])
    (output / 'runtime.txt').write_text(runtime.stdout + runtime.stderr)
    runtime.check_returncode()
    metadata = {k: env[k] for k in ('OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_PROC_BIND', 'OMP_PLACES')}
    (output / 'environment.txt').write_text(
        f'threads={args.threads}\nruns={args.runs}\nrepeats={args.repeats}\n'
        'matrix=default; rows=10000 cols=100000 normal_nnz=32 long_rows=32 '
        'long_nnz=20000 seed=20260926\n' +
        '\n'.join(f'{k}={v}' for k, v in metadata.items()) + '\n')

    events = []
    with (output / 'event-probes.txt').open('w') as file:
        for event in ['task-clock', 'cycles', 'instructions', 'cache-misses',
                      'branch-misses', 'context-switches', 'cpu-migrations', 'page-faults']:
            result = run(['perf', 'stat', '-e', event, '--', 'true'])
            file.write(f'event={event} exit={result.returncode}\n{result.stdout}{result.stderr}\n')
            if result.returncode == 0 and not re.search(
                    r'not supported|not counted|<not', result.stderr, re.I):
                events.append(event)
    if not events:
        raise RuntimeError('No perf events are available. See event-probes.txt for permission '
                           'or kernel errors; rerun after configuring perf access.')
    (output / 'events.txt').write_text('\n'.join(events) + '\n')
    print('Available perf events:', ', '.join(events), flush=True)
    samples = {}
    with (output / 'commands.txt').open('w') as commands:
        for name in ['serial', 'omp1', 'thread']:
            command = ['perf', 'stat', '-r', str(args.runs), '-e', ','.join(events),
                       '-o', str(output / f'{name}.stat.txt'), '--', str(binary),
                       '--verbose', '--no-baseline', '--only', name, '--threads', str(args.threads),
                       '--repeats', str(args.repeats)]
            commands.write(repr(command) + '\n'); commands.flush()
            print(f'Measuring {name} ({args.runs} processes)...', flush=True)
            result = run(command)
            (output / f'{name}.stdout.txt').write_text(result.stdout)
            (output / f'{name}.stderr.txt').write_text(result.stderr)
            result.check_returncode()
            times = [float(t) for t in re.findall(r'\btime_ms=([0-9.]+)', result.stdout)]
            if (len(times) != args.runs or
                    result.stdout.count('All correctness checks passed.') != args.runs or
                    result.stdout.count('OpenMP enabled') != args.runs):
                raise RuntimeError(f'{name}: missing measurements or correctness checks')
            samples[name] = times

    baseline = statistics.median(samples['serial'])
    lines = ['# PDF 3.5：serial、OpenMP 1、std::thread 的 perf 比较', '',
             f'每种实现 {args.runs} 次独立进程，每个进程预热一次、计时 {args.repeats} 次调用。',
             '矩阵使用默认参数与固定种子；全部正确性检查通过。', '',
             '| 实现 | 中位耗时/ms | 最小/ms | 最大/ms | 加速比 | 并行效率 |',
             '|---|---:|---:|---:|---:|---:|']
    with (output / 'summary.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['strategy', 'runs', 'median_ms', 'min_ms', 'max_ms',
                         'speedup_serial_median', 'parallel_efficiency'])
        for name, times in samples.items():
            median = statistics.median(times)
            speedup = baseline / median
            efficiency = speedup / args.threads if name != 'serial' else None
            writer.writerow([name, len(times), median, min(times), max(times), speedup, efficiency])
            efficiency_text = f'{efficiency:.2%}' if efficiency is not None else '—'
            lines.append(f'| {name} | {median:.6f} | {min(times):.6f} | {max(times):.6f} '
                         f'| {speedup:.3f} | {efficiency_text} |')
    lines += ['', '加速比以串行中位数为 T1，效率为加速比 / 请求线程数。',
              'perf 的原始计数及重复测量波动见各实现的 `.stat.txt`。',
              'chrono 只计 SpMV 调用；perf 统计整个进程，包括矩阵生成、参考计算、',
              '预热、校验和输出。不能直接将进程级计数视为 SpMV 循环的计数。',
              'OpenMP 使用 close/cores 绑定；std::thread 未显式绑定。',
              '三个实现依次测量，顺序、频率与系统负载可能影响结果。',
              '未采集到的硬件事件不能用于推断 IPC、缓存未命中率或分支失误率。',
              'task-clock / elapsed 得到的 CPU 利用量不等于并行效率。',
              '请结合原始 perf 数据分析，不应仅凭运行时间认定缓存或带宽瓶颈。', '']
    (output / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'Saved evidence to {output}')


if __name__ == '__main__':
    try:
        main()
    except (subprocess.CalledProcessError, RuntimeError) as error:
        sys.exit(str(error))
