#!/usr/bin/env python3
"""Measure static/guided/dynamic with CPU Counters and thread diagnostics on this M4 Mac."""
import argparse
import csv
import json
import os
from pathlib import Path
import plistlib
import statistics
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ['Cycles', 'Instructions', 'L1D_CACHE_MISS_LD', 'LD_UNIT_UOP']
STRATEGIES = [('static', 'omp1'), ('guided', 'omp2'), ('dynamic', 'omp4')]


def table_rows(path):
    ids = {}

    def decode(e):
        if e.get('ref'):
            return ids[e.get('ref')]
        result = (e.tag, e.text or '', e.get('fmt', ''), tuple(decode(c) for c in e))
        if e.get('id'):
            ids[e.get('id')] = result
        return result

    for _, element in ET.iterparse(path, events=('end',)):
        if element.tag == 'row':
            yield [decode(c) for c in element]
            element.clear()


def summarize_counters(counters, signposts, repeats):
    intervals = [r for r in table_rows(signposts) if r[3][1] == 'MeasuredSpMV']
    if len(intervals) != 1:
        raise RuntimeError('Expected exactly one measured signpost interval')
    interval = intervals[0]
    begin = int(interval[0][1])
    end = begin + int(interval[1][1])
    process = interval[7][2]
    totals = [0] * len(EVENTS)
    cores = {}
    seen = set()
    samples = 0
    for row in table_rows(counters):
        if row[2][2] != process or row[7][0] != 'pmc-events':
            continue
        timestamp = int(row[0][1])
        # Omit boundary samples; the 1ms sample period can straddle warm-up.
        if not begin + 1_000_000 <= timestamp <= end:
            continue
        key = (timestamp, row[1][2], row[3][2])
        if key in seen:
            continue
        seen.add(key)
        values = list(map(int, row[7][1].split()))
        if len(values) != len(EVENTS):
            raise RuntimeError('Counter order/length mismatch')
        totals = [a + b for a, b in zip(totals, values)]
        core = 'P' if 'P Core' in row[3][2] else 'E'
        cores.setdefault(core, [0] * len(EVENTS))
        cores[core] = [a + b for a, b in zip(cores[core], values)]
        samples += 1
    if not totals[0] or not totals[3]:
        raise RuntimeError('No usable CPU counter samples')
    return dict(events=EVENTS, totals=totals, by_core_type=cores, samples=samples,
                begin_ns=begin, end_ns=end,
                cycles_per_call=totals[0] / repeats,
                instructions_per_call=totals[1] / repeats,
                ipc=totals[1] / totals[0],
                l1d_load_misses_per_call=totals[2] / repeats,
                l1d_load_miss_fraction=totals[2] / totals[3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--runs', type=int, default=3, help='hardware traces per strategy')
    parser.add_argument('--repeats', type=int, default=10000, help='calls per hardware trace')
    parser.add_argument('--timing-runs', type=int, default=10)
    parser.add_argument('--timing-repeats', type=int, default=500)
    parser.add_argument('--summarize-only', action='store_true', help='rebuild report from saved raw measurements')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/schedule_metrics')
    args = parser.parse_args()
    if min(args.threads, args.runs, args.repeats, args.timing_runs, args.timing_repeats) <= 0:
        parser.error('counts must be positive')
    output = args.output.resolve()
    if args.summarize_only:
        metadata = json.loads((output / 'metadata.json').read_text())
        for key in ['threads', 'runs', 'repeats', 'timing_runs', 'timing_repeats']:
            setattr(args, key, metadata[key])
        diagnostics = {name: [] for name, _ in STRATEGIES}
        for line in (output / 'threads.raw.jsonl').read_text().splitlines():
            row = json.loads(line)
            diagnostics[row['strategy']].append(row)
        counters = {name: [json.loads((output / f'{name}.{i + 1}.summary.json').read_text())
                           for i in range(args.runs)] for name, _ in STRATEGIES}
        write_summary(args, output, diagnostics, counters)
        return
    if output.exists():
        parser.error('choose a new output directory')
    output.mkdir(parents=True)
    env = dict(os.environ, DEVELOPER_DIR=os.environ.get('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer'),
               OMP_DYNAMIC='FALSE')
    env.pop('CSR_CPU_BREAKDOWN', None)
    xcode = Path(env['DEVELOPER_DIR']).parent
    shared = xcode / 'SharedFrameworks'
    frameworks = xcode / 'Applications/Instruments.app/Contents/Frameworks'
    env['CSR_INSTRUMENTS_PLUGIN'] = str(frameworks / 'InstrumentsPlugIn.framework/Versions/A/InstrumentsPlugIn')
    compiler = env.get('CXX', '/opt/homebrew/opt/llvm/bin/clang++')
    commands = []

    def run(command):
        commands.append(list(map(str, command)))
        (output / 'commands.json').write_text(json.dumps(commands, indent=2))
        result = subprocess.run(list(map(str, command)), env=env, capture_output=True, text=True)
        with (output / 'tool.log').open('a') as log:
            log.write(json.dumps(commands[-1]) + '\n' + result.stdout + result.stderr + '\n')
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    run(['python3', ROOT / 'tools/compare_schedule_cpu.py', '--threads', args.threads,
         '--runs', args.timing_runs, '--repeats', args.timing_repeats, '--output', output / 'timing'])
    driver = output / 'schedule_threads'
    run([compiler, '-O2', '-g', '-std=c++17', '-fopenmp', '-pthread',
         ROOT / 'tools/analyze_schedules.cpp', '-o', driver])
    diagnostics = {name: [] for name, _ in STRATEGIES}
    with (output / 'threads.raw.jsonl').open('w') as raw:
        for index in range(args.timing_runs):
            order = STRATEGIES[index % 3:] + STRATEGIES[:index % 3]
            for name, selector in order:
                data = json.loads(run([driver, '--only', selector, '--threads', args.threads,
                                       '--repeats', args.timing_repeats]))
                assert data['correct'] and len(data['threads']) == args.threads
                for key, total in [('rows', 10000), ('long_rows', 32), ('nnz', 958976)]:
                    assert sum(t[key] for t in data['threads']) == total * args.timing_repeats
                diagnostics[name].append(data)
                raw.write(json.dumps(dict(strategy=name, run=index + 1, **data)) + '\n')
    print('CPU 时间、上下文切换和逐线程诊断已完成，开始硬件计数采集。', flush=True)
    helper = output / 'counter_options'
    run(['/usr/bin/clang', '-framework', 'Foundation', f'-Wl,-rpath,{shared}',
         f'-Wl,-rpath,{frameworks}', ROOT / 'tools/counter_options.m', '-o', helper])
    options = json.loads(run(['xcrun', 'xctrace', 'record', '--template', 'CPU Counters', '--show-recording-options']))
    options['CPU Counters']['configurationType'] = dict(manual={})
    options['CPU Counters']['allEventsAndFormulas'] = json.loads(run([helper] + EVENTS))
    options['CPU Counters']['countingLevel'] = 'EL0'
    options_path = output / 'counter-options.json'
    options_path.write_text(json.dumps(options, indent=2))
    counters = {name: [] for name, _ in STRATEGIES}
    binary = output / 'timing/compare_schedule_cpu'
    for index in range(args.runs):
        order = STRATEGIES[index % 3:] + STRATEGIES[:index % 3]
        for name, selector in order:
            prefix = output / f'{name}.{index + 1}'
            trace = prefix.with_suffix(prefix.suffix + '.trace')
            stdout = Path(str(prefix) + '.stdout.json')
            print(f'采集 {name}：{index + 1}/{args.runs}', flush=True)
            run(['xcrun', 'xctrace', 'record', '--template', 'CPU Counters',
                 '--recording-options', options_path, '--output', trace, '--time-limit', '60s',
                 '--target-stdout', stdout, '--env', 'OMP_DYNAMIC=FALSE',
                 '--launch', '--', binary, '--only', selector, '--threads', args.threads,
                 '--repeats', args.repeats])
            result = json.loads(stdout.read_text())
            assert result['correct'] and result['repeats'] == args.repeats
            for schema, suffix in [('counters-profile', 'counters'), ('OSSignpostIntervals', 'signposts')]:
                run(['xcrun', 'xctrace', 'export', '--input', trace,
                     '--xpath', f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
                     '--output', str(prefix) + f'.{suffix}.xml'])
            data = summarize_counters(Path(str(prefix) + '.counters.xml'), Path(str(prefix) + '.signposts.xml'), args.repeats)
            data.update(profile_timing=result, strategy=name, run=index + 1)
            counters[name].append(data)
            Path(str(prefix) + '.summary.json').write_text(json.dumps(data, indent=2))
    # Preserve the local event inventory: L2 TLB events are not L2 cache events.
    inventory = plistlib.loads(Path('/usr/share/kpep/as4.plist').read_bytes())
    (output / 'm4_event_inventory.json').write_text(json.dumps(inventory, indent=2))
    for name in ['tools/profile_schedule_metrics.py', 'tools/counter_options.m', 'tools/analyze_schedules.cpp']:
        (output / Path(name).name).write_bytes((ROOT / name).read_bytes())
    metadata = dict(vars(args), output=str(output), events=EVENTS, warmup=1,
                    environment={k: v for k, v in env.items() if k.startswith(('OMP_', 'KMP_', 'DEVELOPER_DIR'))})
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    write_summary(args, output, diagnostics, counters)


def write_summary(args, output, diagnostics, counters):
    timing_path = output / 'timing_accounted/summary.csv'
    if not timing_path.exists():
        timing_path = output / 'timing/summary.csv'
    timing = {r['strategy']: r for r in csv.DictReader(timing_path.open())}
    accounted = all('accounted_ipc' in r for r in timing.values())
    summary = []
    thread_summary = []
    for name, _ in STRATEGIES:
        t = timing['dynamic,1' if name == 'dynamic' else name]
        row = dict(strategy=name, wall_ms=float(t['wall_ms']), cpu_ms=float(t['cpu_ms']),
                   voluntary_switches_per_call=float(t['voluntary_switches_per_call']),
                   involuntary_switches_per_call=float(t['involuntary_switches_per_call']))
        for key in ['cycles_per_call', 'instructions_per_call', 'ipc', 'l1d_load_misses_per_call', 'l1d_load_miss_fraction']:
            row[key] = statistics.median(r[key] for r in counters[name])
        if accounted:
            row['cycles_per_call'] = float(t['accounted_cycles_per_call'])
            row['instructions_per_call'] = float(t['accounted_instructions_per_call'])
            row['ipc'] = float(t['accounted_ipc'])
        row['work_ms'] = statistics.mean(t['work_ms'] for r in diagnostics[name] for t in r['threads'])
        row['barrier_ms'] = statistics.mean(t['barrier_ms'] for r in diagnostics[name] for t in r['threads'])
        summary.append(row)
        for tid in range(args.threads):
            samples = [r['threads'][tid] for r in diagnostics[name]]
            thread_summary.append(dict(strategy=name, tid=tid,
                nnz_share=statistics.mean(s['nnz'] / args.timing_repeats / 958976 for s in samples),
                work_ms=statistics.mean(s['work_ms'] for s in samples),
                barrier_ms=statistics.mean(s['barrier_ms'] for s in samples)))
    for filename, rows in [('summary.csv', summary), ('threads.csv', thread_summary)]:
        with (output / filename).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    s, g, d = summary
    lines = [f'# 三种调度方式的指标对照（{args.threads} 线程）', '',
             f'默认矩阵、固定种子；普通测量 {args.timing_runs} 轮×{args.timing_repeats} 次，'
             f'硬件计数 {args.runs} 轮×{args.repeats} 次。每次均先预热 1 次并校验，再正式计时。', '',
             '| 指标 | static | guided | dynamic,1 |', '|---|---:|---:|---:|']
    for label, key, factor, digits in [
            ('整体时间/ms', 'wall_ms', 1, 6), ('累计 CPU/ms', 'cpu_ms', 1, 6),
            ('进程 CPU 周期/次' if accounted else '采样 CPU 周期/次', 'cycles_per_call', 1, 0), ('进程 IPC' if accounted else '采样 IPC', 'ipc', 1, 3),
            ('进程指令/次' if accounted else '采样指令/次', 'instructions_per_call', 1, 0),
            ('采样 L1D Load Miss/次（推测执行）', 'l1d_load_misses_per_call', 1, 0),
            ('L1D Load Miss/Load µop（%）', 'l1d_load_miss_fraction', 100, 2),
            ('平均线程工作阶段/ms', 'work_ms', 1, 6), ('平均线程屏障阶段/ms', 'barrier_ms', 1, 6),
            ('自愿上下文切换/次', 'voluntary_switches_per_call', 1, 4),
            ('非自愿上下文切换/次', 'involuntary_switches_per_call', 1, 4)]:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{r[key] * factor:.{digits}f}' for r in [s, g, d]) + ' |')
    lines += ['| L2 Cache Miss | 未采集 | 未采集 | 未采集 |',
              '| 实际 DRAM 带宽 | 未采集 | 未采集 | 未采集 |', '',
              f"本次整体耗时最短的是 {min(summary, key=lambda r: r['wall_ms'])['strategy']}。"
              f"dynamic 相对 guided 的整体耗时增加 {d['wall_ms']/g['wall_ms']-1:.1%}，"
              f"每次调用指令数增加 {d['instructions_per_call']/g['instructions_per_call']-1:.1%}。"
              '因此较高 IPC 不代表完成相同 SpMV 任务更快。',
              'dynamic 的 Load Miss/Load µop 比值较低，也可能受运行库额外访存增加分母影响；'
              '不能直接解释为 CSR 数据访问局部性改善。', '',
              ('周期和 IPC 采用未插桩版本的 proc_pid_rusage(RUSAGE_INFO_V4) 批次差值，累计所有线程的用户态和内核态活动。'
               if accounted else '周期和 IPC 来自 Instruments 采样。'),
              'L1D 硬件计数为 CPU Counters 手动事件模式、EL0，仅筛选 MeasuredSpMV 标记内的目标进程。'
              '计数是 Instruments counters-profile 导出的采样增量之和，并非精确的起止硬件计数；边界舍弃约 1 ms。'
              '每次调用计数除以正式调用次数，各轮取中位数。L1D 比值=推测执行 Load Miss/Load µop，'
              '不是退休指令缓存未命中率。P/E 分项保存在各 trace 的 summary.json 中。',
              '线程工作/屏障时间来自另一次 schedule(runtime) 插桩诊断，常驻线程组，按线程和调用取平均；'
              '工作阶段含计算、调度和抢占，屏障阶段含等待与同步，不能当作纯计算或相加为总时间。',
              '上下文切换由 getrusage 的计时批次差值得到，跨所有线程累计；不是迁核次数。',
              '本机公开的 as4 CPU 事件列表未列出 L2 Cache Miss 或 DRAM 流量事件；L2 TLB Miss 不是 L2 Cache Miss。'
              '本次未获得可归属该进程的 DRAM 读写字节计数，不能把逻辑访存量或 L1 未命中量当作真实内存流量。',
              'IPC 单独不足以判断访存瓶颈；应结合缓存事件、调度指令和等待时间。未绑定 P/E 核，频率和系统负载会影响结果。', '',
              '逐线程明细见 `threads.csv`；各轮原始线程统计见 `threads.raw.jsonl`；汇总见 `summary.csv`。', '',
              '参考：[Apple CPU Counters](https://developer.apple.com/videos/play/wwdc2025/308/)；'
              '[Apple XNU 计数说明](https://github.com/apple-oss-distributions/xnu/blob/main/doc/observability/recount.md)。事件定义见本目录 `m4_event_inventory.json`。原始 trace、XML、源码快照和命令均保留。', '']
    report = '\n'.join(lines)
    (output / 'report.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
