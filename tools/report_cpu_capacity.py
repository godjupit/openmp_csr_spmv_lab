#!/usr/bin/env python3
"""Summarize the equal-work capacity run, with optional Instruments CPU mapping."""
import csv
import json
from pathlib import Path
import statistics
import sys
from collections import defaultdict


def main():
    out=Path(sys.argv[1])
    rows=list(csv.DictReader((out/'summary.csv').open()))
    raw=[json.loads(line) for line in (out/'raw.jsonl').open()]
    metadata=json.loads((out/'metadata.json').read_text())
    def stat(mode,p):return next(r for r in rows if r['mode']==mode and int(r['threads'])==p)
    coremap=json.loads((out/'core_map.json').read_text()) if (out/'core_map.json').exists() else {}
    observations=defaultdict(list)
    total=0
    for sample in raw:
        if sample['mode']!='compute' or sample['threads']!=10:continue
        for thread in sample['workers']:
            for job in thread['jobs']:
                total+=1
                if job['begin_cpu']==job['end_cpu'] and job['cpu_ms']>0:
                    operations=metadata['iterations_per_job']*metadata['flops_per_iteration']
                    observations[job['begin_cpu']].append(operations/(job['cpu_ms']*1e6))
    core_rows=[]
    for cpu, rates in sorted(observations.items()):
        core_rows.append(dict(cpu=cpu,core=coremap.get(str(cpu),'type unverified'),samples=len(rates),
                              median_gflops=statistics.median(rates),min_gflops=min(rates),max_gflops=max(rates)))
    with (out/'core_estimates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(core_rows[0]));writer.writeheader();writer.writerows(core_rows)
    c10=stat('compute',10)
    stream_rates=[float(r['throughput']) for r in rows if r['mode']=='stream']
    lines=['# 等工作量 CPU 能力测试', '',
           '**配置与方法**', '',
           'Apple M4，4 个性能核 P、6 个能效核 E，共 10 核、10 个逻辑处理器；macOS 26.7.1。',
           'Clang 23.1.2、libomp；`-O2 -std=c++17 -fopenmp -pthread`；不绑定核心、不修改 QoS。',
           '每个配置独立启动 10 次进程，顺序打乱，先预热后计时，全部 120 次运行通过结果和工作量校验。', '',
           '- **计算测试**：固定 120 个等量任务，每个任务执行 100 万次循环；每次循环有 8 条独立的双精度 NEON 向量 FMA，共 32 FLOP。总计 38.4 亿 FLOP，各线程等分任务。汇编已核对每次循环确有 8 条 `fmla.2d`，结果被实际使用并验证。',
           '- **访存测试**：三个 double 数组，共 288 MB；重复 12 次 `z=x+0.5*y`，各线程处理相同元素数。逻辑数据量为 34.56 亿字节，分配、初始化和校验不计时。该指标不是硬件测得的 DRAM 流量。', '',
           '**线程数与吞吐量**', '',
           '| 线程数 | 计算耗时/ms | 计算 GFLOP/s | 加速比 | 并行效率 | 访存逻辑 GB/s |',
           '|---:|---:|---:|---:|---:|---:|']
    for p in [1,2,4,6,8,10]:
        c=stat('compute',p);m=stat('stream',p)
        lines.append(f"| {p} | {float(c['median_ms']):.3f} | {float(c['throughput']):.2f} | {float(c['speedup']):.2f}× | {float(c['efficiency']):.1%} | {float(m['throughput']):.2f} |")
    lines += ['',
              '耗时为 10 次运行的中位数，吞吐量为固定总工作量 / 中位耗时；加速比以同一任务的单线程为基准。',
              '计算 GFLOP/s 是本 FMA 内核的实测值，不是所有指令类型的通用算力或芯片理论峰值。', '',
              '**各 CPU 的采样估计**', '',
              '仅使用 10 线程计算测试中，任务开始与结束查询到同一 CPU 编号的区间。',
              f'共 {total} 个任务区间，保留 {sum(len(v) for v in observations.values())} 个；以任务 FLOP / 线程 CPU 时间估算吞吐。',
              ('CPU 类型由单独的 Instruments Time Profiler 采样确认。' if coremap else '本次未提供 CPU 类型映射，不根据编号推断性能核或能效核。') + '首尾编号相同仍不能排除中途迁移，',
              '这些值是系统调度下的区间估计，不是绑定到每个物理核心后的独占峰值。', '',
              '| CPU 编号 | 类型 | 样本数 | 区间 GFLOP/s 中位数 |',
              '|---:|---|---:|---:|']
    for r in core_rows:
        kind='P 性能核' if 'P Core' in r['core'] else 'E 能效核' if 'E Core' in r['core'] else '未确认'
        lines.append(f"| {r['cpu']} | {kind} | {r['samples']} | {r['median_gflops']:.2f} |")
    lines += ['', '**对并行实验的意义**', '',
              f"1. 10 线程计算吞吐为 {float(c10['throughput']):.2f} GFLOP/s，是单线程的 {float(c10['speedup']):.2f} 倍，而非 10 倍。不同类型核心的速度、调度和同步都影响扩展，不能把 10 核简单理解成 10 个同速核心。",
              f"2. 10 线程计算任务的平均占核数约 {float(c10['average_busy_cores']):.2f}（线程 CPU 时间之和 / 墙钟时间），与 {float(c10['efficiency']):.1%} 的加速效率是不同指标；前者只统计本任务，不是整机 CPU 利用率。",
              f'3. 顺序访存吞吐为 {min(stream_rates):.2f}–{max(stream_rates):.2f} GB/s，当前测量中增加线程收益很小，呈现平台趋势；这不等于直接证明 SpMV 的 DRAM 带宽已经饱和。',
              '4. 等量任务不保证等时完成。评估 static 负载时，除了行数、非零元素数，还应看线程实际耗时；动态调度可让完成较快的线程继续领取任务，其收益仍需在 SpMV 上验证。', '',
              '参考：[Apple：Apple silicon 性能调优与任务调度](https://developer.apple.com/documentation/apple-silicon/tuning-your-code-s-performance-for-apple-silicon)。',
              '数据：[汇总](summary.csv) · [逐线程数据](workers.csv) · [核心区间估计](core_estimates.csv) · [原始数据](raw.jsonl)。',
              '复现：在项目目录运行 `python3 tools/benchmark_cpu_capacity.py --output output/cpu_capacity_run2`。',
              '代码与分析使用 AI 辅助。', '']
    (out/'report.md').write_text('\n'.join(lines))
    print('Wrote',out/'report.md')


if __name__=='__main__':main()
