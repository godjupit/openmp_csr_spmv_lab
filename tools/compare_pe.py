#!/usr/bin/env python3
"""Compare P/E core FP64 throughput with identical QoS and task sizes."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT / 'output/pe_comparison')
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():parser.error('choose a new output directory')
    root=Path(__file__).resolve().parents[1]
    mapping=json.loads((root/'output/cpu_capacity_results/core_map.json').read_text())
    out.mkdir(parents=True)
    compiler=os.environ.get('CXX','/opt/homebrew/opt/llvm/bin/clang++')
    build=[compiler,'-O2','-std=c++17','-fopenmp','-pthread',str(root/'tools/compare_pe.cpp'),'-o',str(out/'compare_pe')]
    subprocess.run(build,check=True)
    for name in ['compare_pe.cpp','compare_pe.py','cpu_capacity.cpp']:
        (out/name).write_bytes((root/'tools'/name).read_bytes())
    (out/'core_map.json').write_text(json.dumps(mapping,indent=2))
    (out/'metadata.json').write_text(json.dumps(dict(build=build,runs=10,jobs=12000,iterations_per_job=100000,
                                                flops_per_iteration=32,worker_threads=10,qos='USER_INITIATED'),indent=2))
    results=[]
    with (out/'raw.jsonl').open('w') as file:
        for run in range(10):
            proc=subprocess.run([str(out/'compare_pe')],text=True,capture_output=True,check=True)
            data=json.loads(proc.stdout)
            assert data['correct'] and len(data['samples'])==12000 and data['threads']==10
            data['run']=run+1;file.write(json.dumps(data)+'\n');file.flush()
            for kind in ['P Core','E Core']:
                kept=[s for s in data['samples'] if s['begin_cpu']==s['end_cpu'] and
                      kind in mapping.get(str(s['begin_cpu']),'') and s['cpu_ms']>0]
                if len(kept)<100:raise RuntimeError(f'too few actual {kind} intervals ({len(kept)})')
                row=dict(run=run+1,core_type=kind,kept=len(kept),
                         mean_cpu_ms=statistics.mean(s['cpu_ms'] for s in kept),
                         mean_wall_ms=statistics.mean(s['wall_ms'] for s in kept),
                         observed_cpus=','.join(str(i) for i in sorted(set(s['begin_cpu'] for s in kept))))
                row['gflops']=3.2/row['mean_cpu_ms'];results.append(row)
                print(f"Round {run+1}: {kind}, accepted={len(kept)}, {row['gflops']:.2f} GFLOP/s",flush=True)
    with (out/'runs.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary=[]
    for kind in ['P Core','E Core']:
        selected=[r for r in results if r['core_type']==kind]
        cpu=statistics.median(r['mean_cpu_ms'] for r in selected)
        summary.append(dict(core_type=kind,runs=len(selected),samples=sum(r['kept'] for r in selected),
                            cpu_ms_per_job=cpu,wall_ms_per_job=statistics.median(r['mean_wall_ms'] for r in selected),
                            gflops=3.2/cpu,min_gflops=min(r['gflops'] for r in selected),max_gflops=max(r['gflops'] for r in selected)))
    with (out/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    p,e=summary;ratio=p['gflops']/e['gflops']
    lines=['# P 核与 E 核性能对比','',
           '配置：Apple M4，同一双精度 NEON FMA 内核，10 个工作线程均使用 USER_INITIATED QoS，编译选项 `-O2 -std=c++17 -fopenmp -pthread`。',
           '独立运行 10 轮；每轮预热后动态分发 12000 个等量任务，每个任务执行 10 万次循环、320 万 FLOP，让两类核心在相同负载条件下持续工作。',
           '按实际 CPU 编号分类，仅保留首尾同一 CPU 的区间；不将不同 QoS 的结果混作 P/E 硬件差距。',
           'CPU 类型映射来自此前单独的 Instruments 采样。全部 120000 个任务通过计算结果校验。','',
           '| 类型 | 有效样本 | 每任务 CPU 耗时/ms | 每核区间吞吐 GFLOP/s |',
           '|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['core_type']} | {r['samples']} | {r['cpu_ms_per_job']:.6f} | {r['gflops']:.2f} |")
    lines+=['',f'在该计算内核和当前共同负载下，P 核每核吞吐约为 E 核的 **{ratio:.2f} 倍**，完成同量任务的 CPU 耗时低约 **{(1-1/ratio)*100:.1f}%**。',
            '先求每轮每类核心有效任务的平均 CPU 耗时，再取 10 轮中位数；吞吐按 320 万 FLOP / CPU 耗时计算，不是 P/E 核群总吞吐。',
            '这是当前频率与调度条件下的实测差距，不是同频 IPC、独占单核峰值或所有程序的通用性能比。',
            '未强制绑核；首尾 CPU 相同仍不能完全排除中途迁移，结果属于区间采样估计。','',
            '数据：[各轮统计](runs.csv) · [汇总](summary.csv) · [原始任务区间](raw.jsonl)。',
            '复现：`python3 tools/compare_pe.py --output output/pe_comparison_run2`。代码与分析使用 AI 辅助。','']
    (out/'report.md').write_text('\n'.join(lines));print('\n'.join(lines))


if __name__=='__main__':main()
