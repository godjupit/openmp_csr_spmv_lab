#!/usr/bin/env python3
"""Inspect long-row positions using the project's actual C++ matrix generator."""
import argparse
from bisect import bisect_left
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/long_row_distribution')
    args=parser.parse_args();out=args.output.resolve()
    if (out/'report.md').exists():parser.error('choose a new directory to preserve the previous report')
    out.mkdir(parents=True,exist_ok=True)
    compiler=os.environ.get('CXX','/opt/homebrew/opt/llvm/bin/clang++')
    build=[compiler,'-O2','-std=c++17','-fopenmp','-pthread',str(ROOT/'tools/long_row_positions.cpp'),'-o',str(out/'long_row_positions')]
    subprocess.run(build,check=True)
    data=json.loads(subprocess.check_output([str(out/'long_row_positions')],text=True))
    positions=data['positions'];assert positions==sorted(set(positions)) and len(positions)==32
    assert data['rows']==10000 and data['nnz']==958976
    (out/'matrix.json').write_text(json.dumps(data,indent=2))
    (out/'metadata.json').write_text(json.dumps(dict(build=build,seed=data['seed']),indent=2))
    with (out/'positions.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['row_index_0_based','stored_entries'])
        writer.writerows((p,data['long_nnz']) for p in positions)
    bins=[]
    for width in [1000,1250,2500]:
        for start in range(0,data['rows'],width):
            end=min(start+width,data['rows'])
            count=bisect_left(positions,end)-bisect_left(positions,start)
            bins.append(dict(bin_width=width,begin_row=start,end_row_inclusive=end-1,long_rows=count,
                             stored_entries=(end-start)*data['normal_nnz']+count*(data['long_nnz']-data['normal_nnz'])))
    with (out/'bins.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(bins[0]));writer.writeheader();writer.writerows(bins)
    gaps=[b-a for a,b in zip(positions,positions[1:])]
    with (out/'gaps.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['left_long_row','right_long_row','row_index_distance'])
        writer.writerows((a,b,b-a) for a,b in zip(positions,positions[1:]))
    windows=[]
    for width in [1000,1250,2500]:
        counts=[bisect_left(positions,start+width)-bisect_left(positions,start) for start in range(data['rows']-width+1)]
        windows.append(dict(width=width,min_long_rows=min(counts),max_long_rows=max(counts),
                            first_max_begin=counts.index(max(counts))))
    (out/'sliding_windows.json').write_text(json.dumps(windows,indent=2))
    lines=['# 默认矩阵的长行分布','',
           '配置：10000 行、100000 列，种子 20260926；32 个长行各含 20000 个条目，其余行各 32 个。',
           '直接复用 C++ 矩阵生成函数，以下行号均从 0 开始。','', '**32 个长行的位置**','','```text']
    for i in range(0,len(positions),8):lines.append(', '.join(str(x) for x in positions[i:i+8]))
    lines+=['```','','**每 1000 行中的长行数量**','','| 行区间 | 长行数 |','|---|---:|']
    for b in bins:
        if b['bin_width']==1000:lines.append(f"| {b['begin_row']}–{b['end_row_inclusive']} | {b['long_rows']} |")
    lines+=['', '**与 static 分配的对应关系**','',
            '4 线程按每 2500 行划分，长行数为 **9、9、7、7**；',
            '8 线程按每 1250 行划分，长行数为 **4、5、6、3、5、2、3、4**。',
            '相同矩阵下，区间划分不同，就会呈现不同程度的线程负载差异。','',
            f'相邻长行的行号距离：最小 **{min(gaps)}**，最大 **{max(gaps)}**，平均 **{statistics.mean(gaps):.2f}**，中位数 **{statistics.median(gaps):.0f}**。',
            '这些距离不是短行数量；两长行之间的短行数还需减 1。']
    for window in windows:
        lines.append(f"任意连续 {window['width']} 行中，长行数最少 {window['min_long_rows']} 个、最多 {window['max_long_rows']} 个；第一个最密窗口从第 {window['first_max_begin']} 行开始。")
    lines+=['','生成方式是打乱全部行号后选出 32 行，没有人为设置聚集或等间距。',
            '本样本确有局部疏密差异：3000–3999 行有 6 个长行，4000–4999 和 7000–7999 各只有 1 个。',
            '4 个大区间中数量接近，是这次固定随机样本的结果，不能推断所有种子都同样均衡。',
            '当前统计描述单个矩阵，不构成对随机数发生器的统计检验。','',
            '数据：[长行位置](positions.csv) · [分区计数](bins.csv) · [相邻距离](gaps.csv)。',
            '复现：`python3 tools/analyze_long_rows.py --output output/long_rows_run2`。','']
    (out/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':main()
