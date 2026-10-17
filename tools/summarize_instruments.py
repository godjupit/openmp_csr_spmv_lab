#!/usr/bin/env python3
"""Summarize exported xctrace time-profile tables; weight is sampled CPU time."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import xml.etree.ElementTree as ET


def summarize(path):
    root = ET.parse(path).getroot()
    ids = {e.get('id'): e for e in root.iter() if e.get('id')}

    def resolve(element):
        while element is not None and element.get('ref'):
            element = ids[element.get('ref')]
        return element

    leaf, inclusive, threads = Counter(), Counter(), Counter()
    total = 0
    count = 0
    for row in root.iter('row'):
        stack = resolve(row.find('tagged-backtrace'))
        weight = resolve(row.find('weight'))
        thread = resolve(row.find('thread'))
        if stack is None or weight is None:
            continue
        frames = [resolve(f) for f in stack.findall('frame')]
        names = [f.get('name', '?') for f in frames]
        if not names:
            continue
        value = int(weight.text) / 1e6
        total += value
        count += 1
        leaf[names[0]] += value
        for name in set(names):
            inclusive[name] += value
        if thread is not None:
            threads[thread.get('fmt', '?')] += value
    if not count:
        raise RuntimeError(f'{path}: no CPU samples found')
    return count, total, leaf, inclusive, threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    lines = ['# Time Profiler 采样摘要', '',
             '以下为整个进程的 CPU 采样，包含准备、预热和计算。',
             'Weight 是按样本权重估计的累计 CPU 时间，不是墙钟耗时。',
             '多个线程的 CPU 时间可以叠加；调用栈中各函数的 inclusive 权重不能直接相加。',
             '每种策略只有一次 trace；统计性耗时对比应使用未开采样器的独立重复运行。', '']
    for name in ['serial', 'omp1', 'thread']:
        count, total, leaf, inclusive, threads = summarize(args.directory / f'{name}.samples.xml')
        with (args.directory / f'{name}.hotspots.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['function', 'self_weight_ms', 'self_percent', 'inclusive_weight_ms'])
            for function, value in leaf.most_common():
                writer.writerow([function, value, value / total * 100, inclusive[function]])
        lines += [f'## {name}', '', f'样本数：{count}；累计采样权重：{total:.1f} ms；'
                  f'采样到的线程数：{len(threads)}。', '',
                  '| 热点（栈顶 Self Weight） | 权重/ms | 占比 |', '|---|---:|---:|']
        for function, value in leaf.most_common(8):
            label = function.replace('|', '\\|').replace('\n', ' ')
            if len(label) > 180:
                label = label[:177] + '...'
            lines.append(f'| {label} | {value:.1f} | {value / total:.2%} |')
        lines += ['']
        print(name, 'samples=', count, 'weight_ms=', total, 'threads=', len(threads))
        for function, value in leaf.most_common(5):
            print(f'  {value/total:.2%} {function[:140]}')
    (args.directory / 'hotspots.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
