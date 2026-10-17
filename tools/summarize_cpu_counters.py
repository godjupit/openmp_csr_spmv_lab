#!/usr/bin/env python3
"""Aggregate xctrace MetricTable ratios by core type, weighted by CPU cycles."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xml', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Keep compact reference values instead of retaining the full XML tree.
    ids = {}

    def decode(e):
        if e.get('ref'):
            return ids[e.get('ref')]
        value = (e.tag, e.text or '', e.get('fmt', ''), tuple(decode(c) for c in e))
        if e.get('id'):
            ids[e.get('id')] = value
        return value

    cycles = defaultdict(float)
    weighted = defaultdict(float)
    docs = {}
    current_key = None
    current_cycle = None
    interval_count = 0
    table_node = None
    for event, element in ET.iterparse(args.xml, events=('start', 'end')):
        if event == 'start' and element.tag == 'node':
            table_node = element
        if event != 'end' or element.tag != 'row':
            continue
        cells = [decode(e) for e in element]
        if len(cells) == 10:
            time, duration, mnemonic, label, thread, process, value, core, ratio, doc = cells
            if 'csr_spmv_lab' in process[2]:
                key = (time[1], duration[1], thread[2], core[2])
                core_type = 'P Core' if 'P Core' in core[2] else 'E Core'
                if mnemonic[1] == 'cycle':
                    current_key = key
                    current_cycle = float(value[1])
                    cycles[core_type] += current_cycle
                    interval_count += 1
                elif ratio[1] == '1':
                    if key != current_key:
                        raise RuntimeError('Metric rows do not follow their cycle denominator')
                    weighted[core_type, label[1]] += current_cycle * float(value[1])
                if doc[0] != 'sentinel':
                    docs[label[1]] = '\n'.join(c[1] for c in doc[3])
        element.clear()
        if table_node is not None:
            table_node.remove(element)
    if not cycles:
        raise RuntimeError('No target-process counters found')
    with args.output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['core_type', 'metric', 'cycle_weighted_fraction', 'cpu_cycles'])
        for (core_type, metric), value in sorted(weighted.items()):
            ratio = value / cycles[core_type]
            writer.writerow([core_type, metric, ratio, cycles[core_type]])
            print(f'{core_type}: {metric}: {ratio:.4%}')
    args.output.with_suffix('.definitions.txt').write_text(
        '\n\n'.join(f'{name}\n{text}' for name, text in docs.items()))
    print('Target-process measurement intervals:', interval_count)


if __name__ == '__main__':
    main()
