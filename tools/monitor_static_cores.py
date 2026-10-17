#!/usr/bin/env python3
"""Monitor actual CPU placements of OpenMP static workers with Time Profiler."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def positive(value):
    result=int(value)
    if result<1:raise argparse.ArgumentTypeError('must be positive')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=positive,default=4)
    parser.add_argument('--repeats',type=positive,default=20000)
    parser.add_argument('--output',type=Path,default=ROOT/'output/static_core_monitor')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():parser.error('choose a new output directory')
    out.mkdir(parents=True)
    env=dict(os.environ,DEVELOPER_DIR=os.environ.get('DEVELOPER_DIR','/Applications/Xcode.app/Contents/Developer'),OMP_DYNAMIC='FALSE')
    compiler=os.environ.get('CXX','/opt/homebrew/opt/llvm/bin/clang++')
    binary=out/'monitor_static_cores'
    flags=['-O2','-g','-std=c++17','-fopenmp','-pthread']
    subprocess.run([compiler,*flags,'-c',str(ROOT/'tools/monitor_static_cores.cpp'),'-o',str(out/'monitor.o')],check=True)
    subprocess.run([compiler,*flags,str(out/'monitor.o'),'-o',str(binary)],check=True)
    subprocess.run(['xcrun','dsymutil',str(binary)],env=env,check=True)
    (out/'source.cpp').write_bytes((ROOT/'csr_spmv_lab.cpp').read_bytes())
    for name in ['monitor_static_cores.cpp','monitor_static_cores.py']:
        (out/name).write_bytes((ROOT/'tools'/name).read_bytes())
    command=['xcrun','xctrace','record','--template','Time Profiler','--output',str(out/'static.trace'),
             '--time-limit','60s','--target-stdout',str(out/'target.json'),'--env','OMP_DYNAMIC=FALSE',
             '--launch','--',str(binary),'--threads',str(args.threads),'--repeats',str(args.repeats)]
    (out/'metadata.json').write_text(json.dumps(dict(command=command,flags=flags,
         runtime_environment={k:v for k,v in env.items() if k.startswith(('OMP_','KMP_'))}),indent=2))
    print('Recording actual static workers with Time Profiler...',flush=True)
    result=subprocess.run(command,env=env,capture_output=True,text=True)
    (out/'record.log').write_text(result.stdout+result.stderr);result.check_returncode()
    target=json.loads((out/'target.json').read_text())
    assert target['correct'] and target['threads']==args.threads
    export=['xcrun','xctrace','export','--input',str(out/'static.trace'),'--xpath',
            '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]','--output',str(out/'samples.xml')]
    print('Exporting CPU samples...',flush=True)
    subprocess.run(export,env=env,check=True)
    root=ET.parse(out/'samples.xml').getroot()
    ids={e.get('id'):e for e in root.iter() if e.get('id')}
    def resolve(e):
        while e is not None and e.get('ref'):e=ids[e.get('ref')]
        return e
    mapping={int(t['native_tid']):int(t['omp_thread']) for t in target['thread_mapping']}
    observations=defaultdict(list)
    for row in root.iter('row'):
        thread=resolve(row.find('thread'));core=resolve(row.find('core'))
        stack=resolve(row.find('tagged-backtrace'))
        if thread is None or core is None or stack is None:continue
        tid=int(resolve(thread.find('tid')).text)
        if tid not in mapping:continue
        # Count samples whose stacks include the production static function or callback.
        # Running OpenMP runtime frames underneath that call are included.
        frames=[resolve(e) for e in stack.findall('frame')]
        if not any('spmv_omp_student_4' in frame.get('name','') for frame in frames):continue
        stamp=int(resolve(row.find('sample-time')).text)
        weight=int(resolve(row.find('weight')).text)
        label=core.get('fmt','')
        kind='P' if 'P Core' in label else 'E' if 'E Core' in label else '?'
        observations[mapping[tid]].append(dict(time_ns=stamp,omp_thread=mapping[tid],native_tid=tid,
                                             cpu=int(core.text),core_type=kind,weight_ns=weight))
    summary=[]
    with (out/'cpu_samples.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['time_ns','omp_thread','native_tid','cpu','core_type','weight_ns'])
        writer.writeheader()
        for logical in range(args.threads):
            samples=sorted(observations[logical],key=lambda s:s['time_ns'])
            if not samples:raise RuntimeError(f'No static kernel samples for OpenMP thread {logical}')
            writer.writerows(samples)
            count=Counter(s['cpu'] for s in samples)
            weight=Counter()
            for s in samples:weight[s['core_type']]+=s['weight_ns']
            total=sum(weight.values())
            changes=sum(a['cpu']!=b['cpu'] for a,b in zip(samples,samples[1:]))
            summary.append(dict(omp_thread=logical,native_tid=samples[0]['native_tid'],samples=len(samples),
                                cpu_sample_counts=dict(sorted(count.items())),p_weight_fraction=weight['P']/total,
                                e_weight_fraction=weight['E']/total,observed_cpu_changes=changes))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'thread_summary.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['omp_thread','native_tid','samples','cpu_sample_counts','P_fraction','E_fraction','observed_cpu_changes'])
        for r in summary:writer.writerow([r['omp_thread'],r['native_tid'],r['samples'],json.dumps(r['cpu_sample_counts']),r['p_weight_fraction'],r['e_weight_fraction'],r['observed_cpu_changes']])
    for r in summary:
        print(f"thread {r['omp_thread']} (OS tid {r['native_tid']}): CPUs={r['cpu_sample_counts']}; "
              f"P={r['p_weight_fraction']:.2%}, E={r['e_weight_fraction']:.2%}, sampled changes={r['observed_cpu_changes']}",flush=True)
    print('Correctness passed. Samples show observed placement, not fixed affinity or all migrations.')
    print(f'Saved to {out}')


if __name__=='__main__':main()
