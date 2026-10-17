#!/usr/bin/env python3
"""Package the measured three-strategy version, report and unchanged personal notes."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/submission_materials.zip')
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        parser.error('choose a new archive path')
    target.parent.mkdir(parents=True, exist_ok=True)
    final = ROOT / 'output/final_report'
    source = (final / 'csr_spmv_lab.cpp').read_bytes()
    if source != (ROOT / 'csr_spmv_lab.cpp').read_bytes():
        parser.error('project source differs from measured source; freeze and validate the intended version first')
    files = {}

    def add(path, name=None):
        path = Path(path)
        files[name or str(path.relative_to(ROOT))] = path.read_bytes()

    for name in ['csr_spmv_lab.cpp', 'report.md', '优化过程.md']:
        add(ROOT / name)
    add(final / 'pdf/report.pdf', 'report.pdf')
    for path in (ROOT / 'tools').iterdir():
        if path.is_file() and path.suffix in {'.py', '.cpp', '.hpp', '.m', '.sh'}:
            add(path)
    simple_suffixes = {'.csv', '.json', '.jsonl', '.txt', '.md', '.cpp', '.hpp', '.sh'}
    for dirname in ['benchmark_three', 'simd_ablation', 'correctness_capture']:
        for path in (final / dirname).rglob('*'):
            if path.is_file() and path.suffix in simple_suffixes | {'.png'}:
                add(path)
    add(final / 'csr_spmv_lab.cpp')
    add(final / 'report_before_completion.md')
    # Preserve actual Time Profiler traces; leave large CPU-counter raw exports local.
    for path in (final / 'instruments').rglob('*'):
        if path.is_file() and (any(p.suffix == '.trace' for p in path.parents)
                               or path.suffix in {'.csv', '.txt', '.md', '.cpp'}):
            add(path)
    for dirname in ['report_results', 'parameter_results', 'guided_chunks_4t_5000_20261017',
                    'guided_chunks_reverse_4t_5000_20261017', 'guided_chunks_random5_4t_5000_20261017',
                    'guided_chunks_random5_4t_5000_20261017_repeat2']:
        for path in (ROOT / 'output' / dirname).iterdir():
            if path.is_file() and path.suffix in simple_suffixes:
                add(path)
    counters = ROOT / 'output/schedules_hardware_4t_20261017_run2'
    for path in counters.iterdir():
        if path.is_file() and (path.name.endswith('.summary.json') or path.name in
                              {'counter-options.json', 'metadata.json', 'm4_event_inventory.json', 'commands.json', 'xctrace-version.txt'}):
            add(path)
    readme = '''# OpenMP CSR SpMV 材料

最终代码只注册 omp1、omp2、omp3：逐行 guided；guided,16 + SIMD；64 行块 guided。
report.pdf 为正文 4 页、正确性截图附录 1 页。report.md 是对应可编辑源文件。

运行（Mac，需 Homebrew LLVM/libomp）：
  bash tools/run.sh --threads 8 --repeats 5000

复测基准：
  CXX=/opt/homebrew/opt/llvm/bin/clang++ python3 tools/benchmark.py --runs 10 --repeats 5000 --threads 8 --output output/benchmark_reproduce

固定 guided,16 的 SIMD 消融：
  python3 tools/simd_ablation.py --output output/simd_ablation_reproduce

生成真实终端截图（macOS Terminal/屏幕录制权限）：
  python3 tools/capture_correctness.py --output output/correctness_reproduce

报告重新导出（XeLaTeX、fontspec、xeCJK、booktabs、tabularx，使用 macOS 宋体）：
  python3 tools/build_report.py

性能分析使用 Instruments。最终源码的 Time Profiler 原始 trace 在 output/final_report/instruments/。
CPU Counters 对照实验的事件配置与各轮统计保留在 output/schedules_hardware_4t_20261017_run2/；其大体积原始 trace/XML 保留于本机同名目录，可按记录命令重采。
Linux perf 与 Instruments 的工具差异已在报告中说明。

个人材料待本人核对：优化过程.md 按原样复制，未由 AI 改写或润色。请本人确认 300–1500 字、过程与来源披露，并按实际情况附相关 AI 对话截图。该项未确认前，这份压缩包是材料草稿。
历史探索记录不代表最终策略 2、3 的参数；当前结论以 report.md 和对应源码哈希为准。
'''
    files['提交说明.md'] = readme.encode()
    manifest = '\n'.join(f'{hashlib.sha256(data).hexdigest()}  {name}' for name, data in sorted(files.items())) + '\n'
    files['SHA256SUMS.txt'] = manifest.encode()
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            executable = name == 'tools/run.sh' or name == 'tools/profile_instruments.sh'
            info.external_attr = (0o100755 if executable else 0o100644) << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert archive.read('csr_spmv_lab.cpp') == source
        assert archive.read('优化过程.md') == (ROOT / '优化过程.md').read_bytes()
        assert 'report.pdf' in archive.namelist()
    print(f'{target}: {len(files)} files; {target.stat().st_size / 1024 / 1024:.1f} MiB; integrity checks passed')


if __name__ == '__main__':
    main()
