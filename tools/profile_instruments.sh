#!/bin/bash
# Capture one strategy per trace, using the current project source.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
output_dir="${1:-$project_dir/output/instruments_results_new}"
compiler_path="${CXX:-/opt/homebrew/opt/llvm/bin/clang++}"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
if [[ "$(uname -s)" != Darwin ]]; then
    echo 'This script requires macOS and full Xcode.' >&2
    exit 1
fi
if [[ -e "$output_dir" ]]; then
    echo "Output already exists; choose a new directory: $output_dir" >&2
    exit 1
fi
xcrun --find xctrace >/dev/null
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
cp "$project_dir/csr_spmv_lab.cpp" "$output_dir/source.cpp"
"$compiler_path" -O2 -g -std=c++17 -fopenmp -pthread \
     -c "$project_dir/csr_spmv_lab.cpp" -o "$output_dir/csr_spmv_lab.o"
"$compiler_path" -O2 -g -std=c++17 -fopenmp -pthread \
    "$output_dir/csr_spmv_lab.o" -o "$output_dir/csr_spmv_lab"
xcrun dsymutil "$output_dir/csr_spmv_lab"
{
    date -u
    sw_vers
    sysctl machdep.cpu.brand_string hw.physicalcpu hw.logicalcpu
    "$compiler_path" --version
    xcrun xctrace version
    otool -L "$output_dir/csr_spmv_lab"
    echo 'flags=-O2 -g -std=c++17 -fopenmp -pthread'
    echo 'OMP_DYNAMIC=FALSE; no explicit affinity; threads=8; repeats=10000'
} > "$output_dir/environment.txt"
for strategy in serial omp1 thread; do
    xcrun xctrace record --template 'Time Profiler' \
        --output "$output_dir/$strategy.trace" --time-limit 60s \
        --target-stdout "$output_dir/$strategy.profile.stdout.txt" \
        --env OMP_DYNAMIC=FALSE --launch -- "$output_dir/csr_spmv_lab" \
        --verbose --no-baseline --only "$strategy" --threads 8 --repeats 10000
    grep -q 'All correctness checks passed.' "$output_dir/$strategy.profile.stdout.txt"
    xcrun xctrace export --input "$output_dir/$strategy.trace" \
        --toc --output "$output_dir/$strategy.toc.xml"
    xcrun xctrace export --input "$output_dir/$strategy.trace" \
        --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
        --output "$output_dir/$strategy.samples.xml"
done
python3 "$project_dir/tools/summarize_instruments.py" "$output_dir"
echo "Saved Time Profiler traces to $output_dir"
