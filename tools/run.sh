#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
compiler="${CXX:-/opt/homebrew/opt/llvm/bin/clang++}"
mkdir -p "$project_dir/output"

"$compiler" -O2 -std=c++17 -fopenmp -pthread \
    "$project_dir/csr_spmv_lab.cpp" -o "$project_dir/output/csr_spmv_lab_omp"

exec "$project_dir/output/csr_spmv_lab_omp" "$@"
