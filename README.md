# OpenMP CSR SpMV 实验

比较四种 OpenMP 策略：guided 按行调度、guided＋SIMD、64 行分块动态调度、static 按行调度。
串行和 `std::thread` 保留为对照，各策略均与串行结果校验。

## 文件位置

- [csr_spmv_lab.cpp](csr_spmv_lab.cpp)：主源码；[csr.pdf](csr.pdf)：作业要求。
- [report.md](report.md)：主实验报告；[优化过程.md](优化过程.md)：个人过程记录，保留在根目录。
- `tools/`：编译、基准测试与分析工具。
- `output/`：数据、日志、图表、编译产物、各次测试的自动摘要与源码快照。

## 编译运行（本机 macOS）

```bash
./tools/run.sh --threads 8 --repeats 100
./tools/run.sh --only omp4 --threads 4 --repeats 100
```

`tools/run.sh` 每次先编译最新源码，编译失败则停止，编译产物放入 `output/`。编译不计入运行时间。

`--only` 支持 `all`、`serial`、`thread`、`omp1` 至 `omp4`。
默认只输出策略、平均耗时（毫秒）和相对串行加速比的表格。单独运行时也包含串行对照。两者各预热 1 次，再各计时 `--repeats` 次。`--verbose` 显示详细参数和供脚本解析的结果；性能采集可用 `--no-baseline` 跳过额外串行计时。
默认矩阵为 10000×100000，普通行各 32 个存储条目，32 个长行各 20000 个，种子 20260926。

| 策略 | 实现 |
|---|---|
| `omp1` | `schedule(guided)` 按行分配 |
| `omp2` | guided＋行内 `omp simd reduction(+:sum)` |
| `omp3` | 每 64 行为一个任务，`schedule(dynamic)` |
| `omp4` | 最基础的 `schedule(static)` 按行分配 |

## 测试命令

以下命令从项目根目录执行。重测使用新输出目录，以保留已有数据。

单配置重复计时：

```bash
CXX=/opt/homebrew/opt/llvm/bin/clang++ python3 tools/benchmark.py --runs 10 --repeats 500 --threads 8 --output output/benchmark_run2
```

矩阵规模和线程数：

```bash
python3 tools/sweep_parameters.py --output output/parameter_run2
```

guided 逐线程负载（自动编译，统计实际动态分配和线程耗时）：

```bash
python3 tools/analyze_guided.py --threads 4 --runs 10 --repeats 500 --output output/guided_run2
```

static 逐线程负载：

```bash
python3 tools/analyze_static.py --threads 4 --output output/static_4t_run2
python3 tools/analyze_static.py --threads 8 --output output/static_8t_run2
```

P/E 核差异与等工作量 CPU 测试：

```bash
python3 tools/compare_pe.py --output output/pe_run2
python3 tools/benchmark_cpu_capacity.py --output output/cpu_capacity_run2
```

Instruments 采样及查看：

```bash
bash tools/profile_instruments.sh output/instruments_run2
open -a Instruments output/instruments_results/omp1.trace
```

Linux / WSL 的 perf 对比：

```bash
python3 tools/profile_perf.py --threads 8 --runs 10 --repeats 1000 --output output/perf_run2
```

## 已有结果

- [基础计时](output/report_results/summary.csv)、[参数实验](output/parameter_results/summary.csv)。
- [4 线程 static 负载](output/static_analysis_4t/report.md)、[8 线程 static 负载](output/static_analysis/report.md)。
- [P/E 核比较](output/pe_comparison/report.md)、[CPU 吞吐测试](output/cpu_capacity_results/report.md)。
- [性能指标分析](output/csr_analysis.md)、[长行分布](output/long_row_distribution/report.md)。

逐线程计时来自插桩诊断循环，与原始实现耗时分开统计。CPU 核型结果属于调度条件下的采样估计。
Instruments 是本机替代分析工具；作业 PDF 第 3.5 节原文要求 Linux perf。

## 查看 static 工作线程实际运行的核心

```bash
python3 tools/monitor_static_cores.py --threads 4 --output output/static_core_monitor_run2
open -a Instruments output/static_core_monitor/static.trace
```

使用 Time Profiler 采样原始 static 调用栈，将 OpenMP 线程编号对应到系统线程 ID，
记录实际 CPU 编号、P/E 核型及采样间换核情况。这是运行位置观测，不是强制绑核。
结果见 `output/static_core_monitor/thread_summary.csv`，逐样本记录见 `cpu_samples.csv`。
