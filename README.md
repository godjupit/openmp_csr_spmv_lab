# OpenMP CSR SpMV 实验

最终保留三种 OpenMP 策略：逐行 guided、guided,16＋SIMD、64 行块 guided。
串行和 `std::thread` 保留为对照，各策略均与串行结果校验。

## 文件位置

- [csr_spmv_lab.cpp](csr_spmv_lab.cpp)：主源码；[csr.pdf](csr.pdf)：作业要求。
- [report.md](report.md)：主实验报告；[优化过程.md](优化过程.md)：个人过程记录，保留在根目录。
- `tools/`：编译、基准测试与分析工具。
- `output/`：数据、日志、图表、编译产物、各次测试的自动摘要与源码快照。

## 当前报告材料

- [报告 PDF：共 4 页（正文 3 页＋截图附录 1 页）](report.pdf)
- [真实终端正确性截图](correctness.png)
- [材料压缩包](submission_materials.zip)：个人过程记录原样附入，仍需本人核对并补相关 AI 对话截图。
- [最终基准原始数据](output/final_report/benchmark_three/raw.csv)、[SIMD 消融](output/final_report/simd_ablation/summary.csv)、[Instruments 热点](output/final_report/instruments/hotspots.md)
- 性能分析按本机环境使用 Instruments；与作业指定 Linux perf 的差异已在报告说明。

## 编译运行（本机 macOS）

```bash
./tools/run.sh --threads 8 --repeats 5000
./tools/run.sh --only omp2 --threads 4 --repeats 5000
```

`tools/run.sh` 每次先编译最新源码，编译失败则停止，编译产物放入 `output/`。编译不计入运行时间。

`--only` 支持 `all`、`serial`、`thread`、`omp1` 至 `omp3`。
默认只输出策略、平均耗时（毫秒）和相对串行加速比的表格。单独运行时也包含串行对照。两者各预热 1 次，再各计时 `--repeats` 次。`--verbose` 显示详细参数和供脚本解析的结果；性能采集可用 `--no-baseline` 跳过额外串行计时。
默认矩阵为 10000×100000，普通行各 32 个存储条目，32 个长行各 20000 个，种子 20260926。

| 策略 | 实现 |
|---|---|
| `omp1` | `schedule(guided)` 按行分配 |
| `omp2` | `schedule(guided, 16)`＋行内 `omp simd reduction(+:sum)` |
| `omp3` | 每 64 行为一个迭代块，`schedule(guided)` 分配这些块 |

## 测试命令

以下命令从项目根目录执行。重测使用新输出目录，以保留已有数据。

三种调度的 CPU、缓存与线程指标（macOS + 完整 Xcode，自动编译；硬件计数 3 轮、普通计时 10 轮）：

```bash
python3 tools/profile_schedule_metrics.py --threads 4 --runs 3 --repeats 10000 --output output/schedule_metrics_run2
```

输出 `report.md`、`summary.csv`、`threads.csv` 和原始 Instruments trace。使用相同计算体的 static、guided、dynamic,1 对照内核，不计入三个提交策略。L2 缓存与实际 DRAM 带宽未采集时明确留空；L1D 事件比值不等同于退休指令的缓存未命中率。

static 与逐行 dynamic 的累计 CPU 时间对比（自动编译，计算体相同，仅调度不同）：

```bash
python3 tools/compare_schedule_cpu.py --threads 4 --runs 10 --repeats 500 --output output/schedule_cpu_run2
```

累计 CPU 时间包含所有线程的计算、调度及忙等；平均占用核数为 CPU 时间除以墙钟时间，不能直接视作有效计算效率。

增加 `--breakdown` 可逐行读取线程 CPU 时钟，校准后估计计算与剩余开销，输出 `breakdown.csv`。此模式有明显插桩扰动，不能代表原程序的精确拆分；若出现负开销估计，说明校准误差过大。

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

独立逐行 dynamic 对照内核的诊断工具：

```bash
python3 tools/analyze_dynamic.py --threads 4 --runs 10 --repeats 500 --output output/dynamic_run2
```

static 与 dynamic 诊断工具使用独立对照内核，见 `tools/exploration_kernels.hpp`；不依赖已移除的策略 4。

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
