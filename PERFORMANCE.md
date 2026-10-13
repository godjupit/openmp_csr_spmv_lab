# 3.5 Linux 性能工具：单策略比较

## 环境与方法

测量日期：2026-10-13。Ubuntu 22.04.5 LTS，WSL2 内核
`6.18.33.2-microsoft-standard-WSL2`，perf 版本 `5.15.216`。
WSL 中 lscpu 报告 CPU 型号为 Intel Core Ultra 5 125H，18 个逻辑 CPU、
9 个核心；这是虚拟机看到的拓扑，不代表宿主机完整物理拓扑。
环境原始记录见 `perf_results/lscpu.txt`、`uname.txt`、`perf-list.txt` 和
`perf-version.txt`。

编译命令：

```bash
g++ -O2 -g -std=c++17 -fopenmp -pthread csr_spmv_lab.cpp -o csr_spmv_lab
```

输入为默认固定种子矩阵：10000 行、100000 列、958976 个非零元素，
普通行 32 个非零元素，32 个长行各有 20000 个非零元素。
比较串行、OpenMP 2（`schedule(guided, 16)`）、std::thread（连续行分块）。
并行实现请求 8 个线程；串行仍只使用一个线程。

实际执行的命令：

```bash
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
for strategy in serial omp2 thread; do
    perf stat -r 7 \
        -e task-clock,context-switches,cpu-migrations,page-faults \
        ./csr_spmv_lab --only "$strategy" --threads 8 --repeats 1000
done
```

`--only` 只对指定实现执行预热和重复计时，保留一次串行参考计算用于
正确性校验，不运行其他策略的 benchmark。全部运行均通过正确性校验。
chrono 只测量 1000 次目标调用；perf 测量整个进程，还包含矩阵和向量生成、
一次串行参考计算、预热、校验及输出。增大重复次数降低准备工作的相对
影响，但没有完全消除它，不能把进程计数严格视作计算循环的计数。
OpenMP 亲和性设置只作用于 OpenMP，std::thread 没有显式绑定 CPU。

## 事件可用性

对 `cycles,instructions,cache-misses,branch-misses` 的实际探测全部返回
`<not supported>`，见 `perf_results/hardware-probe.txt`。因此使用软件
事件替代，无法报告 IPC、缓存未命中率或分支预测失误率。

本环境 perf 输出事件后缀为 `:u`，即限制在用户态。
`context-switches:u` 和 `cpu-migrations:u` 的结果均为零，这不能说明
程序完全没有上下文切换或 CPU 迁移；这些事件与内核调度有关，当前统计
范围不足以据此判断调度行为。没有修改系统权限或 perf 安全设置。

## 测量结果

chrono 时间为七次独立进程输出的单次平均耗时的算术平均。
加速比使用 chrono 串行均值除以对应并行均值；效率为加速比除以 8。
perf 列为 `perf stat -r 7` 报告的进程级统计。

| 实现 | chrono 单次时间/ms | 加速比 | 并行效率 | perf 进程 elapsed/s | perf task-clock/ms | perf CPUs utilized | page-faults |
|---|---:|---:|---:|---:|---:|---:|---:|
| serial | 0.814339 | 1.000 | — | 0.84470 ± 0.00829 | 847.38 | 1.003 | 6210 |
| OpenMP 2 | 0.248013 | 3.283 | 41.04% | 0.28044 ± 0.00667 | 2158.30 | 7.696 | 6226 |
| std::thread | 0.770242 | 1.057 | 13.22% | 0.8013 ± 0.0124 | 1142.75 | 1.426 | 14233 |

原始程序输出为 `perf_results/{serial,omp2,thread}.stdout.txt`；
perf 原始输出为同目录下相应的 `.stat.txt` 文件。
表中的误差项保留 perf 原始格式，不解释为置信区间。

## 分析

OpenMP 2 的计算区间时间约为串行的 30.5%，加速比为 3.28。
perf 的整个进程耗时也明显下降，但 task-clock 增加到约 2158 ms，
说明减少墙钟时间的同时使用了更多累计 CPU 时间。
CPUs utilized 是累计 CPU 时间与墙钟时间的比值，不能直接当作并行效率；
即使平均占用接近 8 个 CPU，也只有约 41% 的加速效率。

从实现看，guided 调度能动态分配剩余行，有助于应对长短行混合的输入；
OpenMP 运行时通常复用工作线程。调度和同步开销、长行处理的不均衡以及
数据访问开销都可能限制扩展性。当前没有硬件计数，不能确认缓存或内存
带宽是否为主要瓶颈，也没有对线程池复用开销进行单独隔离测量。

std::thread 仅获得约 1.06 倍加速。当前实现每次 SpMV 都创建并 join
8 个线程，1000 次重复共创建 8000 个工作线程；连续行分块也可能因为
长行分布而产生负载不均衡。这些是代码中可见的潜在成本，尚不能仅凭
perf stat 将二者贡献精确分开。
其 page-faults 比其他版本多，可能与反复创建线程时的栈分配和首次触页
有关；该指标不等于磁盘 I/O 次数。平均 CPU 占用仅约 1.43，结合接近串行
的墙钟时间，说明本次配置下增加线程没有带来明显收益。

结论适用于本次输入、绑定方式和 WSL2 环境。三种实现按顺序测试，宿主机
负载、频率变化和虚拟化调度可能影响结果；这些数字不代表所有矩阵或机器。
