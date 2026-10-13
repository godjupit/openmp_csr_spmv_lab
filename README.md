# OpenMP CSR SpMV 并行策略实验

这是一个自包含的 C++ 参考代码，实验对象是“少数行特别长”的 skewed CSR 矩阵。
矩阵和输入向量都由固定随机种子生成，因此不同并行策略可以在同一输入上比较。

代码包含：

- naive 串行 CSR SpMV；
- `std::thread` 的连续行分块并行实现；
- 五个 OpenMP 实验函数入口：`spmv_omp_student_1` 至
  `spmv_omp_student_5`；
- 结果正确性校验器；
- 简单的重复计时和命令行参数。

## 编译运行

```bash
g++ -O2 -std=c++17 -fopenmp -pthread csr_spmv_lab.cpp -o csr_spmv_lab
./csr_spmv_lab --threads 8 --repeats 10
```

如果只想先验证串行和 `std::thread`，也可以省略 `-fopenmp`，但 OpenMP
函数仍然只是占位回退实现。

## 实验入口
请实现
`spmv_omp_student_1`, `spmv_omp_student_2`两种不同的自定义openmp并行策略，并分析两种策略性能差别原因。

如果有其他你觉得有趣有价值的并行策略，也可以继续注册`spmv_omp_student_3`等函数，但`spmv_omp_student`的函数数量不得超过5个（上限为`spmv_omp_student_5`），超过5个的部分将不会被检查。**作业成绩与额外注册函数的数量无关**。

每次实现都会经过同一个 `check_result`，浮点误差允许绝对误差和相对误差各
`1e-10`。可以修改命令行参数来改变 skew 程度，例如：

```bash
./csr_spmv_lab --rows 20000 --cols 200000 \
  --normal-nnz 16 --long-rows 16 --long-nnz 80000 \
  --threads 16 --repeats 20
```

## 使用 perf 比较单个实现

### 不依赖 perf 的重复统计

在 WSL/Linux 中运行（需要 Python 3 和支持 OpenMP 的 g++）：

```bash
python3 benchmark.py --runs 1000 --threads 8
```

脚本自动编译当前源码，默认对串行、std::thread 和五个 OpenMP 实现各启动
1000 次独立进程。统计程序输出的 SpMV 计算耗时，不包含进程启动、矩阵生成
和正确性验证；每个进程先预热一次，再计时。轮换执行顺序以减轻顺序偏差。
输出平均值、中位数，以及串行平均值/并行平均值、串行中位数/并行中位数
两种加速比；大于 1 表示加速。原始数据和汇总保存到
`benchmark_results/raw.csv` 和 `benchmark_results/summary.csv`。

对于很短的计算，可增大每个进程内的重复次数以降低计时波动：

```bash
python3 benchmark.py --runs 1000 --repeats 100 --strategies serial omp2 thread
python3 benchmark.py --runs 1000 --threads 8 -- --rows 20000 --cols 200000
```

`--repeats 100` 时，每个样本是进程内 100 次调用的平均耗时，最终中位数
是这些平均耗时的中位数。脚本始终包含串行基线。运行失败时停止，已完成
的轮次保留在原始 CSV 中。这些统计不提供硬件计数器数据。

`--only` 可选值为 `all`（默认）、`serial`、`thread`、`omp1` 至 `omp5`。
默认模式仍输出全部实现的时间和 OpenMP 加速比。单策略模式只计时选中的
实现，不额外计时串行基线；加速比需用独立串行运行的时间计算。
每次运行仍会生成矩阵、计算一次串行正确性参考、预热并校验选中的实现。
这些准备工作在 chrono 计时之外，但会计入进程级 perf 统计。

```bash
g++ -O2 -g -std=c++17 -fopenmp -pthread csr_spmv_lab.cpp -o csr_spmv_lab
lscpu
uname -a
perf list

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

当前 WSL2 环境实测不支持 `cycles`、`instructions`、`cache-misses` 和
`branch-misses`，因此上述命令使用软件事件。若环境支持，可将硬件事件
加入 `-e`。事件是否可用应以实际运行结果为准，不能仅凭 `perf list` 判断。
OpenMP 绑定变量只控制 OpenMP 线程，`std::thread` 未配置显式亲和性。

可选热点采样（当前环境使用软件事件）：

```bash
perf record -e cpu-clock --call-graph dwarf \
    ./csr_spmv_lab --only omp2 --threads 8 --repeats 1000
perf report
```

三种实现的实测数据和分析见 [PERFORMANCE.md](PERFORMANCE.md)，
原始输出及环境记录保存在 [perf_results](perf_results/)。
