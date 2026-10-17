#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct CsrMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};

struct Options {
    int rows = 10000;
    int cols = 100000;
    int normal_nnz = 32;
    int long_rows = 32;
    int long_nnz = 20000;
    int threads = 8;
    int repeats = 10;
    unsigned seed = 20260926u;
    std::string only = "all";
    bool no_baseline = false;
};

static void usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --rows N          number of rows, default 10000\n"
        << "  --cols N          number of columns, default 100000\n"
        << "  --normal-nnz N    nnz per ordinary row, default 32\n"
        << "  --long-rows N     number of skewed long rows, default 32\n"
        << "  --long-nnz N      nnz per long row, default 20000\n"
        << "  --threads N       std::thread/OpenMP threads, default 8\n"
        << "  --repeats N       timed repetitions, default 10\n"
        << "  --only NAME       all (default), serial, thread, omp1 ... omp4\n"
        << "  --no-baseline     skip timed serial comparison in single-strategy mode\n"
        << "  --seed N          fixed random seed, default 20260926\n";
}

static int read_int_arg(int argc, char** argv, int& i, const std::string& name) {
    if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + name);
    }
    return std::stoi(argv[++i]);
}

static Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-baseline") {
            options.no_baseline = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "--rows") {
            options.rows = read_int_arg(argc, argv, i, arg);
        } else if (arg == "--cols") {
            options.cols = read_int_arg(argc, argv, i, arg);
        } else if (arg == "--normal-nnz") {
            options.normal_nnz = read_int_arg(argc, argv, i, arg);
        } else if (arg == "--long-rows") {
            options.long_rows = read_int_arg(argc, argv, i, arg);
        } else if (arg == "--long-nnz") {
            options.long_nnz = read_int_arg(argc, argv, i, arg);
        } else if (arg == "--threads") {
            options.threads = read_int_arg(argc, argv, i, arg);
        } else if (arg == "--repeats") {
            options.repeats = read_int_arg(argc, argv, i, arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<unsigned>(read_int_arg(argc, argv, i, arg));
        } else if (arg == "--only") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for --only");
            }
            options.only = argv[++i];
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (options.rows <= 0 || options.cols <= 0 || options.normal_nnz < 0 ||
        options.long_rows < 0 || options.long_rows > options.rows ||
        options.long_nnz < 0 || options.long_nnz > options.cols ||
        options.threads <= 0 || options.repeats <= 0) {
        throw std::invalid_argument("invalid option value");
    }
    if (options.only != "all" && options.only != "serial" &&
        options.only != "thread" && options.only != "omp1" &&
        options.only != "omp2" && options.only != "omp3" && options.only != "omp4") {
        throw std::invalid_argument("unknown --only implementation: " + options.only);
    }
#ifndef _OPENMP
    if (options.only.compare(0, 3, "omp") == 0) {
        throw std::invalid_argument("--only omp requires compilation with -fopenmp");
    }
#endif
    return options;
}

static CsrMatrix make_skewed_matrix(const Options& options) {
    CsrMatrix matrix;
    matrix.rows = options.rows;
    matrix.cols = options.cols;
    matrix.row_ptr.resize(static_cast<std::size_t>(matrix.rows) + 1, 0);

    std::mt19937 rng(options.seed);
    std::vector<int> row_ids(matrix.rows);
    std::iota(row_ids.begin(), row_ids.end(), 0);
    std::shuffle(row_ids.begin(), row_ids.end(), rng);
    std::vector<char> is_long(matrix.rows, false);
    for (int i = 0; i < options.long_rows; ++i) {
        is_long[row_ids[i]] = true;
    }

    std::uniform_real_distribution<double> value_dist(-1.0, 1.0);
    std::uniform_int_distribution<int> column_dist(0, options.cols - 1);

    for (int row = 0; row < matrix.rows; ++row) {
        const int nnz = is_long[row] ? options.long_nnz : options.normal_nnz;
        matrix.row_ptr[row + 1] = matrix.row_ptr[row] + nnz;
        for (int k = 0; k < nnz; ++k) {
            matrix.col_idx.push_back(column_dist(rng));
            matrix.values.push_back(value_dist(rng));
        }
    }
    return matrix;
}

static std::vector<double> make_vector(int size, unsigned seed) {
    std::mt19937 rng(seed ^ 0x9e3779b9u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> x(size);
    for (double& value : x) {
        value = dist(rng);
    }
    return x;
}

static void spmv_serial(const CsrMatrix& matrix, const std::vector<double>& x,
                        std::vector<double>& y) {
    for (int row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p) {
            sum += matrix.values[p] * x[matrix.col_idx[p]];
        }
        y[row] = sum;
    }
}

static void spmv_std_thread(const CsrMatrix& matrix, const std::vector<double>& x,
                            std::vector<double>& y, int thread_count) {
    const int workers = std::max(1, std::min(thread_count, matrix.rows));
    std::vector<std::thread> threads;
    threads.reserve(workers);

    for (int worker = 0; worker < workers; ++worker) {
        const int begin = matrix.rows * worker / workers;
        const int end = matrix.rows * (worker + 1) / workers;
        threads.emplace_back([&, begin, end]() {
            for (int row = begin; row < end; ++row) {
                double sum = 0.0;
                for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p) {
                    sum += matrix.values[p] * x[matrix.col_idx[p]];
                }
                y[row] = sum;
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

// 策略 1：guided 按行调度，逐步缩小任务块，平衡长短行的工作量。
static void spmv_omp_student_1(const CsrMatrix& matrix,
                                     const std::vector<double>& x,
                                     std::vector<double>& y) {
    #pragma omp parallel for schedule(guided)
    for (int row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        const auto begin = matrix.row_ptr[row];
        const auto end = matrix.row_ptr[row + 1];
        for (int p = begin; p < end; ++p) {
            sum += matrix.values[p] * x[matrix.col_idx[p]];
        }
        y[row] = sum;
    }
}

// 策略 2：在 guided 行并行的基础上，用 SIMD 归约加速行内点积。
static void spmv_omp_student_2(const CsrMatrix& matrix,
                                      const std::vector<double>& x,
                                      std::vector<double>& y) {
    #pragma omp parallel for schedule(guided)
    for (int row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p) {
            sum += matrix.values[p] * x[matrix.col_idx[p]];
        }
        y[row] = sum;
    }
}


// 策略 3：以 64 行为任务块动态分配，减少逐行分配的调度开销。
static void spmv_omp_student_3(
    const CsrMatrix& matrix,
    const std::vector<double>& x,
    std::vector<double>& y)
{
    constexpr int BLOCK_ROWS = 64;

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < matrix.rows; b += BLOCK_ROWS) {
        int end = std::min(b + BLOCK_ROWS, matrix.rows);

        for (int row = b; row < end; ++row) {
            double sum = 0.0;
            int start = matrix.row_ptr[row];
            int end = matrix.row_ptr[row + 1];
            for (int p = start;
                p < end; ++p) {
                sum += matrix.values[p] * x[matrix.col_idx[p]];
            }

            y[row] = sum;
        }
    }
}


// 策略 4：最基础的 static 按行并行，行内顺序累加。
static void spmv_omp_student_4(const CsrMatrix& matrix,
                               const std::vector<double>& x,
                               std::vector<double>& y) {
    #pragma omp parallel for schedule(static, 16)
    for (int row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p) {
            sum += matrix.values[p] * x[matrix.col_idx[p]];
        }
        y[row] = sum;
    }
}

static bool check_result(const std::vector<double>& expected,
                         const std::vector<double>& actual,
                         double abs_tol = 1e-10, double rel_tol = 1e-10) {
    if (expected.size() != actual.size()) {
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double error = std::abs(expected[i] - actual[i]);
        const double scale = std::max(1.0, std::abs(expected[i]));
        if (error > abs_tol + rel_tol * scale) {
            std::cerr << "mismatch at row " << i << ": expected=" << std::setprecision(17)
                      << expected[i] << ", actual=" << actual[i] << ", error=" << error
                      << '\n';
            return false;
        }
    }
    return true;
}

template <typename Spmv>
static double benchmark_ms(Spmv&& spmv, const std::vector<double>& reference,
                           std::vector<double>& output, int repeats) {
    spmv(output);
    if (!check_result(reference, output)) {
        throw std::runtime_error("correctness check failed");
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        spmv(output);
    }
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() / repeats;
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const CsrMatrix matrix = make_skewed_matrix(options);
        const std::vector<double> x = make_vector(options.cols, options.seed);
        const std::size_t nnz = matrix.values.size();
        const std::vector<double> zero(matrix.rows, 0.0);
        std::vector<double> reference = zero;
        std::vector<double> output = zero;

        spmv_serial(matrix, x, reference);

        std::cout << "CSR SpMV skewed-matrix experiment\n"
                  << "rows=" << matrix.rows << ", cols=" << matrix.cols
                  << ", nnz=" << nnz << ", long_rows=" << options.long_rows
                  << ", long_nnz=" << options.long_nnz << ", seed=" << options.seed
                  << ", requested_threads=" << options.threads << '\n';
#ifdef _OPENMP
        std::cout << "OpenMP enabled, _OPENMP=" << _OPENMP << '\n';
        omp_set_num_threads(options.threads);
        omp_set_dynamic(0);
#else
        std::cout << "OpenMP disabled: compile with -fopenmp to enable it.\n";
#endif

        // Show a separately warmed and timed serial comparison by default.
        // Profiling tools use --no-baseline to isolate the target implementation.
        if (options.only != "all") {
            double serial_ms = 0.0;
            if (options.only != "serial" && !options.no_baseline) {
                serial_ms = benchmark_ms(
                    [&](std::vector<double>& y) { spmv_serial(matrix, x, y); },
                    reference, output, options.repeats);
            }
            const double elapsed_ms = benchmark_ms(
                [&](std::vector<double>& y) {
                    if (options.only == "serial") spmv_serial(matrix, x, y);
                    else if (options.only == "thread")
                        spmv_std_thread(matrix, x, y, options.threads);
                    else if (options.only == "omp1") spmv_omp_student_1(matrix, x, y);
                    else if (options.only == "omp2") spmv_omp_student_2(matrix, x, y);
                    else if (options.only == "omp3") spmv_omp_student_3(matrix, x, y);
                    else spmv_omp_student_4(matrix, x, y);
                }, reference, output, options.repeats);
            if (!check_result(reference, output)) {
                throw std::runtime_error("final correctness check failed");
            }
            std::cout << std::fixed << std::setprecision(6)
                      << "strategy=" << options.only << ", warmup=1, repeats=" << options.repeats
                      << ", time_ms=" << elapsed_ms;
            if (options.only == "serial" || !options.no_baseline) {
                if (options.only == "serial") serial_ms = elapsed_ms;
                std::cout << ", serial_ms=" << serial_ms
                          << ", speedup=" << serial_ms / elapsed_ms << 'x';
            }
            std::cout << '\n' << "All correctness checks passed.\n";
            return 0;
        }

        const double serial_ms = benchmark_ms(
            [&](std::vector<double>& y) { spmv_serial(matrix, x, y); },
            reference, output, options.repeats);
        const double thread_ms = benchmark_ms(
            [&](std::vector<double>& y) {
                spmv_std_thread(matrix, x, y, options.threads);
            },
            reference, output, options.repeats);
        const double openmp_1_ms = benchmark_ms(
            [&](std::vector<double>& y) {
                spmv_omp_student_1(matrix, x, y);
            },
            reference, output, options.repeats);
        const double openmp_2_ms = benchmark_ms(
            [&](std::vector<double>& y) {
                spmv_omp_student_2(matrix, x, y);
            },
            reference, output, options.repeats);

        // 所有保留策略统一校验并计时。
        const double openmp_3_ms = benchmark_ms(
            [&](std::vector<double>& y) {
                spmv_omp_student_3(matrix, x, y);
            },
            reference, output, options.repeats);
        const double openmp_4_ms = benchmark_ms(
            [&](std::vector<double>& y) {
                spmv_omp_student_4(matrix, x, y);
            },
            reference, output, options.repeats);
        std::cout << std::fixed << std::setprecision(3)
                  << "serial                  " << serial_ms << " ms\n"
                  << "std::thread             " << thread_ms << " ms\n"
                  << "OpenMP 1 (guided)        " << openmp_1_ms << " ms, speedup="
                  << serial_ms / openmp_1_ms << "x\n"
                  << "OpenMP 2 (guided+SIMD)   " << openmp_2_ms << " ms, speedup="
                  << serial_ms / openmp_2_ms << "x\n"
                  << "OpenMP 3 (block dynamic) " << openmp_3_ms << " ms, speedup="
                  << serial_ms / openmp_3_ms << "x\n"
                  << "OpenMP 4 (static)        " << openmp_4_ms << " ms, speedup="
                  << serial_ms / openmp_4_ms << "x\n"
                  << "All correctness checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        usage(argv[0]);
        return 1;
    }
    return 0;
}
