#include <ctime>
#include <sys/resource.h>
#ifdef __APPLE__
#include <libproc.h>
#include <unistd.h>
#include <os/signpost.h>
#endif
#define main csr_lab_original_main
#include "../csr_spmv_lab.cpp"
#undef main

static double thread_cpu_seconds() {
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0)
        std::abort();
    return value.tv_sec + value.tv_nsec * 1e-9;
}

struct alignas(64) CpuBreakdown {
    double body = 0.0;
    long long rows = 0;
    double empty_interval = 0.0;
    double pair_cost = 0.0;
};

template<int Schedule>
static void instrumented_spmv(const CsrMatrix& matrix, const std::vector<double>& x,
                             std::vector<double>& y, std::vector<CpuBreakdown>& stats) {
    #pragma omp parallel
    {
        auto& local = stats[omp_get_thread_num()];
        auto row_work = [&](int row) {
            const double begin = thread_cpu_seconds();
            double sum = 0.0;
            for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p)
                sum += matrix.values[p] * x[matrix.col_idx[p]];
            y[row] = sum;
            const double end = thread_cpu_seconds();
            local.body += end - begin;
            ++local.rows;
        };
        if constexpr (Schedule == 1) {
            #pragma omp for schedule(dynamic, 1)
            for (int row = 0; row < matrix.rows; ++row) row_work(row);
        } else if constexpr (Schedule == 2) {
            #pragma omp for schedule(guided)
            for (int row = 0; row < matrix.rows; ++row) row_work(row);
        } else {
            #pragma omp for schedule(static)
            for (int row = 0; row < matrix.rows; ++row) row_work(row);
        }
    }
}

// All variants have the same row body and differ only in scheduling.
template<int Schedule>
static void parallel_spmv(const CsrMatrix& matrix, const std::vector<double>& x,
                          std::vector<double>& y) {
    auto row_work = [&](int row) {
        double sum = 0.0;
        for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p)
            sum += matrix.values[p] * x[matrix.col_idx[p]];
        y[row] = sum;
    };
    if constexpr (Schedule == 1) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int row = 0; row < matrix.rows; ++row) row_work(row);
    } else if constexpr (Schedule == 2) {
        #pragma omp parallel for schedule(guided)
        for (int row = 0; row < matrix.rows; ++row) row_work(row);
    } else {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < matrix.rows; ++row) row_work(row);
    }
}

static double process_cpu_seconds() {
    timespec value{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0)
        throw std::runtime_error("process CPU clock failed");
    return value.tv_sec + value.tv_nsec * 1e-9;
}

#ifndef CSR_SCHEDULE_NO_MAIN
int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.only != "serial" && options.only != "omp1" && options.only != "omp2" && options.only != "omp3")
            throw std::invalid_argument("use serial, omp1 (static), omp2 (guided), or omp3 (dynamic)");
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads);
        const auto matrix = make_skewed_matrix(options);
        const auto x = make_vector(options.cols, options.seed);
        std::vector<double> reference(matrix.rows), output(matrix.rows);
        spmv_serial(matrix, x, reference);
        const bool breakdown = std::getenv("CSR_CPU_BREAKDOWN") != nullptr;
        std::vector<CpuBreakdown> stats(options.threads);
        if (breakdown && options.only != "serial") {
            // Calibrate each worker's empty timing interval and full timer pair.
            // Warm up the calibration loop before measuring it.
            #pragma omp parallel
            {
                auto& local = stats[omp_get_thread_num()];
                for (int pass = -1; pass < 1; ++pass) {
                    double intervals = 0.0;
                    const double start = thread_cpu_seconds();
                    for (int i = 0; i < 20000; ++i) {
                        const double a = thread_cpu_seconds();
                        const double b = thread_cpu_seconds();
                        intervals += b - a;
                    }
                    const double end = thread_cpu_seconds();
                    if (pass >= 0) {
                        local.empty_interval = intervals / 20000;
                        local.pair_cost = (end - start) / 20000;
                    }
                }
            }
        }
        auto kernel = [&] {
            if (options.only == "serial") spmv_serial(matrix, x, output);
            else if (breakdown && options.only == "omp1") instrumented_spmv<0>(matrix, x, output, stats);
            else if (breakdown && options.only == "omp2") instrumented_spmv<2>(matrix, x, output, stats);
            else if (breakdown) instrumented_spmv<1>(matrix, x, output, stats);
            else if (options.only == "omp1") parallel_spmv<0>(matrix, x, output);
            else if (options.only == "omp2") parallel_spmv<2>(matrix, x, output);
            else parallel_spmv<1>(matrix, x, output);
        };
        kernel();  // Warm-up and correctness are outside the measured interval.
        if (!check_result(reference, output)) throw std::runtime_error("warm-up check failed");
        for (auto& s : stats) { s.body = 0; s.rows = 0; }
        rusage usage_start{}, usage_end{};
#ifdef __APPLE__
        const auto log = os_log_create("csr.schedules", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
        const auto signpost = os_signpost_id_generate(log);
        os_signpost_interval_begin(log, signpost, "MeasuredSpMV");
        rusage_info_v4 accounting_start{}, accounting_end{};
        if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t*>(&accounting_start)) != 0)
            throw std::runtime_error("proc_pid_rusage failed");
#endif
        if (getrusage(RUSAGE_SELF, &usage_start) != 0) throw std::runtime_error("getrusage failed");
        const auto start = std::chrono::steady_clock::now();
        const double cpu_start = process_cpu_seconds();
        for (int i = 0; i < options.repeats; ++i) kernel();
        const double cpu_stop = process_cpu_seconds();
        const auto stop = std::chrono::steady_clock::now();
        if (getrusage(RUSAGE_SELF, &usage_end) != 0) throw std::runtime_error("getrusage failed");
#ifdef __APPLE__
        if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t*>(&accounting_end)) != 0)
            throw std::runtime_error("proc_pid_rusage failed");
        os_signpost_interval_end(log, signpost, "MeasuredSpMV");
#endif
        if (!check_result(reference, output)) throw std::runtime_error("final check failed");
        std::cout << std::setprecision(12) << "{\"wall_ms\":"
                  << std::chrono::duration<double, std::milli>(stop - start).count() / options.repeats
                  << ",\"cpu_ms\":" << (cpu_stop - cpu_start) * 1000 / options.repeats
                  << ",\"voluntary_switches\":" << usage_end.ru_nvcsw - usage_start.ru_nvcsw
                  << ",\"involuntary_switches\":" << usage_end.ru_nivcsw - usage_start.ru_nivcsw
                  << ",\"repeats\":" << options.repeats
                  << ",\"correct\":true";
#ifdef __APPLE__
        std::cout << ",\"accounted_cycles\":" << accounting_end.ri_cycles - accounting_start.ri_cycles
                  << ",\"accounted_instructions\":" << accounting_end.ri_instructions - accounting_start.ri_instructions;
#endif
        if (breakdown && options.only != "serial") {
            double body = 0, empty = 0, timer = 0;
            long long rows = 0;
            for (const auto& s : stats) {
                body += s.body;
                empty += s.empty_interval * s.rows;
                timer += s.pair_cost * s.rows;
                rows += s.rows;
            }
            const double scale = 1000.0 / options.repeats;
            const double corrected_compute = (body - empty) * scale;
            const double corrected_total = (cpu_stop - cpu_start - timer) * scale;
            std::cout << ",\"body_cpu_raw_ms\":" << body * scale
                      << ",\"timer_interval_bias_ms\":" << empty * scale
                      << ",\"timer_cost_ms\":" << timer * scale
                      << ",\"compute_cpu_est_ms\":" << corrected_compute
                      << ",\"total_cpu_est_ms\":" << corrected_total
                      << ",\"overhead_cpu_est_ms\":" << corrected_total - corrected_compute
                      << ",\"rows_timed\":" << rows;
        }
        std::cout << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
