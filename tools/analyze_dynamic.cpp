#define main csr_lab_original_main
#include "../csr_spmv_lab.cpp"
#undef main

struct alignas(64) DynamicStats {
    long long rows = 0, long_rows = 0, nnz = 0;
    double work = 0.0, barrier = 0.0;
};

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads);
        const auto matrix = make_skewed_matrix(options);
        const auto x = make_vector(options.cols, options.seed);
        std::vector<double> reference(matrix.rows), output(matrix.rows);
        spmv_serial(matrix, x, reference);
        const double serial_ms = benchmark_ms(
            [&](auto& y) { spmv_serial(matrix, x, y); }, reference, output, options.repeats);
        const double dynamic_ms = benchmark_ms(
            [&](auto& y) { spmv_omp_student_4(matrix, x, y); }, reference, output, options.repeats);
        std::vector<DynamicStats> stats(options.threads);
        int actual_threads = 0;
        bool correct = true;
        // A separate instrumented persistent team observes actual dynamic ownership.
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            #pragma omp single
            actual_threads = omp_get_num_threads();
            for (int repeat = -1; repeat < options.repeats; ++repeat) {
                DynamicStats local;
                #pragma omp barrier
                const double start = omp_get_wtime();
                #pragma omp for schedule(dynamic) nowait
                for (int row = 0; row < matrix.rows; ++row) {
                    double sum = 0.0;
                    for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p)
                        sum += matrix.values[p] * x[matrix.col_idx[p]];
                    output[row] = sum;
                    ++local.rows;
                    local.nnz += matrix.row_ptr[row + 1] - matrix.row_ptr[row];
                    if (matrix.row_ptr[row + 1] - matrix.row_ptr[row] > options.normal_nnz)
                        ++local.long_rows;
                }
                const double end = omp_get_wtime();
                #pragma omp barrier
                const double synced = omp_get_wtime();
                if (repeat < 0) {
                    #pragma omp single
                    correct = check_result(reference, output);
                } else {
                    auto& total = stats[tid];
                    total.rows += local.rows;
                    total.long_rows += local.long_rows;
                    total.nnz += local.nnz;
                    total.work += end - start;
                    total.barrier += synced - end;
                }
            }
        }
        if (!correct || !check_result(reference, output)) throw std::runtime_error("diagnostic correctness failed");
        if (actual_threads != options.threads) throw std::runtime_error("unexpected thread count");
        std::cout << std::setprecision(12) << "{\"serial_ms\":" << serial_ms
                  << ",\"dynamic_ms\":" << dynamic_ms << ",\"warmup\":1,\"repeats\":" << options.repeats
                  << ",\"matrix_nnz\":" << matrix.values.size() << ",\"correct\":true,\"threads\":[";
        for (int tid = 0; tid < actual_threads; ++tid) {
            const auto& s = stats[tid];
            if (tid) std::cout << ',';
            std::cout << "{\"tid\":" << tid << ",\"rows\":" << s.rows
                      << ",\"long_rows\":" << s.long_rows << ",\"nnz\":" << s.nnz
                      << ",\"work_ms\":" << s.work * 1000 / options.repeats
                      << ",\"barrier_ms\":" << s.barrier * 1000 / options.repeats << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
