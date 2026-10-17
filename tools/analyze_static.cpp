// Reuse the current matrix generator, serial reference and production static kernel.
#define main csr_lab_original_main
#include "../csr_spmv_lab.cpp"
#undef main

struct alignas(64) StaticThreadStats {
    int rows = 0;
    int long_rows = 0;
    int first_row = -1;
    int last_row = -1;
    std::size_t nnz = 0;
    double compute_seconds = 0.0;
    double barrier_seconds = 0.0;
};

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads);
        const CsrMatrix matrix = make_skewed_matrix(options);
        const std::vector<double> x = make_vector(options.cols, options.seed);
        std::vector<double> reference(matrix.rows), output(matrix.rows);
        spmv_serial(matrix, x, reference);
        std::vector<StaticThreadStats> stats(options.threads);
        int actual_threads = 0;

        // Untimed ownership pass: observe OpenMP's actual static partition.
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& local = stats[tid];
            #pragma omp single
            actual_threads = omp_get_num_threads();
            #pragma omp for schedule(static)
            for (int row = 0; row < matrix.rows; ++row) {
                const int count = matrix.row_ptr[row + 1] - matrix.row_ptr[row];
                ++local.rows;
                local.nnz += count;
                if (count > options.normal_nnz) ++local.long_rows;
                if (local.first_row < 0) local.first_row = row;
                local.last_row = row;
            }
        }
        if (actual_threads != options.threads) throw std::runtime_error("runtime did not create the requested threads");

        // Production measurements do not contain thread-level instrumentation.
        const double serial_ms = benchmark_ms(
            [&](std::vector<double>& y) { spmv_serial(matrix, x, y); },
            reference, output, options.repeats);
        const double static_ms = benchmark_ms(
            [&](std::vector<double>& y) { spmv_omp_student_4(matrix, x, y); },
            reference, output, options.repeats);
        if (!check_result(reference, output)) throw std::runtime_error("static result failed");

        // Diagnostic pass: persistent team; each call ends at an explicit barrier.
        // Record computation before the barrier and the barrier phase separately.
        #pragma omp parallel
        {
            auto& local = stats[omp_get_thread_num()];
            for (int repeat = -1; repeat < options.repeats; ++repeat) {
                #pragma omp barrier
                const double start = omp_get_wtime();
                #pragma omp for schedule(static) nowait
                for (int row = 0; row < matrix.rows; ++row) {
                    double sum = 0.0;
                    for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p) {
                        sum += matrix.values[p] * x[matrix.col_idx[p]];
                    }
                    output[row] = sum;
                }
                const double computed = omp_get_wtime();
                #pragma omp barrier
                const double synchronized = omp_get_wtime();
                if (repeat >= 0) {
                    local.compute_seconds += computed - start;
                    local.barrier_seconds += synchronized - computed;
                }
            }
        }
        if (!check_result(reference, output)) throw std::runtime_error("diagnostic result failed");
        std::cout << std::setprecision(12)
                  << "{\"serial_ms\":" << serial_ms << ",\"static_ms\":" << static_ms
                  << ",\"actual_threads\":" << actual_threads
                  << ",\"matrix_nnz\":" << matrix.values.size()
                  << ",\"correct\":true,\"threads\":[";
        for (int tid = 0; tid < actual_threads; ++tid) {
            const auto& local = stats[tid];
            if (tid) std::cout << ',';
            std::cout << "{\"tid\":" << tid << ",\"rows\":" << local.rows
                      << ",\"first_row\":" << local.first_row << ",\"last_row\":" << local.last_row
                      << ",\"long_rows\":" << local.long_rows << ",\"nnz\":" << local.nnz
                      << ",\"compute_ms\":" << local.compute_seconds * 1000.0 / options.repeats
                      << ",\"barrier_ms\":" << local.barrier_seconds * 1000.0 / options.repeats << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
