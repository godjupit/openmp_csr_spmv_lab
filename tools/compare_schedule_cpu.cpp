#include <ctime>
#define main csr_lab_original_main
#include "../csr_spmv_lab.cpp"
#undef main

// Both variants have the same row body and differ only in scheduling.
template<bool Dynamic>
static void parallel_spmv(const CsrMatrix& matrix, const std::vector<double>& x,
                          std::vector<double>& y) {
    auto row_work = [&](int row) {
        double sum = 0.0;
        for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p)
            sum += matrix.values[p] * x[matrix.col_idx[p]];
        y[row] = sum;
    };
    if constexpr (Dynamic) {
        #pragma omp parallel for schedule(dynamic, 1)
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

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.only != "serial" && options.only != "omp1" && options.only != "omp4")
            throw std::invalid_argument("use serial, omp1 (static), or omp4 (dynamic)");
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads);
        const auto matrix = make_skewed_matrix(options);
        const auto x = make_vector(options.cols, options.seed);
        std::vector<double> reference(matrix.rows), output(matrix.rows);
        spmv_serial(matrix, x, reference);
        auto kernel = [&] {
            if (options.only == "serial") spmv_serial(matrix, x, output);
            else if (options.only == "omp1") parallel_spmv<false>(matrix, x, output);
            else parallel_spmv<true>(matrix, x, output);
        };
        kernel();  // Warm-up and correctness are outside the measured interval.
        if (!check_result(reference, output)) throw std::runtime_error("warm-up check failed");
        const auto start = std::chrono::steady_clock::now();
        const double cpu_start = process_cpu_seconds();
        for (int i = 0; i < options.repeats; ++i) kernel();
        const double cpu_stop = process_cpu_seconds();
        const auto stop = std::chrono::steady_clock::now();
        if (!check_result(reference, output)) throw std::runtime_error("final check failed");
        std::cout << std::setprecision(12) << "{\"wall_ms\":"
                  << std::chrono::duration<double, std::milli>(stop - start).count() / options.repeats
                  << ",\"cpu_ms\":" << (cpu_stop - cpu_start) * 1000 / options.repeats
                  << ",\"correct\":true}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
