#define main csr_lab_original_main
#include "../csr_spmv_lab.cpp"
#undef main

static void guided_chunk(const CsrMatrix& matrix, const std::vector<double>& x,
                         std::vector<double>& y, int chunk) {
    #pragma omp parallel for schedule(guided, chunk)
    for (int row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p)
            sum += matrix.values[p] * x[matrix.col_idx[p]];
        y[row] = sum;
    }
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::invalid_argument("first argument: chunk (0=serial), followed by matrix/thread/repeat options");
        const int chunk = std::stoi(argv[1]);
        const std::vector<int> allowed{0, 1, 4, 8, 16, 32, 64, 128};
        if (std::find(allowed.begin(), allowed.end(), chunk) == allowed.end())
            throw std::invalid_argument("unsupported chunk");
        const Options options = parse_options(argc - 1, argv + 1);
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads);
        const auto matrix = make_skewed_matrix(options);
        const auto x = make_vector(options.cols, options.seed);
        std::vector<double> reference(matrix.rows), output(matrix.rows);
        spmv_serial(matrix, x, reference);
        const double elapsed = benchmark_ms([&](auto& y) {
            if (chunk == 0) spmv_serial(matrix, x, y);
            else guided_chunk(matrix, x, y, chunk);
        }, reference, output, options.repeats);
        if (!check_result(reference, output)) throw std::runtime_error("final check failed");
        std::cout << std::setprecision(12) << "{\"chunk\":" << chunk << ",\"time_ms\":" << elapsed
                  << ",\"warmup\":1,\"repeats\":" << options.repeats << ",\"correct\":true}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
