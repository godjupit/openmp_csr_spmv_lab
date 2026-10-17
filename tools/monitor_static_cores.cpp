// Observe native thread IDs, then profile the unmodified static implementation.
#include "exploration_kernels.hpp"
#include <pthread.h>
#include <cstdint>

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        omp_set_num_threads(options.threads);
        omp_set_dynamic(0);
        const auto matrix = make_skewed_matrix(options);
        const auto x = make_vector(options.cols, options.seed);
        std::vector<double> reference(matrix.rows), output(matrix.rows);
        spmv_serial(matrix, x, reference);
        std::vector<std::uint64_t> before(options.threads), after(options.threads);
        std::vector<int> errors(options.threads);
        int actual_threads = 0;
        auto identify = [&](std::vector<std::uint64_t>& ids) {
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                errors[tid] = pthread_threadid_np(nullptr, &ids[tid]);
                #pragma omp single
                actual_threads = omp_get_num_threads();
            }
        };
        identify(before);
        if (actual_threads != options.threads) throw std::runtime_error("thread count mismatch");
        const double measured_ms = benchmark_ms(
            [&](std::vector<double>& y) { spmv_static_baseline(matrix, x, y); },
            reference, output, options.repeats);
        identify(after);
        if (before != after || std::any_of(errors.begin(), errors.end(), [](int e){return e != 0;}))
            throw std::runtime_error("native thread mapping changed or could not be read");
        if (!check_result(reference, output)) throw std::runtime_error("static correctness failed");
        std::cout << std::setprecision(12) << "{\"correct\":true,\"threads\":" << actual_threads
                  << ",\"repeats\":" << options.repeats << ",\"profiled_ms\":" << measured_ms
                  << ",\"thread_mapping\":[";
        for (int tid = 0; tid < actual_threads; ++tid) {
            if (tid) std::cout << ',';
            std::cout << "{\"omp_thread\":" << tid << ",\"native_tid\":" << before[tid] << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
