// Reuse the same verified FP64 FMA kernel with equal QoS across all workers.
#define main capacity_original_main
#include "cpu_capacity.cpp"
#undef main
#include <chrono>
#include <pthread/qos.h>

struct PESample { int tid; std::size_t before, after; double wall_ms, cpu_ms; };

int main() {
    try {
        constexpr int jobs = 12000;
        constexpr std::uint64_t iterations = 100000;
        omp_set_dynamic(0);
        omp_set_num_threads(10);
        std::vector<std::vector<PESample>> samples(10);
        std::vector<int> errors(10);
        int actual_threads = 0;
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& local = samples[tid];
            local.reserve(jobs);
            errors[tid] = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
            volatile double warmup = compute_job(iterations, tid * 0.00001);
            (void)warmup;
            #pragma omp single
            actual_threads = omp_get_num_threads();
            #pragma omp barrier
            // Keep all cores supplied with identical-sized tasks. Do not use different QoS.
            #pragma omp for schedule(dynamic,1)
            for (int job = 0; job < jobs; ++job) {
                std::size_t before = 999, after = 999;
                const int query_before = pthread_cpu_number_np(&before);
                const auto wall_start = std::chrono::steady_clock::now();
                const double cpu_start = thread_cpu_time();
                const double seed = job * 0.000001;
                const double value = compute_job(iterations, seed);
                const double cpu_ms = (thread_cpu_time() - cpu_start) * 1000;
                const double wall_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wall_start).count();
                const int query_after = pthread_cpu_number_np(&after);
                const double expected = 13.6 + 16 * seed + 16.0 * iterations * (1.0000001 * 0.0000013);
                if (query_before || query_after || !std::isfinite(value) || std::abs(value - expected) > 1e-7)
                    errors[tid] = -1;
                local.push_back({tid, before, after, wall_ms, cpu_ms});
            }
        }
        if (actual_threads != 10 || std::any_of(errors.begin(), errors.end(), [](int e){return e != 0;}))
            throw std::runtime_error("thread count, QoS, CPU query or output validation failed");
        std::size_t count = 0;
        for (const auto& local : samples) count += local.size();
        if (count != jobs) throw std::runtime_error("job count mismatch");
        std::cout << std::setprecision(12) << "{\"qos\":\"USER_INITIATED\",\"threads\":10,\"jobs\":"
                  << jobs << ",\"flops_per_job\":" << iterations * 32
                  << ",\"correct\":true,\"samples\":[";
        bool first = true;
        for (const auto& local : samples) for (auto s : local) {
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"tid\":" << s.tid << ",\"begin_cpu\":" << s.before << ",\"end_cpu\":" << s.after
                      << ",\"cpu_ms\":" << s.cpu_ms << ",\"wall_ms\":" << s.wall_ms << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
