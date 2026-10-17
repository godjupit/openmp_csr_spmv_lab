#include <arm_neon.h>
#include <omp.h>
#include <pthread.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Eight independent vector FMA chains: 8 * 2 doubles * 2 FLOPs = 32 FLOPs/iteration.
__attribute__((noinline)) double compute_job(std::uint64_t iterations, double seed = 0.0) {
    float64x2_t a = {0.1 + seed, 0.2 + seed}, b = {0.3 + seed, 0.4 + seed},
                  c = {0.5 + seed, 0.6 + seed}, d = {0.7 + seed, 0.8 + seed},
                  e = {0.9 + seed, 1.0 + seed}, f = {1.1 + seed, 1.2 + seed},
                  g = {1.3 + seed, 1.4 + seed}, h = {1.5 + seed, 1.6 + seed};
    const auto mul = vdupq_n_f64(1.0000001);
    const auto add = vdupq_n_f64(0.0000013);
    for (std::uint64_t i = 0; i < iterations; ++i) {
        a = vfmaq_f64(a, mul, add); b = vfmaq_f64(b, mul, add);
        c = vfmaq_f64(c, mul, add); d = vfmaq_f64(d, mul, add);
        e = vfmaq_f64(e, mul, add); f = vfmaq_f64(f, mul, add);
        g = vfmaq_f64(g, mul, add); h = vfmaq_f64(h, mul, add);
    }
    return vaddvq_f64(vaddq_f64(vaddq_f64(vaddq_f64(a,b),vaddq_f64(c,d)),
                              vaddq_f64(vaddq_f64(e,f),vaddq_f64(g,h))));
}

double thread_cpu_time() {
    timespec t{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t)) throw std::runtime_error("thread CPU clock failed");
    return t.tv_sec + t.tv_nsec * 1e-9;
}

struct JobSample {
    std::size_t begin_cpu, end_cpu;
    double wall_ms, cpu_ms;
};

struct alignas(64) Stats {
    double active = 0, cpu = 0, checksum = 0, seed_sum = 0;
    std::uint64_t units = 0;
    std::map<std::size_t, int> observed_cpus;
    std::vector<JobSample> jobs;
};

int main(int argc, char** argv) {
    try {
        int threads = 10, jobs = 120, passes = 12;
        std::uint64_t iterations = 1000000, elements = 12000000;
        std::string mode = "compute";
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (++i == argc) throw std::runtime_error("missing argument");
            const std::string value = argv[i];
            if (key == "--threads") threads = std::stoi(value);
            else if (key == "--jobs") jobs = std::stoi(value);
            else if (key == "--iterations") iterations = std::stoull(value);
            else if (key == "--elements") elements = std::stoull(value);
            else if (key == "--passes") passes = std::stoi(value);
            else if (key == "--mode") mode = value;
            else throw std::runtime_error("unknown argument");
        }
        if (threads < 1 || threads > 10 || jobs < 1 || passes < 1 || !iterations || !elements ||
            jobs % threads || elements % threads || (mode != "compute" && mode != "stream"))
            throw std::runtime_error("invalid configuration: equal per-thread work required");
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        std::vector<Stats> stats(threads);
        std::vector<double> x, y, z;
        double expected_job = 0;
        if (mode == "stream") {
            x.resize(elements); y.resize(elements); z.resize(elements);
            #pragma omp parallel for schedule(static)
            for (std::uint64_t i = 0; i < elements; ++i) {
                x[i] = 1.0 + double(i % 17) * 0.01;
                y[i] = 2.0 + double(i % 13) * 0.01;
                z[i] = 0;
            }
        } else {
            expected_job = compute_job(iterations);
            const double expected = 13.6 + 16.0 * double(iterations) * (1.0000001 * 0.0000013);
            if (!std::isfinite(expected_job) || std::abs(expected_job - expected) > 1e-6)
                throw std::runtime_error("FMA recurrence validation failed");
        }
        // Initialize the team and warm the relevant kernel before timing.
        int actual = 0;
        #pragma omp parallel
        {
            #pragma omp single
            actual = omp_get_num_threads();
            if (mode == "compute") stats[omp_get_thread_num()].checksum = compute_job(10000);
            else {
                #pragma omp for simd schedule(static)
                for (std::uint64_t i = 0; i < elements; ++i) z[i] = x[i] + 0.5 * y[i];
            }
        }
        if (actual != threads) throw std::runtime_error("requested thread count not available");
        double begin = 0, end = 0;
        #pragma omp parallel
        {
            auto& local = stats[omp_get_thread_num()];
            local.checksum = 0;
            local.jobs.reserve(jobs / threads);
            #pragma omp single
            begin = omp_get_wtime();
            const double started = omp_get_wtime();
            const double cpu_started = thread_cpu_time();
            if (mode == "compute") {
                #pragma omp for schedule(static) nowait
                for (int job = 0; job < jobs; ++job) {
                    std::size_t first_cpu = 999, last_cpu = 999;
                    pthread_cpu_number_np(&first_cpu);
                    const double job_start = omp_get_wtime();
                    const double job_cpu = thread_cpu_time();
                    const double seed = job * 0.0001;
                    local.checksum += compute_job(iterations, seed);
                    const double cpu_elapsed = thread_cpu_time() - job_cpu;
                    const double job_elapsed = omp_get_wtime() - job_start;
                    pthread_cpu_number_np(&last_cpu);
                    local.jobs.push_back({first_cpu, last_cpu, job_elapsed * 1000, cpu_elapsed * 1000});
                    ++local.observed_cpus[first_cpu]; ++local.observed_cpus[last_cpu];
                    local.seed_sum += seed;
                    ++local.units;
                }
            } else {
                for (int pass = 0; pass < passes; ++pass) {
                    #pragma omp for simd schedule(static) nowait
                    for (std::uint64_t i = 0; i < elements; ++i) z[i] = x[i] + 0.5 * y[i];
                }
                local.units = elements / threads * passes;
            }
            local.cpu = thread_cpu_time() - cpu_started;
            local.active = omp_get_wtime() - started;
            #pragma omp barrier
            #pragma omp single
            end = omp_get_wtime();
        }
        if (mode == "compute") {
            for (const auto& local : stats)
                if (local.units != std::uint64_t(jobs / threads) ||
                    std::abs(local.checksum - (expected_job * jobs / threads + 16.0 * local.seed_sum)) > 1e-8 * std::abs(local.checksum))
                    throw std::runtime_error("compute output/work count mismatch");
        } else {
            for (std::uint64_t i = 0; i < elements; ++i)
                if (z[i] != x[i] + 0.5 * y[i]) throw std::runtime_error("stream output mismatch");
        }
        const double work = mode == "compute" ? double(jobs) * iterations * 32.0 : double(elements) * passes * 24.0;
        std::cout << std::setprecision(12) << "{\"mode\":\"" << mode << "\",\"threads\":" << actual
                  << ",\"elapsed_ms\":" << (end - begin) * 1000
                  << ",\"work\":" << work << ",\"throughput\":" << work / (end - begin) / 1e9
                  << ",\"correct\":true,\"workers\":[";
        for (int tid = 0; tid < threads; ++tid) {
            const auto& local = stats[tid];
            if (tid) std::cout << ',';
            std::cout << "{\"tid\":" << tid << ",\"units\":" << local.units
                      << ",\"active_ms\":" << local.active * 1000 << ",\"cpu_ms\":" << local.cpu * 1000
                      << ",\"cpu_samples\":{";
            bool first = true;
            for (auto [core, count] : local.observed_cpus) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << '"' << core << "\":" << count;
            }
            std::cout << "},\"jobs\":[";
            bool first_job = true;
            for (auto sample : local.jobs) {
                if (!first_job) std::cout << ',';
                first_job = false;
                std::cout << "{\"begin_cpu\":" << sample.begin_cpu << ",\"end_cpu\":" << sample.end_cpu
                          << ",\"wall_ms\":" << sample.wall_ms << ",\"cpu_ms\":" << sample.cpu_ms << '}';
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
