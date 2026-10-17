#include <omp.h>
#include <pthread.h>
#include <cstdio>
int main() {
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    std::printf("proc_bind=%d num_places=%d\n", int(omp_get_proc_bind()), omp_get_num_places());
    #pragma omp parallel
    {
        std::size_t cpu = 999;
        pthread_cpu_number_np(&cpu);
        #pragma omp critical
        std::printf("thread=%d place=%d current_cpu=%zu\n", omp_get_thread_num(), omp_get_place_num(), cpu);
    }
}
