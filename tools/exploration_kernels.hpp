// Independent exploration baselines; not registered submission strategies.
#define main csr_lab_original_main
#include "../csr_spmv_lab.cpp"
#undef main

static void spmv_static_baseline(const CsrMatrix& matrix, const std::vector<double>& x,
                                 std::vector<double>& y) {
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p)
            sum += matrix.values[p] * x[matrix.col_idx[p]];
        y[row] = sum;
    }
}

static void spmv_dynamic_baseline(const CsrMatrix& matrix, const std::vector<double>& x,
                                  std::vector<double>& y) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p)
            sum += matrix.values[p] * x[matrix.col_idx[p]];
        y[row] = sum;
    }
}
