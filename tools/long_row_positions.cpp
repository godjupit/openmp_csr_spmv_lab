#define main csr_lab_original_main
#include "../csr_spmv_lab.cpp"
#undef main

int main() {
    const Options options;
    const CsrMatrix matrix = make_skewed_matrix(options);
    std::cout << "{\"rows\":" << matrix.rows << ",\"cols\":" << matrix.cols
              << ",\"seed\":" << options.seed << ",\"nnz\":" << matrix.values.size()
              << ",\"normal_nnz\":" << options.normal_nnz << ",\"long_nnz\":" << options.long_nnz
              << ",\"positions\":[";
    bool first = true;
    for (int row = 0; row < matrix.rows; ++row) {
        if (matrix.row_ptr[row + 1] - matrix.row_ptr[row] != options.long_nnz) continue;
        if (!first) std::cout << ',';
        first = false;
        std::cout << row;
    }
    std::cout << "]}\n";
}
