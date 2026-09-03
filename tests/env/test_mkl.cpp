#include <mkl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        std::cout << "[TEST] oneMKL\n";

        const double A[4] = {
            1.0, 2.0,
            3.0, 4.0};

        const double B[4] = {
            5.0, 6.0,
            7.0, 8.0};

        double C[4] = {};

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            2,
            2,
            2,
            1.0,
            A,
            2,
            B,
            2,
            0.0,
            C,
            2);

        const double expected[4] = {
            19.0, 22.0,
            43.0, 50.0};

        for (int i = 0; i < 4; ++i) {
            if (std::abs(C[i] - expected[i]) > 1e-12) {
                throw std::runtime_error("DGEMM result mismatch");
            }
        }

        std::cout << "       DGEMM OK\n";
        std::cout << "[OK] oneMKL works\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
