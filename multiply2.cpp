#include "Matrix.hpp"

int main() {
    alignas(16) float a[4] = {1, 2, 3, 4};
    alignas(16) float b[4] = {1, 2, 3, 4};

    __m128 sser_a = _mm_load_ps(a);
    __m128 sser_b = _mm_load_ps(b);

    __m128 sser_result = _mm_dp_ps(sser_a, sser_b, 0b11110001);

    float final = _mm_cvtss_f32(sser_result);
    std::cout << final << std::endl; // Attendu 30

    Matrix<float, 1, 4> c{1, 2, 3, 4};
    Matrix<float, 4, 3> d{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    auto r = multiply(c, d);

    std::cout << r << std::endl;


    //std::cout << transpose(r) << std::endl;


}