#include "Matrix.hpp"

int main() {


    Matrix<float, 3, 4> c{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    Matrix<float, 4, 7> d{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35};

    auto r = multiply(c, d);

    std::cout << r << std::endl;

    static_assert(HAS_SSE == 1);

    //std::cout << transpose(r) << std::endl;


}