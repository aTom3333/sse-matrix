#include "Matrix.hpp"

int main() {


    Matrix<float, 12, 4> a{1, 2, 3, 4,
                           5, 6, 7, 8, 
                           9, 10, 11, 12, 
                           13, 14, 15, 16, 
                           17, 18, 19, 20, 
                           21, 22, 23, 24, 
                           25, 26, 27, 28, 
                           29, 30, 31, 32, 
                           33, 34, 35, 36, 
                           37, 38, 39, 40, 
                           41, 42, 43, 44, 
                           45, 46, 47, 48};
    
    Matrix<float, 4, 12> b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
                           25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 
                           37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
    
    Matrix<float, 3, 4>  c{1, 2, 3, 
                           4, 5, 6, 
                           7, 8, 9, 
                           10, 11, 12};
    
    Matrix<float, 4, 7>  d{1, 2, 3, 4, 
                           5, 6, 7, 8, 
                           9, 10, 11, 12, 
                           13, 14, 15, 16, 
                           17, 18, 19, 20, 
                           21, 22, 23, 24, 
                           25, 26, 27, 28};
    
    Matrix<float, 4, 7>  e{1, 2, 3, 4, 
                           5, 6, 7, 8, 
                           9, 10, 11, 12, 
                           13, 14, 15, 16, 
                           17, 18, 19, 20, 
                           21, 22, 23, 24, 
                           25, 26, 27, 28};

    auto r = multiply(c, d);

    auto sum = d + e;
    auto dif = sum - e;


    auto result = a * b;

    std::cout << r << std::endl;
    std::cout << sum << std::endl;
    std::cout << dif << std::endl;
    
    std::cout << result << std::endl;

    static_assert(HAS_SSE == 1);

    //std::cout << transpose(r) << std::endl;

    // std::cin.ignore();
}