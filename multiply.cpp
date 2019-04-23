#include <immintrin.h>
#include <iostream>

#ifdef __i386
	__inline__ uint64_t rdtsc() {
	  uint64_t x;
	  __asm__ volatile ("rdtsc" : "=A" (x));
	  return x;
}
#elif __amd64
__inline__ uint64_t rdtsc() {
	uint64_t a, d;
	__asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
	return (d<<32) | a;
}
#endif



template<typename FP, size_t ROW, size_t COL>
struct matrix;

template<size_t ROW, size_t COL>
struct matrix<float, ROW, COL> {
    alignas(64) float data[ROW*COL] = {};

    float const& operator()(size_t i, size_t j) const {
        return data[i*width() + j];
    }

    float& operator()(size_t i, size_t j) {
        return data[i*width() + j];
    }
    
    constexpr size_t width() const {
        return COL;
    }
    constexpr size_t height() const {
        return ROW;
    }
};

template <typename FP, size_t ROW, size_t COL>
matrix<FP, COL, ROW> transpose(const matrix<FP, ROW, COL>& m){
    matrix<FP, COL, ROW> output;
    for (size_t i = 0; i < m.height(); ++i)
        for (size_t j = 0; j < m.width(); ++j){
            output(j, i) = m(i, j);
        }
    return output;
}

template<typename FP, size_t ROW, size_t COL1, size_t COL2>
struct multiply_helper 
{
    using return_type = matrix<FP, ROW, COL2>;
    using op1_type = matrix<FP, ROW, COL1>;
    using op2_type = matrix<FP, COL1, COL2>;
    
    // Fallback implementation
    static return_type multiply(op1_type const& a, op2_type const& b)
    {
        return_type output;

        for (size_t i = 0; i < ROW; ++i) {
            for (size_t j = 0; j < COL2; ++j) {
                for (size_t k = 0; k < COL1; ++k) {
                    output(i, j) += a(i, k) * b(k, j);
                }
            }
        }
        return output;
    }
};

template<size_t ROW, size_t COL2>
struct multiply_helper<float, ROW, 4, COL2>
{
    using return_type = matrix<float, ROW, COL2>;
    using op1_type = matrix<float, ROW, 4>;
    using op2_type = matrix<float, 4, COL2>;

    // Implementation using SSE
    static return_type multiply(op1_type const& a, op2_type const& b)
    {
        auto b_transposed = transpose(b);
        
        return_type result;
        
        for(size_t line = 0; line < a.height(); ++line) 
        {
            __m128 cur_line = _mm_load_ps(a.data + line*a.width());
            
            size_t done_cols = 0;
            while(done_cols < b.width())
            {
                switch(b.width()-done_cols)
                {
                    case 0: break; // Rien à faire mais on ne devrait jamais arriver là
                    case 1:
                    {
                        // SSE 128bits
                        __m128 col = _mm_load_ps(b_transposed.data + done_cols*b.height());
                        __m128 r = _mm_dp_ps(cur_line, col, 0xF1);
                        result(line, done_cols) = _mm_cvtss_f32(r);
                        ++done_cols;
                        break;
                    }
                    
                    case 2:
                    case 3:
                    default:
                    {
                        // SSE 256bits
                        __m256 line_doubled = _mm256_castps128_ps256(cur_line);
                        line_doubled = _mm256_insertf128_ps(line_doubled, cur_line, 1);
                        __m256 cols = _mm256_load_ps(b_transposed.data + done_cols*b.height());
                        __m256 r = _mm256_mul_ps(line_doubled, cols);
                        r = _mm256_hadd_ps(r, r);
                        r = _mm256_hadd_ps(r, r);
                        result(line, done_cols) = _mm256_cvtss_f32(r);
                        r = _mm256_permute2f128_ps(r, r, 1);
                        result(line, done_cols+1) = _mm256_cvtss_f32(r);
                        done_cols += 2;
                        break;
                    }

                    //default: // 4 and over
                    {
                        // SSE 512bits
                        done_cols += 4;
                        break;
                    }
                }
            }    
        }
        
        return result;
    }
};


template<typename FP, size_t ROW, size_t COL1, size_t COL2>
matrix<FP, ROW, COL2> multiply(matrix<FP, ROW, COL1> const& a, matrix<FP, COL1, COL2> const& b) 
{
    return multiply_helper<FP, ROW, COL1, COL2>::multiply(a, b);
}

template<typename FP, size_t ROW, size_t COL>
std::ostream& operator<<(std::ostream& o, matrix<FP, ROW, COL> const& m) {
    for (size_t i = 0; i < m.height(); ++i) {
        o << "[ ";
        for (size_t j = 0; j < m.width(); ++j) {
            o << m(i, j) << " ";
        }
        o << "]" << std::endl;
    }
    return o;
}




int main() {
    alignas(16) float a[4] = {1, 2, 3, 4};
    alignas(16) float b[4] = {1, 2, 3, 4};
    
    __m128 sser_a = _mm_load_ps(a);
    __m128 sser_b = _mm_load_ps(b);
    
    __m128 sser_result = _mm_dp_ps(sser_a, sser_b, 0b11110001);
    
    float final = _mm_cvtss_f32(sser_result);
    std::cout << final << std::endl; // Attendu 30

    matrix<float, 1, 4> c{1, 2, 3, 4};
    matrix<float, 4, 3> d{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    
    auto r = multiply(c, d);
    
    std::cout << r << std::endl;
    
       
    std::cout << transpose(r) << std::endl;
    
    
}