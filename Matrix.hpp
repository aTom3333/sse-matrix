#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <iostream>
#include <cstddef>
#include "test_macro.hpp"

#if HAS_SSE
    #include <immintrin.h>
#endif

#if HAS_AVX
    #define SSE_ALIGN 32
#elif HAS_SSE4_1 // Essayer d'autoriser le SSE avec des versions antérieures
    #define SSE_ALIGN 16
#else
    #define SSE_ALIGN 0
#endif

#define ALIGN_FOR(T) SSE_ALIGN < alignof(T) ? alignof(T) : SSE_ALIGN


template<typename FP, size_t ROW, size_t COL>
class Matrix {
public:
    Matrix() = default;
    Matrix(std::initializer_list<FP>);
    
    // accessor
    constexpr size_t width() const noexcept;
    constexpr size_t height() const noexcept;
    
    FP const& operator()(size_t i, size_t j) const noexcept;
    FP& operator()(size_t i, size_t j) noexcept;
    
    FP const* data() const noexcept;
    FP* data() noexcept;
    
    
private:
    alignas(ALIGN_FOR(FP)) float d[ROW*COL] = {};
};


template<typename FP, size_t ROW, size_t COL>
Matrix<FP, ROW, COL>::Matrix(std::initializer_list<FP> init)
{
    std::copy_n(init.begin(), std::min(init.size(), ROW*COL), d);
}

template<typename FP, size_t ROW, size_t COL>
constexpr size_t Matrix<FP, ROW, COL>::width() const noexcept
{
    return COL;
}

template<typename FP, size_t ROW, size_t COL>
constexpr size_t Matrix<FP, ROW, COL>::height() const noexcept
{
    return ROW;
}

template<typename FP, size_t ROW, size_t COL>
FP const& Matrix<FP, ROW, COL>::operator()(size_t i, size_t j) const noexcept
{
    return d[i*width() + j];
}

template<typename FP, size_t ROW, size_t COL>
FP& Matrix<FP, ROW, COL>::operator()(size_t i, size_t j) noexcept
{
    return d[i*width() + j];
}

template<typename FP, size_t ROW, size_t COL>
FP const* Matrix<FP, ROW, COL>::data() const noexcept
{
    return d;
}

template<typename FP, size_t ROW, size_t COL>
FP* Matrix<FP, ROW, COL>::data() noexcept
{
    return d;
}

template <typename FP, size_t ROW, size_t COL>
Matrix<FP, COL, ROW> transpose(const Matrix<FP, ROW, COL>& m){
    Matrix<FP, COL, ROW> output;
    for (size_t i = 0; i < m.height(); ++i)
        for (size_t j = 0; j < m.width(); ++j){
            output(j, i) = m(i, j);
        }
    return output;
}


namespace {
    namespace detail {
        template<typename FP, size_t ROW, size_t COL1, size_t COL2>
        struct multiply_helper
        {
            using return_type = Matrix<FP, ROW, COL2>;
            using op1_type = Matrix<FP, ROW, COL1>;
            using op2_type = Matrix<FP, COL1, COL2>;

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

        #if HAS_SSE4_1
        template<size_t ROW, size_t COL2>
        struct multiply_helper<float, ROW, 4, COL2>
        {
            using return_type = Matrix<float, ROW, COL2>;
            using op1_type = Matrix<float, ROW, 4>;
            using op2_type = Matrix<float, 4, COL2>;

            // Implementation using SSE
            static return_type multiply(op1_type const& a, op2_type const& b)
            {
                auto b_transposed = transpose(b);

                return_type result;

                for(size_t line = 0; line < a.height(); ++line)
                {
                    __m128 cur_line = _mm_load_ps(a.data() + line*a.width());

                    size_t done_cols = 0;
                    while(done_cols < b.width())
                    {
                        switch(b.width()-done_cols)
                        {
                            case 0: break; // Rien à faire mais on ne devrait jamais arriver là
                            case 1:
                            #if !HAS_AVX
                            default:
                            #endif
                            {
                                // SSE 128bits
                                __m128 col = _mm_load_ps(b_transposed.data() + done_cols*b.height());
                                __m128 r = _mm_dp_ps(cur_line, col, 0xF1);
                                result(line, done_cols) = _mm_cvtss_f32(r);
                                ++done_cols;
                                break;
                            }

                            #if HAS_AVX
                            case 2:
                            case 3:
                            #if !HAS_AVX512
                            default:
                            #endif
                            {
                                // SSE 256bits
                                __m256 line_doubled = _mm256_castps128_ps256(cur_line);
                                line_doubled = _mm256_insertf128_ps(line_doubled, cur_line, 1);
                                __m256 cols = _mm256_load_ps(b_transposed.data() + done_cols*b.height());
                                __m256 r = _mm256_mul_ps(line_doubled, cols);
                                r = _mm256_hadd_ps(r, r);
                                r = _mm256_hadd_ps(r, r);
                                result(line, done_cols) = _mm256_cvtss_f32(r);
                                r = _mm256_permute2f128_ps(r, r, 1);
                                result(line, done_cols+1) = _mm256_cvtss_f32(r);
                                done_cols += 2;
                                break;
                            }

                            #if HAS_AVX512
                            default: // 4 and over
                            {
                                // SSE 512bits
                                __m256 line_doubled = _mm256_castps128_ps256(cur_line);
                                line_doubled = _mm256_insertf128_ps(line_doubled, cur_line, 1);
                                __m512 line_quad = _mm512_castps256_ps512(line_doubled);
                                line_quad = _mm512_insertf32x8(line_quad, line_doubled, 1);
                                __m512 cols = _mm512_load_ps(b_transposed.data() + done_cols*b.height());
                                __m512 r = _mm512_mul_ps(line_quad, cols);
                                __m256 lo = _mm512_castps512_ps256(r);
                                __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(r),1));
                                
                                lo = _mm256_hadd_ps(lo, lo);
                                lo = _mm256_hadd_ps(lo, lo);
                                hi = _mm256_hadd_ps(hi, hi);
                                hi = _mm256_hadd_ps(hi, hi);

                                result(line, done_cols) = _mm256_cvtss_f32(lo);
                                lo = _mm256_permute2f128_ps(lo, lo, 1);
                                result(line, done_cols) = _mm256_cvtss_f32(lo);
                                
                                result(line, done_cols+2) = _mm256_cvtss_f32(hi);
                                hi = _mm256_permute2f128_ps(hi, hi, 1);
                                result(line, done_cols+3) = _mm256_cvtss_f32(hi);
                                done_cols += 4;
                                break;
                            }
                            #endif
                            #endif
                        }
                    }
                }

                return result;
            }
        };
        #endif

    }
}



template<typename FP, size_t ROW, size_t COL1, size_t COL2>
Matrix<FP, ROW, COL2> multiply(Matrix<FP, ROW, COL1> const& a, Matrix<FP, COL1, COL2> const& b)
{
    return detail::multiply_helper<FP, ROW, COL1, COL2>::multiply(a, b);
}


template<typename FP, size_t ROW, size_t COL>
std::ostream& operator<<(std::ostream& o, Matrix<FP, ROW, COL> const& m) {
    for (size_t i = 0; i < m.height(); ++i) {
        o << "[ ";
        for (size_t j = 0; j < m.width(); ++j) {
            o << m(i, j) << " ";
        }
        o << "]\n";
    }
    return o;
}



#endif // MATRIX_HPP