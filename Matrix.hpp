#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <iostream>
#include <cstddef>
#include "test_macro.hpp"
#include "utils.hpp"

#if HAS_SSE
    #include <immintrin.h>
#endif

#if HAS_AVX512
#define SSE_ALIGN 64
#elif HAS_AVX
    #define SSE_ALIGN 32
#elif HAS_SSE4_1 // Essayer d'autoriser le SSE avec des versions antérieures
    #define SSE_ALIGN 16
#else
    #define SSE_ALIGN 0
#endif

#define ALIGN_FOR(T) SSE_ALIGN < alignof(T) ? alignof(T) : SSE_ALIGN

// Public interface

template<typename FP, size_t ROW, size_t COL>
class Matrix {
    static_assert(ROW * COL > 0, "Dimensions of the matrix must be strictly positive.");

public:
    Matrix() = default;
    Matrix(std::initializer_list<FP>);
    
    // accessors
    constexpr size_t width() const noexcept;
    constexpr size_t height() const noexcept;
    
    FP const& operator()(size_t i, size_t j) const noexcept;
    FP& operator()(size_t i, size_t j) noexcept;
    
    FP const* data() const noexcept;
    FP* data() noexcept;

    FP const* begin() const noexcept;
    FP* begin() noexcept;
    FP const* end() const noexcept;
    FP* end() noexcept;
    
private:
    alignas(ALIGN_FOR(FP)) float d[ROW*COL] = {};
};

template<typename FP, size_t ROW, size_t COL1, size_t COL2>
Matrix<FP, ROW, COL2> operator*(Matrix<FP, ROW, COL1> const& lhs, Matrix<FP, COL1, COL2> const& rhs);

template<typename FP, size_t ROW, size_t COL>
Matrix<FP, ROW, COL> operator+(Matrix<FP, ROW, COL> const& lhs, Matrix<FP, ROW, COL> const& rhs);

template<typename FP, size_t ROW, size_t COL>
Matrix<FP, ROW, COL> operator-(Matrix<FP, ROW, COL> const& lhs, Matrix<FP, ROW, COL> const& rhs);

template<typename FP, size_t ROW, size_t COL>
std::ostream& operator<<(std::ostream& o, Matrix<FP, ROW, COL> const& m);




// Implementation details

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

template<typename FP, size_t ROW, size_t COL>
FP const* Matrix<FP, ROW, COL>::begin() const noexcept
{
    return data();
}

template<typename FP, size_t ROW, size_t COL>
FP* Matrix<FP, ROW, COL>::begin() noexcept
{
    return data();
}

template<typename FP, size_t ROW, size_t COL>
FP const* Matrix<FP, ROW, COL>::end() const noexcept
{
    return &d[ROW * COL];
}

template<typename FP, size_t ROW, size_t COL>
FP* Matrix<FP, ROW, COL>::end() noexcept
{
    return &d[ROW * COL];
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
                        auto count = ctz((uintptr_t)(void*)(b_transposed.data()+done_cols*b.height()));
                        size_t current_alignment = (1ULL << (count-4));
                        // We mesure the alignment of memory by counting the trailing zero bits
                        
                        
                        // We will use the biggest simd instructions that are available and applicabale
                        // Meaning we use them if there still enough data to fill the simd registers
                        // And the memory is correctly aligned
                        // After reflexion, it would seem that this useless, the only way for the pointer
                        // To not be aligned for the maximum size simd is after executing a smaller simd iteration
                        // Meaning we are at the end of the loop, after that the pointer is reset to the first element
                        // which is correctly aligned
                        switch(b.width()-done_cols)
                        {
                            case 0: 
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

        template<typename FP, size_t ROW, size_t COL>
        struct add_helper {
            using op_type = Matrix<FP, ROW, COL>;

            // Fallback implementation
            static op_type add(op_type const& a, op_type const& b) {
                op_type output;

                size_t done = 0;
                while (done < ROW * COL) {
                    switch (ROW * COL - done) {
                        case 0:
                            break; // Nothing
                        case 1:
                        case 2:
                        case 3:
#if !HAS_SSE4_1
                            default:
#endif
                            // Basic implementation
                            *(output.data() + done) = *(a.data() + done) + *(b.data() + done);
                            ++done;
                            break;

#if HAS_SSE4_1
                        case 4:
                        case 5:
                        case 6:
                        case 7:
#if !HAS_AVX
                            default:
#endif
                        {
                            __m128 lhs = _mm_load_ps(a.data() + done);
                            __m128 rhs = _mm_load_ps(b.data() + done);
                            __m128 r   = _mm_add_ps(lhs, rhs);
                            _mm_store_ps(output.data() + done, r);
                            done += 4;
                            break;
                        }

#if HAS_AVX
                        case 8:
                        case 9:
                        case 10:
                        case 11:
                        case 12:
                        case 13:
                        case 14:
                        case 15:
#if !HAS_AVX512
                        default:
#endif
                        {
                            __m256 lhs = _mm256_load_ps(a.data() + done);
                            __m256 rhs = _mm256_load_ps(b.data() + done);
                            __m256 r   = _mm256_add_ps(lhs, rhs);
                            _mm256_store_ps(output.data() + done, r);
                            done += 8;
                            break;
                        }

#if HAS_AVX512
                        default:
                        {
                            __m512 lhs = _mm512_load_ps(a.data() + done);
                            __m512 rhs = _mm512_load_ps(b.data() + done);
                            __m512 r = _mm512_add_ps(lhs, rhs);
                            _mm512_store_ps(output.data() + done, r);
                            done += 16;
                            break;
                        }
#endif // HAS_AVX512
#endif // HAS_AVX
#endif // HAS_SSE4_1
                    }
                }

                return output;
            }
        };

        template<typename FP, size_t ROW, size_t COL>
        struct sub_helper {
            using op_type = Matrix<FP, ROW, COL>;

            // Fallback implementation
            static op_type sub(op_type const& a, op_type const& b) {
                op_type output;

                size_t done = 0;
                while (done < ROW * COL) {
                    switch (ROW * COL - done) {
                        case 0:
                            break; // Nothing
                        case 1:
                        case 2:
                        case 3:
#if !HAS_SSE4_1
                            default:
#endif
                            // Basic implementation
                            *(output.data() + done) = *(a.data() + done) - *(b.data() + done);
                            ++done;
                            break;

#if HAS_SSE4_1
                        case 4:
                        case 5:
                        case 6:
                        case 7:
#if !HAS_AVX
                            default:
#endif
                        {
                            __m128 lhs = _mm_load_ps(a.data() + done);
                            __m128 rhs = _mm_load_ps(b.data() + done);
                            __m128 r   = _mm_sub_ps(lhs, rhs);
                            _mm_store_ps(output.data() + done, r);
                            done += 4;
                            break;
                        }

#if HAS_AVX
                        case 8:
                        case 9:
                        case 10:
                        case 11:
                        case 12:
                        case 13:
                        case 14:
                        case 15:
#if !HAS_AVX512
                        default:
#endif
                        {
                            __m256 lhs = _mm256_load_ps(a.data() + done);
                            __m256 rhs = _mm256_load_ps(b.data() + done);
                            __m256 r   = _mm256_sub_ps(lhs, rhs);
                            _mm256_store_ps(output.data() + done, r);
                            done += 8;
                            break;
                        }

#if HAS_AVX512
                        default:
                        {
                            __m512 lhs = _mm512_load_ps(a.data() + done);
                            __m512 rhs = _mm512_load_ps(b.data() + done);
                            __m512 r = _mm512_sub_ps(lhs, rhs);
                            _mm512_store_ps(output.data() + done, r);
                            done += 16;
                            break;
                        }
#endif // HAS_AVX512
#endif // HAS_AVX
#endif // HAS_SSE4_1
                    }
                }

                return output;
            }
        };
    }
}

template<typename FP, size_t ROW, size_t COL1, size_t COL2>
Matrix<FP, ROW, COL2> multiply(Matrix<FP, ROW, COL1> const& lhs, Matrix<FP, COL1, COL2> const& rhs)
{
    return detail::multiply_helper<FP, ROW, COL1, COL2>::multiply(lhs, rhs);
}

template<typename FP, size_t ROW, size_t COL>
Matrix<FP, ROW, COL> add(Matrix<FP, ROW, COL> const& lhs, Matrix<FP, ROW, COL> const& rhs) {
    return detail::add_helper<FP, ROW, COL>::add(lhs, rhs);
}

template<typename FP, size_t ROW, size_t COL>
Matrix<FP, ROW, COL> substract(Matrix<FP, ROW, COL> const& lhs, Matrix<FP, ROW, COL> const& rhs) {
    return detail::sub_helper<FP, ROW, COL>::sub(lhs, rhs);
}

template<typename FP, size_t ROW, size_t COL1, size_t COL2>
Matrix<FP, ROW, COL2> operator*(Matrix<FP, ROW, COL1> const& lhs, Matrix<FP, COL1, COL2> const& rhs)
{
    return multiply(lhs, rhs);
}

template<typename FP, size_t ROW, size_t COL>
Matrix<FP, ROW, COL> operator+(Matrix<FP, ROW, COL> const& lhs, Matrix<FP, ROW, COL> const& rhs)
{
    return add<FP, ROW, COL>(lhs, rhs);
}

template<typename FP, size_t ROW, size_t COL>
Matrix<FP, ROW, COL> operator-(Matrix<FP, ROW, COL> const& lhs, Matrix<FP, ROW, COL> const& rhs) {
    return substract<FP, ROW, COL>(lhs, rhs);
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