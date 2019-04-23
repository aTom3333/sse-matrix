#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <cstddef>
#include "test_macro.hpp"


#if HAS_AVX
    #define SSE_ALIGN 32
#elif HAS_SSE4_1 // Essayer d'autoriser le SSE avec des versions antérieures
    #define SSE_ALIGN 16
#else
    #define SSE_ALIGN 0
#endif

#define ALIGN_FOR(T) std::max<size_t>(SSE_ALIGN, alignof(T))


template<typename FP, size_t ROW, size_t COL>
struct matrix {
    alignas(ALIGN_FOR(FP)) float data[ROW*COL] = {};

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


#endif // MATRIX_HPP