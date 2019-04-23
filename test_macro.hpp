#ifndef TEST_MACRO_HPP
#define TEST_MACRO_HPP


#if defined(__AVX2__)
    #define HAS_AVX2 1
#else
    #define HAS_AVX2 0
#endif

#if defined(__AVX__)
    #define HAS_AVX 1
#else
    #define HAS_AVX 0
#endif

#if HAS_AVX || defined(__SSE4_2__) // Pas de tests pour msvc
    #define HAS_SSE4_2 1
#else
    #define HAS_SSE4_2 0
#endif

#if HAS_SSE4_2 || defined(__SSE4_1__) // Pas de tests pour msvc
    #define HAS_SSE4_1 1
#else
    #define HAS_SSE4_1 0
#endif

#if HAS_SSE4_1 || defined(__SSE3__) // Pas de tests pour msvc
    #define HAS_SSE3 1
#else
    #define HAS_SSE3 0
#endif

#if HAS_SSE3 || defined(__SSE2__) || ( defined(_M_IX86_FP) && _M_IX86_FP == 2 )
    #define HAS_SSE2 1
#else 
    #define HAS_SSE2 0
#endif

#if HAS_SSE2 || defined(__SSE__) || ( defined(_M_IX86_FP) && _M_IX86_FP == 1 )
    #define HAS_SSE 1
#else 
    #define HAS_SSE 0
#endif

#endif // TEST_MACRO_HPP
