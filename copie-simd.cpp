#include <immintrin.h>
#include <iostream>

/* define this somewhere */
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

int main() {

	// Static arrays are stored into the stack thus we need to add an alignment attribute to tell the compiler to correctly align both arrays.
	float array0[ 4 ] __attribute__ ((aligned(16))) = { 0.0f, 1.0f, 2.0f, 3.0f };
	float array1[ 4 ] __attribute__ ((aligned(16)));

	uint64_t t;
        t = rdtsc();

	// Load 4 values from the first array into a SSE register.
	__m128 r0 = _mm_load_ps(array0);

	// Store the content of the register into the second array.
	_mm_store_ps( array1 , r0 );

        t = rdtsc() - t;

        std::cout << t << std::endl;


	return 0;
}