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

int main() {

	float array0[ 4 ] __attribute__ ((aligned(16))) = { 0.0f, 1.0f, 2.0f, 3.0f };
	float array1[ 4 ] __attribute__ ((aligned(16)));

	uint64_t t;
	t = rdtsc();
	for (int i = 0; i < 4; i++) {
		array1[i] = array0[i];
	}
	t = rdtsc() - t;

	std::cout << t << std::endl;
}