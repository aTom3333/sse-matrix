#ifndef UTILS_HPP
#define UTILS_HPP

#ifdef __GNUC__
#include <x86intrin.h>
#endif

inline unsigned int ctz(size_t a)
{    
    #ifdef __GNUC__
    return __builtin_ctzl(a);
    #elif defined(_MSC_VER)
    return _ctzl(a);
    #else
        unsigned c;
        if (n) {
            n = (n ^ (n - 1)) >> 1;
            for (c = 0; n; c++)
                n >>= 1;
            return c;
        } else {
            return sizeof(a)*8;
        }
    #endif
}


#endif // UTILS_HPP