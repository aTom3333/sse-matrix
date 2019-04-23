cmake_minimum_required(VERSION 3.10)

project(plop)

set(CMAKE_CXX_STANDARD 17)

add_executable(multiply2 multiply2.cpp test_macro.hpp Matrix.hpp)

if(MSVC)
    target_compile_options(multiply2 PUBLIC /arch:AVX2)
else(MSVC)
    target_compile_options(multiply2 PUBLIC -mavx2)
endif(MSVC)