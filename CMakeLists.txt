cmake_minimum_required(VERSION 3.10)

project(plop)

set(CMAKE_CXX_STANDARD 17)

add_executable(naive naive.cpp)
add_executable(simd copie-simd.cpp)
add_executable(multiply multiply.cpp)
add_executable(multiply2 multiply2.cpp test_macro.hpp matrix.hpp)
target_compile_options(multiply2 PUBLIC -mavx2)