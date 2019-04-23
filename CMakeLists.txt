cmake_minimum_required(VERSION 3.10)

project(plop)

add_executable(naive naive.cpp)
add_executable(simd copie-simd.cpp)
add_executable(multiply multiply.cpp test_macro.hpp matrix.hpp)
target_compile_options(multiply PUBLIC -mavx2)