cmake_minimum_required(VERSION 3.10)

project(plop)

add_executable(bin/naive naive.cpp)
add_executable(bin/simd copie-simd.cpp)
add_executable(multiply multiply.cpp)
target_compile_options(multiply PUBLIC -mavx2)