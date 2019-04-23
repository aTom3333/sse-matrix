cmake_minimum_required(VERSION 3.10)

project(plop)

set(CMAKE_CXX_STANDARD 17)

add_executable(multiply2 multiply2.cpp test_macro.hpp Matrix.hpp)

IF(MSVC)
   ADD_DEFINITIONS(/arch:AVX2)
ENDIF(MSVC)