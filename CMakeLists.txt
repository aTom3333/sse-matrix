cmake_minimum_required(VERSION 3.10)

project(SMID)

set(EXECUTABLE_NAME "matrix-multiply")

set(CMAKE_CXX_STANDARD 17)

add_executable(${EXECUTABLE_NAME} multiply2.cpp test_macro.hpp Matrix.hpp)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_HOME_DIRECTORY}/bin/${CMAKE_BUILD_TYPE})

IF(MSVC)
    target_compile_options(${EXECUTABLE_NAME} PUBLIC /arch:AVX2)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
   set(CMAKE_CXX_FLAGS_DEBUG "-g")
   set(CMAKE_CXX_FLAGS_RELEASE "-O3")
   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3")
   target_compile_options(${EXECUTABLE_NAME} PUBLIC -mavx2)
ENDIF(MSVC)