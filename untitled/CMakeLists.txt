cmake_minimum_required(VERSION 3.27)
project(untitled)

set(CMAKE_CXX_STANDARD 17)

add_executable(lab1 main.cpp)
target_compile_options(lab1 PRIVATE -march=native)

