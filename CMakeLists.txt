cmake_minimum_required(VERSION 4.0)
project(nnlib)

set(CMAKE_CXX_STANDARD 20)
add_subdirectory(matplotplusplus)

add_executable(nnlib main.cpp
        nnlib.h)
target_link_libraries(nnlib PUBLIC matplot)