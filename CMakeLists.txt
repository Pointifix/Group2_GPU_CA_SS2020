cmake_minimum_required(VERSION 3.8)
project(Group2_GPU_CA_SS2020 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

include_directories(inc)

set(ALL_SRCS # NO MAIN ALLOWED HERE!
    # graph
        src/graph.cu
    # alg
        src/alg.cu
    # sssp
        src/sssp_standard.cu
        src/sssp_sequential.cu
        src/sssp_pinned_memory.cu
        src/sssp_zero_copy_memory.cu
        src/sssp_thrust.cu)

add_executable(Group2_GPU_CA_SS2020
        ${ALL_SRCS}
        src/main.cu)

## Don't know what this is for... it comes with a new CLion CUDA project
# Also: must be done after adding the executables/subdirectories
set_target_properties(
        Group2_GPU_CA_SS2020
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug mode")
    add_definitions("-DDEBUG")

    message("Adding Test executable")
    add_executable(Test
            ${ALL_SRCS}
            test/tests.cpp
            test/test_graph.hpp
            test/test_alg.cuh)
    set_target_properties(
            Test
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release mode")
endif()
