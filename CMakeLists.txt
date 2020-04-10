cmake_minimum_required(VERSION 3.8)
project(Group2_GPU_CA_SS2020 LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

set(ALL_SRCS # NO MAIN ALLOWED HERE!
    # graph
        src/graph.h src/graph.cpp
        src/graph_generator.h
        src/graph_io.h
    # sssp
        src/sssp.h
        src/sssp_standard.h src/sssp_standard.cu
        src/sssp_sequential.h src/sssp_sequential.cpp
        src/sssp_pinned_memory.h src/sssp_pinned_memory.cu
        src/sssp_zero_copy_memory.h src/sssp_zero_copy_memory.cu
        src/sssp_thrust.h src/sssp_thrust.cu)

add_library(SrcLib ${ALL_SRCS})

add_executable(Group2_GPU_CA_SS2020 src/main.cu)
target_link_libraries(Group2_GPU_CA_SS2020 ScrLib)

## Don't know what this is for... it comes with a new CLion CUDA project
# Also: must be done after adding the executables/subdirectories
set_target_properties(
        Group2_GPU_CA_SS2020
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug mode")
    #add_definitions("-DDEBUG")

    message("Adding Test executable")
    add_executable(Test
            test/tests.cpp
            test/test_graph.hpp)
    target_link_libraries(Test SrcLib)
    set_target_properties(
            Test
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release mode")
endif()
