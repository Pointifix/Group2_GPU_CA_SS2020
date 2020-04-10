cmake_minimum_required(VERSION 3.8)
project(Group2_GPU_CA_SS2020 LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

## Add executables/subdirectories
add_executable(Group2_GPU_CA_SS2020
        src/main.cu src/graph.cpp
        src/graph.h src/graph_generator.h src/graph_io.h
        src/sssp.h src/sssp_standard.cu src/sssp_standard.h
        src/sssp_sequential.cpp src/sssp_sequential.h src/sssp_pinned_memory.cu
        src/sssp_pinned_memory.h src/sssp_zero_copy_memory.cu src/sssp_zero_copy_memory.h
        src/sssp_thrust.cu src/sssp_thrust.h)
#set_property(TARGET Group2_GPU_CA_SS2020 PROPERTY CUDA_STANDARD 11)

## Don't know what this is for... it comes with a new CLion CUDA project
# Also: must be done after adding the executables/subdirectories
set_target_properties(
        Group2_GPU_CA_SS2020
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

