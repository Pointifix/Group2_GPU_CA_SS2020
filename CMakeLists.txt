cmake_minimum_required(VERSION 3.8)
project(Group2_GPU_CA_SS2020 CUDA)

set(CMAKE_CUDA_STANDARD 11)

## Add executables/subdirectories
add_executable(Group2_GPU_CA_SS2020 src/main.cu)
#set_property(TARGET Group2_GPU_CA_SS2020 PROPERTY CUDA_STANDARD 11)

## Don't know what this is for... it comes with a new CLion CUDA project
# Also: must be done after adding the executables/subdirectories
set_target_properties(
        Group2_GPU_CA_SS2020
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

