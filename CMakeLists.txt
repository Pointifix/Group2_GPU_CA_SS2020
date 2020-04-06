cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14)

set(CUDA_COMPILER_PATH "/usr/local/cuda-9.0/bin/nvcc" CACHE STRING "CUDA compiler path")
set(CMAKE_CUDA_COMPILER ${CUDA_COMPILER_PATH})

project(test)

include(FindCUDA)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_22,code=sm_22)
include_directories("${CUDA_INCLUDE_DIRS}")

add_subdirectory(src)