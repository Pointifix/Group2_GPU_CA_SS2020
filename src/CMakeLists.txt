cmake_minimum_required(VERSION 3.15)

# set the project name and version
project(Group2_GPU_CA_SS2020)

set(CMAKE_CXX_STANDARD 14)

# add the executable
#add_executable(Group2_GPU_CA_SS2020 test.cpp)
add_executable(Group2_GPU_CA_SS2020 kernel.cu)