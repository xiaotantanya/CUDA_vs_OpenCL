cmake_minimum_required(VERSION 3.20)

project(gpu_test
        DESCRIPTION "test different gpu framework"
        VERSION 0.0.1
        LANGUAGES C CXX)

add_subdirectory(cuda_test)
add_subdirectory(opencl_test)
add_subdirectory(cpu_test)