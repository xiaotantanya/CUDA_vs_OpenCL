cmake_minimum_required(VERSION 3.1) # 3.1 << C_STANDARD 11

# project(HelloOpenCL LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

add_executable(opencl_test opencl_simple_operation.cpp)
add_executable(opencl_matmul opencl_matmul.cpp)
add_executable(opencl_fft opencl_fft.cpp)

target_link_libraries(opencl_test PRIVATE OpenCL::OpenCL)

set_target_properties(opencl_test PROPERTIES C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

target_compile_definitions(opencl_test PRIVATE CL_TARGET_OPENCL_VERSION=100)

target_link_libraries(opencl_matmul PRIVATE OpenCL::OpenCL)

set_target_properties(opencl_matmul PROPERTIES C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

target_compile_definitions(opencl_matmul PRIVATE CL_TARGET_OPENCL_VERSION=100)

target_link_libraries(opencl_fft PRIVATE OpenCL::OpenCL)

set_target_properties(opencl_fft PROPERTIES C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

target_compile_definitions(opencl_fft PRIVATE CL_TARGET_OPENCL_VERSION=100)