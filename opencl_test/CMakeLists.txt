cmake_minimum_required(VERSION 3.1) # 3.1 << C_STANDARD 11

# project(HelloOpenCL LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

add_executable(opencl_test main.cpp)

target_link_libraries(opencl_test PRIVATE OpenCL::OpenCL)

set_target_properties(opencl_test PROPERTIES C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

target_compile_definitions(opencl_test PRIVATE CL_TARGET_OPENCL_VERSION=100)