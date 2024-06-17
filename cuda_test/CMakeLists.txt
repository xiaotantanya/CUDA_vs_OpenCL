cmake_minimum_required(VERSION 3.14)
project(Hello)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
# find_package(CUFFT REQUIRED)

cuda_add_executable(Hello main.cu)

target_link_libraries(Hello ${CUDA_LIBRARIES} cufft)
