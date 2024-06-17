cmake_minimum_required(VERSION 3.14)
# project(Hello)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
# find_package(CUFFT REQUIRED)
find_library(CUFFT_LIB cufft HINTS /usr/local/cuda-11.4/lib64)

cuda_add_executable(cuda_test main.cu)
cuda_add_executable(cuda_fft cuda_fft.cu)

target_link_libraries(cuda_test ${CUDA_LIBRARIES} ${CUFFT_LIB})
target_link_libraries(cuda_fft ${CUDA_LIBRARIES} ${CUFFT_LIB})
