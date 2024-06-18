cmake_minimum_required(VERSION 3.14)
# project(Hello)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
# find_package(CUFFT REQUIRED)
find_library(CUFFT_LIB cufft HINTS /usr/local/cuda-11.4/lib64)

cuda_add_executable(cuda_test cuda_simple_operation.cu)
cuda_add_executable(cuda_fft cuda_fft.cu)
cuda_add_executable(cuda_matmul cuda_matmul.cu)
# cuda_add_executable(cuda_matmul_shared cuda_matmul_shared.cu)

target_link_libraries(cuda_test ${CUDA_LIBRARIES} ${CUFFT_LIB})
target_link_libraries(cuda_fft ${CUDA_LIBRARIES} ${CUFFT_LIB})
target_link_libraries(cuda_matmul ${CUDA_LIBRARIES} ${CUFFT_LIB})
# target_link_libraries(cuda_matmul_shared ${CUDA_LIBRARIES} ${CUFFT_LIB})
