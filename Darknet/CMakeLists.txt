cmake_minimum_required(VERSION 2.6)

project(yolo)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_61;code=sm_61)

include_directories(${PROJECT_SOURCE_DIR}/)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/mish.cu)

include_directories(/home/wh/program/TensorRT-7.0.0.11/include)
link_directories(/home/wh/program/TensorRT-7.0.0.11/lib )

set(OpenCV_DIR /home/wh/program/opencv-4.2.0/build)
find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)


add_executable(yolo main.cpp trt_utils.cpp yolo.cpp)
target_link_libraries(yolo nvinfer nvinfer_plugin nvparsers cudart cublas stdc++fs yololayer ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

