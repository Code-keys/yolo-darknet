cmake_minimum_required(VERSION 2.6)

project(ToEngine)

add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

 
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)
# libtorch
set(Torch_DIR /home/nvidia/.local/lib/python3.6/site-packages/torch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
# add all head and lib
include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../Plugin
    ${CMAKE_CURRENT_SOURCE_DIR}/
)
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../Plugin
    ${CMAKE_CURRENT_SOURCE_DIR}/ 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/../Plugin/yololayer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Plugin/yololayer.cu 

    ${CMAKE_CURRENT_SOURCE_DIR}/../Plugin/mish.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Plugin/mish.cu 

    ${CMAKE_CURRENT_SOURCE_DIR}/../Plugin/calibrator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Plugin/calibrator.cpp 

)
target_link_libraries(myplugins nvinfer cudart ${OpenCV_LIBS})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS}) 

add_executable(ToEngine 
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov5.cpp
)
target_link_libraries(ToEngine nvinfer)
target_link_libraries(ToEngine cudart)
target_link_libraries(ToEngine myplugins)
target_link_libraries(ToEngine ${OpenCV_LIBS})

add_definitions(-O2 -pthread)