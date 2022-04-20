cmake_minimum_required(VERSION 2.6)

project( demoplugins )

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(demoplugins SHARED   

    ${CMAKE_CURRENT_SOURCE_DIR}/demo_plugin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/demo_plugin.cu 

)
target_link_libraries(demoplugins nvinfer cudart)

add_definitions(-O2 -pthread)