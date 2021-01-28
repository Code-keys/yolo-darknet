cmake_minimum_required(VERSION 2.6)

project(yolo) # 1
set(OpenCV_DIR "C:/OpenCV/opencv/build")  #2
set(TRT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-7.0.0.11")  #3

add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode; arch=compute_75;code=sm_75)
####
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
####
include_directories(${TRT_DIR}/include)

# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# setup opencv
find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${TRT_DIR}/lib)

add_executable(yolo ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/yolo.cpp ${PROJECT_SOURCE_DIR}/trt_utils.cpp ${PROJECT_SOURCE_DIR}/mish.cu ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/yololayer.h ${PROJECT_SOURCE_DIR}/mish.h ${PROJECT_SOURCE_DIR}/logging.h ${PROJECT_SOURCE_DIR}/trt_utils.h ${PROJECT_SOURCE_DIR}/Utils.h)

target_link_libraries(yolo  "nvinfer" "nvinfer_plugin")   #5
target_link_libraries(yolo ${OpenCV_LIBS})          #6
target_link_libraries(yolo ${CUDA_LIBRARIES})   #7
target_link_libraries(yolo Threads::Threads)       #8