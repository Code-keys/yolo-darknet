# include(CMakeLists_MacOS.cmake) 
#  CXX_FLAGS中的-D_GLIBCXX_USE_CXX11_ABI=1引起的undefined问题


cmake_minimum_required(VERSION 3.6)

project( Deploy )

#
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_INCLUDE_CURRENT_DIR ON) 

add_definitions(-DUNICODE -D_UNICODE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O3 -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -D_GLIBCXX_USE_CXX11_ABI=1")

####################### Torch  
set(Torch_DIR    /home/nvidia/.local/lib/python3.6/site-packages/torch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
# if (Torch_FOUND) include path: ${TORCH_INCLUDE_DIRS} ${TORCH_LIBRARIES})
include_directories( ${TORCH_INCLUDE_DIRS}   ${CMAKE_CURRENT_SOURCE_DIR}/TorchScpt )


######################  QT
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
 
# set(QT_DIR  C:/ProgramData/Anaconda3/pkgs/qt-5.9.7-vc14h73c81de_0/Library/lib/cmake/Qt5 )
# set(QT_DIR  E:/Chenxu/QT5/5.7/mingw53_32/lib/cmake/Qt5)
find_package(QT NAMES Qt5 COMPONENTS Widgets Core REQUIRED )
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets Core REQUIRED )
# include_directories( C:/ProgramData/Anaconda3/pkgs/qt-5.9.7-vc14h73c81de_0/Library/include/qt )
file(GLOB_RECURSE QRC_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.qrc)
qt5_add_resources(QRC_FILES ${QRC_SOURCE_FILES})
SOURCE_GROUP("Resource Files" FILES ${QRC_SOURCE_FILES})


####################### OpenViNO 
# set(InferenceEngine_DIR 
#       C:/opt/Openvino-2021.4.2/openvino_2021.4.752/deployment_tools/inference_engine/share)
# set(TBB_DIR /opt/intel/openvino/deployment_tools/inference_engine/external/tbb/cmake/)
# find_package(TBB REQUIRED )
# set(InferenceEngine_DIR /opt/intel/openvino/deployment_tools/inference_engine/share )
# find_package(InferenceEngine REQUIRED )
 
####################### TensorRT  
# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64) 
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)
if (CUDA_FOUND)
message( "-- myplugins configing ....   ")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/TensorRt
    ${CMAKE_CURRENT_SOURCE_DIR}/TensorRt/Plugin )

cuda_add_library(myplugins SHARED
    ${PROJECT_SOURCE_DIR}/TensorRt/Plugin/mish.h
    ${PROJECT_SOURCE_DIR}/TensorRt/Plugin/mish.cu 
    ${PROJECT_SOURCE_DIR}/TensorRt/Plugin/yololayer.h 
    ${PROJECT_SOURCE_DIR}/TensorRt/Plugin/yololayer.cu
) 
target_link_libraries(myplugins nvinfer cudart)
SET_TARGET_PROPERTIES(myplugins PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../TensorRt )
message( "-- myplugins configed done ! ")
endif()

######################  NCNN
# NCNN
set( Vulkan_LIBRARY /home/nvidia/CX/vulkan_SDK/x86_64/lib )
set( Vulkan_INCLUDE_DIR /home/nvidia/CX/vulkan_SDK/x86_64/include)  
include_directories(${Vulkan_INCLUDE_DIR})
link_directories(${Vulkan_LIBRARY})
 
set( ncnn_DIR /home/nvidia/CX//ncnn-20211208/lib/cmake/ncnn)
find_package(ncnn PATHS ${ncnn_DIR} REQUIRED)
include_directories(/home/nvidia/CX//ncnn-20211208/include)
link_directories(/home/nvidia/CX//ncnn-20211208/lib)

####################### Darknet 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Darknet)

######################  opencv
find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)
message(find_package(OpenCV))

####################### MainWindows #######################   

message( "" ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.ui
        mainwindow.h

        TensorRt/infeRt_class.h
        Darknet/yolo_class.hpp
        TorchScpt/TorchScpt_class.cpp
) 
add_executable(Deploy 
        ${PROJECT_SOURCES} 
        ${QRC_FILES}
) 
SET_TARGET_PROPERTIES(myplugins PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

target_link_libraries(Deploy ${OpenCV_LIBS})

target_link_libraries(Deploy  Qt${QT_VERSION_MAJOR}::Widgets  Qt${QT_VERSION_MAJOR}::Core )

target_link_libraries(Deploy ${CMAKE_CURRENT_SOURCE_DIR}/Darknet/libdarknet.so   ) #     lib/darknet.dll  # libdarknet.so

target_link_libraries(Deploy ${TORCH_LIBRARIES}) 

target_link_libraries(Deploy  nvinfer cudart  myplugins ) 

add_definitions(-O2 -pthread)