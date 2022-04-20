cmake_minimum_required(VERSION 3.5)

project(Deploy  VERSION 0.1 LANGUAGES CXX)
SET(CMAKE_BUILD_TYPE "Release")

######### windeployqt F:/GitHome/Grad_Paper/DeployOn/build/QT_Release/Deploy.exe  #######


set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set( QT_DIR D:/qt/5.15.2/msvc2019_64/lib/cmake/Qt5 )
find_package(QT PATHS ${QT_DIR} NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)
file(GLOB_RECURSE QRC_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.qrc)
qt5_add_resources(QRC_FILES ${QRC_SOURCE_FILES})
SOURCE_GROUP("Resource Files" FILES ${QRC_SOURCE_FILES})

####################### Torch
set(Torch_DIR  C:/ProgramData/Anaconda3/Lib/site-packages/torch/share/cmake/Torch )
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
# if (Torch_FOUND)  ${TORCH_INCLUDE_DIRS} ${TORCH_LIBRARIES})
include_directories( ${TORCH_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/TorchScpt )


######################  opencv
set(OpenCV_DIR   D:/opencv/build/x64/vc15/lib )
find_package(OpenCV  PATHS ${OpenCV_DIR} )
include_directories(${OpenCV_INCLUDE_DIRS})

######################  darknet
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/Darknet )
link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/Darknet )


####################### OpenViNO
# set(InferenceEngine_DIR
#       C:/opt/Openvino-2021.4.2/openvino_2021.4.752/deployment_tools/inference_engine/share)
# set(TBB_DIR /opt/intel/openvino/deployment_tools/inference_engine/external/tbb/cmake/)
# find_package(TBB REQUIRED )
# set(InferenceEngine_DIR /opt/intel/openvino/deployment_tools/inference_engine/share )
# find_package(InferenceEngine REQUIRED )



######################  NCNN
set(VULKAN_SDK H:/NCNN/vulkan_GPU/SDK/)
set(Vulkan_LIBRARY H:/NCNN/vulkan_GPU/SDK/lib)
set(Vulkan_INCLUDE_DIR H:/NCNN/vulkan_GPU/SDK/include)
include_directories( ${Vulkan_INCLUDE_DIR} )
link_directories( ${Vulkan_LIBRARY} )

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# directory of GPU CPU
find_library( ncnn H:/NCNN/ncnn_cpu/build/install/lib/cmake/ncnn )
include_directories( H:/NCNN/ncnn_cpu/build/install/include )
link_directories( H:/NCNN/ncnn_cpu/build/install/lib )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Ncnn)

####################### TensorRT




set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui

        ${CMAKE_CURRENT_SOURCE_DIR}/Ncnn/ncnn_class.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TorchScpt/TorchScpt_class.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TorchScpt/TorchScpt_class.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Darknet/yolo_class.hpp
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(Deploy
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        ${QRC_FILES}
)
else()
    if(ANDROID)
        add_library(Deploy SHARED
            ${PROJECT_SOURCES}
            ${QRC_FILES}
        )
    else()
        add_executable(Deploy
            ${PROJECT_SOURCES}
            ${QRC_FILES}
        )
    endif()
endif()

target_link_libraries(Deploy PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Core
    darknet
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}

    ncnn
    ${NCNN_LIBS}
    ${VULKAN_LIBS}
)

set_target_properties(Deploy PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER xiaoxiaochenxu.top
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(Deploy)
endif()
