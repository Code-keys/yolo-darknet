CMAKE_MINIMUM_REQUIRED(VERSION 3.9) #设定版本

set(target_name QTVSCD)
set(OpenCV_DIR C:/CV3.4.0/opencv/build/x64/vc14/lib)
set(TRT_DIR C:/CV3.4.0/opencv/build/x64/vc14/lib)
set(TORCH_DIR C:/CV3.4.0/opencv/build/x64/vc14/lib)
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
set(Qt5_DIR C:/CV3.4.0/opencv/build/x64/vc14/lib)
set(CMAKE_PREFIX_PATH /usr/lib/x86_64-linux-gnu)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


#设定工程名 
PROJECT(target_name) 
# SET(CMAKE__CXx_COMPILER" g++") #设定编译器
# set(CMAKE_CXX_STANDARD 14)

# optimizer function modules
set(CUDA_VERSION 10.2)
#set(CMAKE_CUDA_COMPILER ${CUDA_DIR}/bin/nvcc.exe)
FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)
# enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -gencode arch=compute_75,code=sm_75")
# set_target_properties(target_name PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
MESSAGE(" CUDA SET DONE !")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOQRC ON)
#aux_source_directory(${PROJECT_SOURCE_DIR} DIR_SRCS)
set(QRC_SOURCE_FILES ${PROJECT_SOURCE_DIR}/res/qtvscd.qrc)
set(UI_SOURCE_FILES ${PROJECT_SOURCE_DIR}/res/qtvscd.ui)
set(MOC_SOURCE_FILES ${PROJECT_SOURCE_DIR}/res/qtvscd.h)
FIND_PACKAGE(Qt5 REQUIRED  Core Gui Widgets)
qt5_add_resources(RCC ${QRC_SOURCE_FILES})
qt5_wrap_cpp(MSC  ${MOC_SOURCE_FILES})
qt5_wrap_ui(UIC ${UI_SOURCE_FILES})
MESSAGE(" QT SET DONE !")

# FIND_PACKAGE(OpenCV 3 REQUIRED)
MESSAGE(" OPENCV SET DONE !")

#设置头文件目录
set(CMAKE_INCLUDE_CURRENT_DIR ON)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src
                    ${TRT_DIR}/include
                    ${OpenCV_DIR}/include
                    ${CUDA_DIR}/include
                    $) 
#设置依赖库文件目录
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/src
                 ${TRT_DIR}/lib
                 ${OpenCV_DIR}/lib
                 ${CUDA_DIR}/lib         

                 ${TRT_DIR}/bin
                 ${OpenCV_DIR}/bin
                 ${CUDA_DIR}/bin
                ) 

CUDA_ADD_LIBRARY(kernel SHARED
                 src/kernel.cu
                 kernel.h)
TARGET_LINK_LIBRARIES(kernel ${CUDA_LIBRARIES})
MESSAGE(" CUDA KERNEL BUILT !")


ADD_EXECUTABLE(
    ${target_name} 
    main.cpp src/qtvscd.cpp
    kernel.h qtvscd.h
    ${UIC} ${MSC} ${RCC} )
target_link_libraries(  ${target_name} 
                        Qt5::Core 
                        Qt5::Gui 
                        Qt5::Widgets   
                        ${OpenCV_LIBS}
                        ${CUDA_LIBS}
                        kernel)
MESSAGE(" ALL DONE !")