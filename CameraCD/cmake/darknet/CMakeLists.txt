cmake_minimum_required(VERSION 3.5)

project(dark LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)


set(OpenCV_DIR /Users/cxu/opencv-3.4.4/build_xcode)
find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)
message(find_package(OpenCV))



include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ /Users/cxu/libtorch/include ) 
link_directories(${CMAKE_CURRENT_SOURCE_DIR} /Users/cxu/Desktop/dark/qt/darknet/)
message(${CMAKE_CURRENT_SOURCE_DIR})





set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
        yolo_v2_class.hpp
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(dark
        ${PROJECT_SOURCES}
    )
else()
    if(ANDROID)
        add_library(darknet SHARED
            ${PROJECT_SOURCES}
        )
    else()
        add_executable(dark 
            ${PROJECT_SOURCES} 
        )
    endif()
endif()

target_link_libraries(dark PRIVATE Qt${QT_VERSION_MAJOR}::Widgets libdarknet.so ${OpenCV_LIBS})
