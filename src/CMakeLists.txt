cmake_minimum_required(VERSION 3.5)
project(variance_map)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O2 -g -fopenmp -mtune=native -march=native")

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV opencv REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(INCLUDE_DIR
        ${OPENCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/CppProgressBar/include
        )
include_directories(${INCLUDE_DIR})
link_directories(${OPENCV_LIBRARY_DIRS})

add_executable(variance_map main.cpp VarianceComputer.cpp StatisticUtility.h)
target_link_libraries(variance_map ${OPENCV_LIBRARIES})