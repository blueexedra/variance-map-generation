cmake_minimum_required(VERSION 3.9)
project(variance_map)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV opencv4 REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(INCLUDE_DIR
        ${OPENCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/CppProgressBar/include
        )

add_executable(${CMAKE_PROJECT_NAME} main.cpp VarianceComputer.cpp)

target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC -O3 -fopenmp -march=native)
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${INCLUDE_DIR})
target_link_libraries(variance_map PUBLIC ${OPENCV_LIBRARIES} OpenMP::OpenMP_CXX)