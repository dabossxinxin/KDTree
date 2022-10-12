cmake_minimum_required(VERSION 3.8)
project(KDTree)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

include(FindOpenCV)

include_directories(${OpenCV4_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(KD_MAIN src/main.cpp)
add_executable(TEST_KDTREE src/test_kdtree.cpp)
target_link_libraries(KD_MAIN  debug ${OpenCV4_DEBUG_LIB} optimized ${OpenCV4_RELEASE_LIB})
target_link_libraries(TEST_KDTREE  debug ${OpenCV4_DEBUG_LIB} optimized ${OpenCV4_RELEASE_LIB})
