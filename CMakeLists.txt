cmake_minimum_required(VERSION 3.30)

set(PROJECT_NAME metal-test)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

set(SOURCES
    src/add.h
    src/main.cpp
)

set(METAL_SHADERS
    src/add.metal
)

set_source_files_properties(${METAL_SHADERS} PROPERTIES LANGUAGE METAL)

add_executable(${PROJECT_NAME} ${SOURCES} ${METAL_SHADERS})

# add dependencies
include(cmake/CPM.cmake)

CPMAddPackage(
    NAME     metal-cpp
    URL      http://github.com/bkaradzic/metal-cpp/archive/refs/heads/metal-cpp_26.tar.gz
    URL_HASH SHA256=b0e6d0a39618f38049e915238ea1400aaea5292167385acc52a1666fd0509aab
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${metal-cpp_SOURCE_DIR}
    src
)

target_link_libraries(${PROJECT_NAME}
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework Metal"
)
