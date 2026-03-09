cmake_minimum_required(VERSION 3.30)

set(PROJECT_NAME metal-test)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

set(HEADERS
	src/add.h
	src/common.h
	src/global.h
	src/gpu.h
	src/invert.h
	src/invert2.h
	src/invert3.h
	src/kernel.h
	src/kernel_cpu.h
	src/kernel_metal.h
	src/meta.h
	src/mtl.h
	src/reflection.h
)

set(SOURCES
	src/gpu.cpp
	src/main.cpp
	src/mtl.cpp
	src/reflection.cpp
	src/test1.cpp
	src/test2.cpp
	src/test3.cpp
	src/test4.cpp
	src/test5.cpp
)

set(SHADERS
	src/add.metal
	src/invert.metal
	src/invert2.metal
	src/invert3.metal
)

set_source_files_properties(${SHADERS} PROPERTIES LANGUAGE METAL)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES} ${SHADERS})

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
