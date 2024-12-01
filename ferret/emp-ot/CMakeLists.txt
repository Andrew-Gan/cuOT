cmake_minimum_required (VERSION 3.0)
project (emp-ot LANGUAGES C CXX CUDA)
set(NAME "emp-ot")

find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/emp-base.cmake)

find_package(emp-tool REQUIRED)
include_directories(${EMP-TOOL_INCLUDE_DIRS})

# Installation
install(FILES cmake/emp-ot-config.cmake DESTINATION cmake/)
install(DIRECTORY emp-ot DESTINATION include/)

add_library(dev_layer emp-ot/ferret/dev_layer.cu ../../gpu/gpu.o)
target_include_directories(dev_layer PUBLIC "../../gpu" "/usr/local/cuda/include")
target_link_directories(dev_layer PUBLIC "/usr/local/cuda/lib64")
target_link_libraries(dev_layer "cuda" "cudart" "cusparse")

ENABLE_TESTING()
ADD_SUBDIRECTORY(test)