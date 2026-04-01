cmake_minimum_required(VERSION 3.20)
project(flashmlx_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find MLX
set(MLX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/lib/python3.14/site-packages/mlx/share/cmake/MLX")
find_package(MLX REQUIRED)

# Find pybind11
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

# Build the extension module
pybind11_add_module(_flashmlx_engine
    src/engine.cpp
    src/model.cpp
    src/kv_pool.cpp
)
target_include_directories(_flashmlx_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MLX_INCLUDE_DIRS}
)
target_link_libraries(_flashmlx_engine PRIVATE mlx)
target_compile_options(_flashmlx_engine PRIVATE ${MLX_CXX_FLAGS})

# Install to python package directory
install(TARGETS _flashmlx_engine DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python)
