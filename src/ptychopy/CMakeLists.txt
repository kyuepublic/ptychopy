cmake_minimum_required(VERSION 3.11.0)
project(
  ptychopy
  LANGUAGES CXX CUDA
  VERSION 1.0.0)

add_definitions(${ptychoLib_INCLUDE_DIRS} ${CUDA_SDK_INCLUDE_DIR})

find_package(PythonExtensions REQUIRED)
find_package(NumPy REQUIRED)

add_library(ptychopy MODULE "ptychopy.c")
python_extension_module(ptychopy)
set_source_files_properties("ptychopy.c" PROPERTIES LANGUAGE CXX)
target_link_libraries(ptychopy ptychoLib)
target_include_directories(
  ptychopy
  PRIVATE ${NumPy_INCLUDE_DIRS} ${ptychoLib_INCLUDE_DIRS}
          ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_SDK_INCLUDE_DIR})
set_target_properties(ptychopy PROPERTIES
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON
                        POSITION_INDEPENDENT_CODE ON)
install(TARGETS ptychopy LIBRARY DESTINATION "src/ptychopy")
