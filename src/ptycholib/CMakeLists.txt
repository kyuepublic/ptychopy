cmake_minimum_required (VERSION 3.1.0)

project (ptychoLib LANGUAGES CXX CUDA VERSION 1.0)

SET(CMAKE_DEBUG_POSTFIX D)

option (debug "Build ptychoLib with debugging on" OFF)
option (optimize "Build ptychoLib with optimization" ON)

FIND_PACKAGE(Threads REQUIRED)

############################################# CUDA #############################################
find_package (CUDA REQUIRED)
if (debug)
	if (optimize)
		SET(CMAKE_BUILD_TYPE RelWithDebInfo)
    else (optimize)
		SET(CMAKE_BUILD_TYPE Debug)
    endif (optimize)
    add_definitions(-DDEBUG)
    LIST(APPEND CUDA_NVCC_FLAGS "-G;-g;-O0")
else (debug)
	SET(CMAKE_BUILD_TYPE Release)
	LIST(APPEND CUDA_NVCC_FLAGS "-O3")
endif (debug)

#set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35")
set( CUDA_ARCHITECTURE_MINIMUM "2.0" CACHE STRING "Minimum required CUDA compute capability" )
try_run( RUN_RESULT_VAR COMPILE_RESULT_VAR
       ${CMAKE_BINARY_DIR}
       ${CMAKE_CURRENT_SOURCE_DIR}/tools/cudaComputeCapability.cpp
       CMAKE_FLAGS
       -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
       -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
       COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
       RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR
       ARGS ${CUDA_ARCHITECTURE_MINIMUM} )
       message( STATUS ${COMPILE_OUTPUT_VAR} )
if( NOT ${RUN_RESULT_VAR} EQUAL 0 OR NOT ${COMPILE_RESULT_VAR} )
     if ("${CUDA_ARCHITECTURE}" STREQUAL "")
     	message(STATUS "CUDA_ARCHITECTURE not set: CUDA COMPUTE CAPABILITY will be set to the minimum: ${CUDA_ARCHITECTURE_MINIMUM}")
     	set( CUDA_ARCHITECTURE ${CUDA_ARCHITECTURE_MINIMUM} CACHE STRING "CUDA compute capability" )
     endif()
else()
	set( CUDA_ARCHITECTURE ${RUN_OUTPUT_VAR} CACHE STRING "CUDA compute capability" )
	#set( CUDA_ARCHITECTURE "5.2" CACHE STRING "CUDA compute capability" )
endif()
message(STATUS "CUDA COMPUTE CAPABILITY: ${CUDA_ARCHITECTURE}")
string( REGEX REPLACE "([1-9])\\.([0-9])" "\\1\\2" CUDA_ARCH_STR ${CUDA_ARCHITECTURE} )
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_${CUDA_ARCH_STR}")

if(CUDA_VERSION VERSION_LESS "5.0")
	find_path(CUDA_SDK_INCLUDE_DIR
	   cutil.h
	   PATHS ${CUDA_SDK_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
	   PATH_SUFFIXES "common/inc" "samples/common/inc"
	   DOC "Location of cutil.h"
	   NO_DEFAULT_PATH
	   )
else()
	find_path(CUDA_SDK_INCLUDE_DIR
	   helper_functions.h
	   PATHS ${CUDA_SDK_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
	   PATH_SUFFIXES "common/inc" "samples/common/inc"
	   DOC "Location of helper_functions.h"
	   NO_DEFAULT_PATH
	   )
endif()
mark_as_advanced(CUDA_SDK_INCLUDE_DIR)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_INCLUDE_DIR})

find_cuda_helper_libs(curand)
find_cuda_helper_libs(cufft_static)
find_cuda_helper_libs(culibos)
find_cuda_helper_libs(cudadevrt)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cufft_static_LIBRARY} ${CUDA_culibos_LIBRARY} ${CUDA_cudadevrt_LIBRARY} ${CUDA_curand_LIBRARY})

############################################# HDF5 #############################################
set(HDF5_PATH "" CACHE PATH "prefix where HDF5 is installed")
if ("${HDF5_PATH}" STREQUAL "")
	find_package(HDF5)
	if (HDF5_FOUND)
		add_definitions(-DHAVE_HDF5)
		include_directories(${HDF5_INCLUDE_DIRS})
		set(LIBS ${LIBS} ${HDF5_LIBRARIES})
	else (HDF5_FOUND)
		MESSAGE(STATUS "HDF5_PATH not set: the package will be built without parallel hdf5 support. Hdf5 can be downloaded at http://www.hdfgroup.org/HDF5")
	endif (HDF5_FOUND)
else()
	find_library(HDF5_LIBRARIES
	   NAMES hdf5
	   PATHS "${HDF5_PATH}" "/usr/" "/usr/local/"
	   PATH_SUFFIXES "lib"
	   DOC "Location of HDF5 library"
	   NO_DEFAULT_PATH
	   )
	add_definitions(-DHAVE_HDF5)
	include_directories(${HDF5_PATH}/include)
	set(LIBS ${LIBS} ${HDF5_LIBRARIES})
endif()

############################################# SDL2 #############################################
set(SDL2_PATH "" CACHE PATH "prefix where SDL2 is installed")
if ("${SDL2_PATH}" STREQUAL "")
	MESSAGE(STATUS "SDL2_PATH not set: the package will be built without SDL2 visualization support. SDL2 can be downloaded at http://www.libsdl.org/")
else()
	FIND_PACKAGE(OpenGL REQUIRED)
	find_library(SDL2_LIBRARIES
	   NAMES SDL2
	   PATHS "${SDL2_PATH}" "/usr/" "/usr/local/"
	   PATH_SUFFIXES "lib"
	   DOC "Location of SDL2 library"
	   NO_DEFAULT_PATH
	   )

	find_package(GLEW REQUIRED)
    include_directories(${GLEW_INCLUDE_DIRS})

	add_definitions(-DHAVE_SDL2)
	include_directories(${SDL2_PATH}/include/SDL2)
	set(LIBS ${LIBS} ${SDL2_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
endif()

############################################# DIY #############################################
set(DIY_PATH "" CACHE PATH "prefix where DIY is installed")
if ("${DIY_PATH}" STREQUAL "")
	MESSAGE(STATUS "DIY_PATH not set: the package will be built without DIY2 support. Instructions to download DIY2 can be found at https://github.com/diatomic/diy2")
else()
	FIND_PACKAGE(MPI REQUIRED)

	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
	add_definitions(-DHAVE_MPI)
	include_directories(${MPI_INCLUDE_PATH})
	set(DIYLIBS ${DIYLIBS} ${MPI_LIBRARIES})

	include_directories(${DIY_PATH}/include)
endif()
################################################################################################

set(ptycholib_sources
    CartesianScanMesh.cpp
    DiffractionLoader.cpp
    GLResourceManager.cpp
    MLS.cpp
    RenderServer.cpp
    Timer.cpp
    Cuda2DArray.cpp
    Diffractions.cpp
    IPhaser.cpp
    MPIPhaser.cpp
    Sample.cpp
    utilities.cpp
    CudaFFTPlan.cpp
    ePIE.cpp
    IPtychoScanMesh.cpp
    Parameters.cpp
    SpiralScanMesh.cpp
    CudaSmartPtr.cpp
    FileManager.cpp
    ListScanMesh.cpp
    Probe.cpp
    ThreadPool.cpp)

set(ptycholib_kernels
	DiffractionsKernels.cu
	DM.cu ProbeKernels.cu
    SampleKernels.cu
	ScanMeshKernels.cu
	utilitiesKernels.cu)

set(ptycholib_headers
    CartesianScanMesh.h
    datatypes.h
    FileManager.h
    IRenderer.h
    Probe.cuh
    Singleton.h
    Cuda2DArray.hpp
    DiffractionLoader.h
    GLResourceManager.h
    IRunnable.h
    Probe.h
    SpiralScanMesh.h
    CudaFFTPlan.h
    Diffractions.cuh
    IPhaser.h
    ListScanMesh.h
    RenderServer.h
    ThreadPool.h
    CudaSmartPtr.h
    Diffractions.h
    IPhasingMethod.h
    MLS.h
    Sample.cuh
    Timer.h
    CudaSmartPtr.hpp
    DM.h
    IPtychoScanMesh.h
    MPIPhaser.h
    Sample.h
    utilities.h
    CXMath.h
    ePIE.h
    IRenderable.h
    Parameters.h
    ScanMesh.cuh)

set(ptychoLib_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

add_library(ptychoLib STATIC ${ptycholib_sources} ${ptycholib_kernels})
target_include_directories(
  ptychoLib PUBLIC ${ptychoLib_INCLUDE_DIRS}
                    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  ptychoLib PUBLIC ${CUDA_LIBS} ${LIBS} ${DIYLIBS} ${CMAKE_THREAD_LIBS_INIT}
                   ${CMAKE_DL_LIBS})
set_target_properties(ptychoLib PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (NOT SKBUILD)

  add_executable(ptycho ${CMAKE_CURRENT_SOURCE_DIR}/examples/testEPIE.cpp)
  target_include_directories(ptycho PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(ptycho ptychoLib)
  set_target_properties(ptycho PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON
                        POSITION_INDEPENDENT_CODE ON)
  install(TARGETS ptycho RUNTIME)

endif()
