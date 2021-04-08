# Install script for directory: /local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/Cuda2DArray.hpp;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/CudaSmartPtr.hpp;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/CXMath.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/CartesianScanMesh.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/CudaFFTPlan.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/CudaSmartPtr.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/DM.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/DiffractionLoader.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/Diffractions.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/FileManager.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/GLResourceManager.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/IPhaser.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/IPhasingMethod.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/IPtychoScanMesh.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/IRenderable.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/IRenderer.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/IRunnable.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/ListScanMesh.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/MLS.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/MPIPhaser.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/Parameters.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/Probe.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/RenderServer.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/Sample.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/Singleton.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/SpiralScanMesh.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/ThreadPool.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/Timer.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/datatypes.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/ePIE.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/utilities.h;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/Diffractions.cuh;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/Probe.cuh;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/Sample.cuh;/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include/ScanMesh.cuh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/include" TYPE FILE FILES
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/Cuda2DArray.hpp"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/CudaSmartPtr.hpp"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/CXMath.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/CartesianScanMesh.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/CudaFFTPlan.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/CudaSmartPtr.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/DM.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/DiffractionLoader.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/Diffractions.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/FileManager.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/GLResourceManager.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/IPhaser.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/IPhasingMethod.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/IPtychoScanMesh.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/IRenderable.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/IRenderer.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/IRunnable.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ListScanMesh.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/MLS.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/MPIPhaser.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/Parameters.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/Probe.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/RenderServer.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/Sample.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/Singleton.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/SpiralScanMesh.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ThreadPool.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/Timer.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/datatypes.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ePIE.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/utilities.h"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/Diffractions.cuh"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/Probe.cuh"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/Sample.cuh"
    "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ScanMesh.cuh"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/lib/libptychoLib.a")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/lib" TYPE STATIC_LIBRARY FILES "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/libptychoLib.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ptycho" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ptycho")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ptycho"
         RPATH "/local/kyue/program/cuda/lib64:/local/kyue/program/anaconda/lib")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ptycho")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src" TYPE EXECUTABLE FILES "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ptycho")
  if(EXISTS "$ENV{DESTDIR}/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ptycho" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ptycho")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ptycho"
         OLD_RPATH "/local/kyue/program/cuda/lib64:/local/kyue/program/anaconda/lib:"
         NEW_RPATH "/local/kyue/program/cuda/lib64:/local/kyue/program/anaconda/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/ptycho")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/src/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
