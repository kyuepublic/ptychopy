#!/bin/bash

HDF5_PATH=/local/kyue/program/hdf5-1.8.13
SDL2_PATH=
#DIY_PATH=/local/kyue/program/diy2
#DIY_PATH=


cmakeVars="-DHDF5_PATH="$HDF5_PATH
cmakeVars=$cmakeVars" -DSDL2_PATH="$SDL2_PATH
cmakeVars=$cmakeVars" -DDIY_PATH="$DIY_PATH

if [ "$1" = "debug" ]; then
	cmakeVars="-Ddebug=ON -Doptimize=OFF $cmakeVars"
fi

/local/kyue/program/cmake/cmake/bin/cmake $cmakeVars -G "Unix Makefiles" .
#/local/kyue/program/cmake/cmake/bin/cmake -DCMAKE_C_COMPILER=/local/kyue/program/gcc63/bin/gcc $cmakeVars -G "Unix Makefiles" .

make
rm cmake_install.cmake
rm -fr CMakeFiles
rm CMakeCache.txt
