#!/bin/bash

HDF5_PATH="../dependLib/hdf5-1.8.13/hdf5/"
#HDF5_PATH="/local/kyue/hdf5/"
SDL2_PATH=
DIY_PATH=

rm -rf build
mkdir build
cd build

cmakeVars="-DHDF5_PATH="$HDF5_PATH
cmakeVars=$cmakeVars" -DSDL2_PATH="$SDL2_PATH
cmakeVars=$cmakeVars" -DDIY_PATH="$DIY_PATH
#cmakeVars=$cmakeVars" -DOPENCV_PATH="$OPENCV_PATH

if [ "$1" = "debug" ]; then
	cmakeVars="-Ddebug=ON -Doptimize=OFF $cmakeVars"
fi

/local/kyue/program/cmake/cmake/bin/cmake  $cmakeVars -G "Unix Makefiles" ..

make -j install
cd ..

rm -rf build
