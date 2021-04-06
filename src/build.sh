#!/bin/bash

HDF5_PATH=
SDL2_PATH=
DIY_PATH=


rm -rf build
mkdir build
cd build

cmakeVars="-DHDF5_PATH="$HDF5_PATH
cmakeVars=$cmakeVars" -DSDL2_PATH="$SDL2_PATH
cmakeVars=$cmakeVars" -DDIY_PATH="$DIY_PATH

if [ "$1" = "debug" ]; then
	cmakeVars="-Ddebug=ON -Doptimize=OFF $cmakeVars"
fi

cmake  $cmakeVars -G "Unix Makefiles" ..

make
cd ..
rm -rf build
