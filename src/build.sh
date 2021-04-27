#!/bin/bash

HDF5_PATH=/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/dependLib/hdf5-1.8.13
SDL2_PATH=
#DIY_PATH=/local/kyue/anlproject/ptography/pushPtychopy/ptychopy/dependLib/diy2
#DIY_PATH=

#rm -rf build
#mkdir build
#cd build

cmakeVars="-DHDF5_PATH="$HDF5_PATH
cmakeVars=$cmakeVars" -DSDL2_PATH="$SDL2_PATH
cmakeVars=$cmakeVars" -DDIY_PATH="$DIY_PATH

if [ "$1" = "debug" ]; then
	cmakeVars="-Ddebug=ON -Doptimize=OFF $cmakeVars"
fi

/local/kyue/program/cmake/cmake/bin/cmake $cmakeVars -G "Unix Makefiles" .
#/local/kyue/program/cmake/cmake/bin/cmake -DCMAKE_C_COMPILER=/home/beams/USER2IDD/userlib/gcc-6.3.0/bin/gcc $cmakeVars -G "Unix Makefiles" .

make
#cd ..
#rm -rf build
