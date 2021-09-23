/*
 * The MIT License
 *
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  \file   cudaComputeCapability.cpp
 *  \date   Sep 23, 2014
 *  \author "James C. Sutherland"
 *
 *  This utility prints out the compute capability of the GPU on a system.
 *  It can optionally take an argument specifying the minimum required compute
 *  capability and then will not consider any devices on the system that are
 *  below that minimum.
 *
 *  It is meant to be used within the SpatialOps build system.
 */

#include <stdexcept>
#include <iostream>
#include <string>
#include "stdio.h"
#include <cuda_runtime.h>

int main( int argc, char* argv[] )
{
  double minComputeCapability = 3.0;

  if( argc == 2 ){
    sscanf( argv[1],"%lf", &minComputeCapability );
//    std::cout << "MINIMUM COMPUTE CAPABILITY: " << argv[1] << " -> " << minComputeCapability << std::endl;
  }

  const int minMajor = int(minComputeCapability);
  const int minMinor = int(minComputeCapability*10)%10;

  int tottalDeviceCount=0;
  int gpuDeviceCount = 0;
  int major = 999, minor = 999;

  if( cudaGetDeviceCount(&tottalDeviceCount) != cudaSuccess ){
    std::cout << "Couldn't get device count: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return -1;
  }

//  std::cout << "found " << deviceCount << " devices\n";

  /* machines with no GPUs can still report one emulation device */
  for( int device = 0; device < tottalDeviceCount; ++device ){

    cudaDeviceProp props;
    cudaGetDeviceProperties( &props, device );

//    std::cout << "On gpu " << device << " compute capability is " << props.major << "." << props.minor << std::endl;

    if( props.major == 9999 ) continue; // 9999 means emulation only

    if( props.major < minMajor ) continue;
    if( props.major == minMajor && props.minor < minMinor ) continue;

    ++gpuDeviceCount;

    /*  get minimum compute capability of all devices */
    if( major > props.major ){
      major = props.major;
      minor = props.minor;
    }
    else if( minor > props.minor ){
      minor = props.minor;
    }
  }

  if( gpuDeviceCount > 0 ){
    std::cout << major << "." << minor;
    return 0; /* success */
  }

  std::cout << "No valid GPU devices found\n";

  return -1;
}
