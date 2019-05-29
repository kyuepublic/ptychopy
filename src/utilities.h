////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Copyright © 2019, UChicago Argonne, LLC
//
//All Rights Reserved
//
//Software Name: ptychopy
//
//By: Argonne National Laboratory
//
//OPEN SOURCE LICENSE
//
//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
//following conditions are met:
//
//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
//disclaimer.
//
//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
//disclaimer in the documentation and/or other materials provided with the distribution.
//
//3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
//derived from this software without specific prior written permission.
//
//DISCLAIMER
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
//WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <execinfo.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "datatypes.h"

#if CUDA_VERSION >= 5000
	//#include <cuda_runtime.h>
	#include <helper_functions.h>
	#include <driver_types.h>
	#include <vector_types.h>
	#define _LOAD_PGMF sdkLoadPGM<float>
	#define _LOAD_PGMU sdkLoadPGM<unsigned char>
	#define _SAVE_PGMF sdkSavePGM<float>
	#define _SAVE_PGMUB sdkSavePGM<unsigned char>
	#define _SAVE_PPMUB sdkSavePPM4ub

	#define _CHECK_CMDLINE checkCmdLineFlag
	#define cutGetCmdLineArgumenti(argc,argv,arg,intValuePtr) *intValuePtr=getCmdLineArgumentInt(argc,argv,arg)
	#define cutGetCmdLineArgumentstr getCmdLineArgumentString

	#define TIMER_TYPE StopWatchInterface*
	#define _CREATE_TIMER(x) sdkCreateTimer(&x);
	#define _RESET_TIMER(x) sdkResetTimer(&x);
	#define _START_TIMER(x) sdkStartTimer(&x);
	#define _STOP_TIMER(x) sdkStopTimer(&x);
	#define _GET_TIMER_VALUE(x) sdkGetTimerValue(&x);

	#define cutilCheckMsg(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
	inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
	{
		cudaError_t err = cudaGetLastError();

		if (cudaSuccess != err)
		{
			void *array[10];
			size_t size;

			// get void*'s for all entries on the stack
			size = backtrace(array, 10);

			// print out all the frames to stderr
			fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
								file, line, errorMessage, (int)err, cudaGetErrorString(err));
			backtrace_symbols_fd(array, size, STDERR_FILENO);
			exit(-1);
		}
	}

#else
	#include <cutil.h>
	#include <cutil_inline.h>
	//#include <cutil_gl_inline.h>
	#include <stdint.h>
	#define _LOAD_PGMF cutLoadPGMf
	#define _LOAD_PGMU cutLoadPGMub
	#define _SAVE_PGMF cutSavePGMf
	#define _SAVE_PGMUB cutSavePGMub
	#define _SAVE_PPMUB cutSavePPMub
	
	#define _CHECK_CMDLINE cutCheckCmdLineFlag

	#define TIMER_TYPE unsigned int
	#define _CREATE_TIMER(x) cutCreateTimer(&x);
	#define _RESET_TIMER(x) cutResetTimer(x);
	#define _START_TIMER(x) cutStartTimer(x);
	#define _STOP_TIMER(x) cutStopTimer(x);
	#define _GET_TIMER_VALUE(x) cutGetTimerValue(x);
#endif

///////////////////////////////////////////////////////////////////////////////
// Common host and device function
///////////////////////////////////////////////////////////////////////////////
//ceil(a / b)
inline int gh_iDivUp(int a, int b)
{return ((a % b) != 0) ? (a / b + 1) : (a / b);}

//floor(a / b)
inline int gh_iDivDown(int a, int b)
{return a / b;}

//Align a to nearest higher multiple of b
inline int gh_iAlignUp(int a, int b)
{return ((a % b) != 0) ?  (a - a % b + b) : a;}

//Align a to nearest lower multiple of b
inline int gh_iAlignDown(int a, int b)
{return a - a % b;}

inline unsigned int alignFFTSize(unsigned int dataSize)
{
    int hiBit;
    unsigned int lowPOT, hiPOT;

    dataSize = gh_iAlignUp(dataSize, 16);

    for (hiBit = 31; hiBit >= 0; hiBit--)
        if (dataSize & (1U << hiBit))
            break;

    lowPOT = 1U << hiBit;

    if (lowPOT == (unsigned int)dataSize)
        return dataSize;

    hiPOT = 1U << (hiBit + 1);

    if (hiPOT <= 1024)
        return hiPOT;
    else
        return gh_iAlignUp(dataSize, 512);
}

inline unsigned int factorRadix2(unsigned int &log2N, unsigned int n)
{
    if (!n)
    {
        log2N = 0;
        return 0;
    }
    else
    {
        for (log2N = 0; n % 2 == 0; n /= 2, log2N++);
        return n;
    }
}

__host__ int getReductionThreadNum(int size);
__host__ void h_initColorTransferTexture();
__host__ void h_freeColorTransferTexture();
__host__ void h_simulateGaussian(real_t* d_gauss, real_t window,
								unsigned int x, unsigned int y, unsigned int alignedY);
__host__ real_t h_realSum(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY);
__host__ real_t h_realSum(const real_t* a, unsigned int x1, unsigned int x2, unsigned int y1, unsigned int y2, unsigned int alignedY);
__host__ float2 h_maxFloat2(float2* a, unsigned int x, unsigned int y, unsigned int alignedY);
__host__ float2 h_minFloat2(float2* a, unsigned int x, unsigned int y, unsigned int alignedY);
__host__ complex_t h_complexSum(const complex_t* a, unsigned int x1, unsigned int x2, unsigned int y1, unsigned int y2, unsigned int alignedY);
__host__ void h_subtract(const complex_t* a, const complex_t* b, complex_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY);
__host__ void h_multiply(const complex_t* a, const complex_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
								bool normalize = false, unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0);
__host__ void h_multiply(const complex_t* a, const complex_t& b, complex_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY, bool normalize = false);
__host__ void h_realToRGBA(const real_t* d_arr, float4* d_output, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor, float tf, float ts);
__host__ void h_realToGray(const real_t* d_arr, float* d_output, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor, bool outAligned=false);
__host__ void h_normalize(complex_t* a, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor);
__host__ void h_normalize(const complex_t* a, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor);
__host__ void h_normalize(real_t* d_arr, unsigned int x, unsigned int y, unsigned int alignedY);
__host__ real_t h_realMax(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY);
__host__ void h_realComplexAbs(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared=false);
__host__ void h_realComplexPhase(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared=false);
__host__ void h_realComplexReal(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared=false);
__host__ void h_realComplexImag(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared=false);
__host__ void h_adjustFFT(const complex_t* d_psi, complex_t* d_output, const real_t* d_det_mod, const real_t* d_mask, const real_t saturationValue,
								unsigned int modeNum, unsigned int x, unsigned int y, unsigned int alignedY, bool normalize=true);
__host__ real_t h_calculateER(const complex_t* d_psi, const real_t* d_det_mod, unsigned int modeNum,
								unsigned int x, unsigned int y, unsigned int alignedY);
__host__ real_t h_calculateER(const complex_t* d_GT, const complex_t* d_obj,
								unsigned int sx, unsigned int sy, unsigned int qx, unsigned int qy,
								unsigned int x1, unsigned int y1, unsigned int alignedY1,
								unsigned int x2, unsigned int y2, unsigned int alignedY2);
__host__ void h_shiftFFT(real_t* d_data, real_t* d_temp, unsigned int x, unsigned int y, unsigned int alignedY, cudaStream_t* stream=0);
__host__ void h_realRotate90(const real_t* d_data, real_t* d_out, unsigned int x, unsigned int y, unsigned int alignedY, unsigned int times, cudaStream_t* stream=0);
__host__ complex_t h_innerProduct(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
									unsigned int x, unsigned int y, unsigned int alignedY);
__host__ void h_realModalSum(const real_t* d_modes, real_t* d_output, unsigned int modesNum, unsigned int x, unsigned int y, 
								unsigned int alignedY, bool sqaureRoot=false);

__host__ int2 h_realArgMax2D(real_t* d_ncc, unsigned int x, unsigned int y, unsigned int alignedY, unsigned char dir);
__host__ void h_realComplexModulate(const complex_t* d_array1, complex_t* d_array2, int2& peak,
								unsigned int x, unsigned int y, unsigned int alignedY, unsigned char dir);

template<typename T> class Cuda3DArray;
template<typename T> class Cuda3DElement;
template<typename T> class Cuda2DArray;
#include "CudaSmartPtr.h"
#include "Singleton.h"

class GPUProperties
{
private:
	int m_gpuCount;
	unsigned int m_gpuWarpSize;
	unsigned int m_gpuMaxThread;
	size_t m_gpuAvailableMemory;

public:
	GPUProperties();
	~GPUProperties(){}

	int getDeviceCount() 							{return m_gpuCount;}
	unsigned int getGPUMaxThreads() const			{return m_gpuMaxThread;}
	unsigned int alignToWarp(unsigned int dimension){return gh_iAlignUp(dimension, m_gpuWarpSize);}
	size_t getGPUAvailableMemory()  const;
};

typedef Singleton<GPUProperties> GPUQuery;

inline bool calcGrids(unsigned int x, unsigned int alignedY, dim3& grid, dim3& block)
{
	unsigned int maxThreads = GPUQuery::getInstance()->getGPUMaxThreads();
	if (alignedY <= maxThreads)
	{
		unsigned int sliceNum = gh_iDivDown(maxThreads, alignedY);
		grid.x=gh_iDivUp(x, sliceNum); grid.y=1; grid.z=1;
		block.x=alignedY; block.y=sliceNum; block.z=1;
		return true;
	}
	else
	{
		unsigned int sliceNum = gh_iDivUp(alignedY, maxThreads);
		grid.x=x; grid.y=sliceNum; grid.z=1;
		block.x=maxThreads; block.y=1; block.z=1;
		return false;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CXUtil
{
private:
	CudaSmartPtr m_workingMemory;

public:
	CXUtil(){}
	~CXUtil(){}

	int4 applyProjectionApprox(const CudaSmartPtr& object, const Cuda3DArray<complex_t>* probeModes,
								Cuda3DArray<complex_t>* psi, float2 scanPos);
	int4 applyProjectionApprox(const CudaSmartPtr& object, Cuda3DElement<complex_t> probeMode,
								CudaSmartPtr psi, float2 scanPos);
	void gaussSmooth(real_t* d_arr, unsigned int smoothWindow, unsigned int x, unsigned int y);
	void applyModulusConstraint(const Cuda3DArray<complex_t>* d_psi, Cuda3DArray<complex_t>* d_output,
										Cuda3DElement<real_t> d_det_mod, const real_t* d_beamstopMask=0);
	real_t getModalDoubleMax(const Cuda3DArray<real_t>* d_arr);
	real_t getModalDoubleSum(const Cuda3DArray<real_t>* d_arr);
	real_t calculateER(Cuda3DArray<complex_t>* d_psi, Cuda3DElement<real_t> d_det_mod);
	real_t calculateER(const CudaSmartPtr& GT, const CudaSmartPtr& object,
						unsigned int qx, unsigned int qy, unsigned int x, unsigned int y);
};

typedef Singleton<CXUtil> PhaserUtil;

#endif /* UTILITIES_H_ */
