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

#ifndef DIFFRACTIONSERNELS_CU_
#define DIFFRACTIONSERNELS_CU_

#include "Diffractions.cuh"
#include "utilities.h"
#include "reductions.cu"

//__device__ double atomicAdd(double* address, double val)
//{
//    unsigned long long int* address_as_ull =
//                              (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed,
//                        __double_as_longlong(val +
//                               __longlong_as_double(assumed)));
//    } while (assumed != old);
//    return __longlong_as_double(old);
//}

template<unsigned int threadNum>
__global__ void d_calculateSTXM(const real_t* d_intensities, real_t* d_output,
								unsigned int scansAlignedY, unsigned int intensitiesX, unsigned int intensitiesY)
{
	/*extern __shared__ real_t s_addends[];
	unsigned int outputIndex = (blockIdx.x * scansAlignedY) + blockIdx.y;
	unsigned int intensityIndex = (blockIdx.x * gridDim.y) + blockIdx.y;
	intensityIndex = (((intensityIndex * intensitiesX ) + threadIdx.y) * blockDim.x) + threadIdx.x;

	s_addends[threadIdx.x] = 0;

	if(threadIdx.x<intensitiesY)
	{
		for(unsigned int i=blockDim.y; i>0; --i)
		{
			atomicAdd(&s_addends[threadIdx.x], d_intensities[intensityIndex]);
			intensityIndex += threadNum;
		}
	}

	if(threadIdx.y == 0)
	{
		reduceToSum<real_t,threadNum>(s_addends,threadIdx.x);

		if(threadIdx.x == 0)
			d_output[outputIndex] = s_addends[0];
	}*/
}

__global__ void d_symmetrizeDiffraction(const real_t* d_intensities,real_t* d_output, 
										unsigned int dataOffsetX, unsigned int dataOffsetY, unsigned int dataAlignedY, 
										unsigned int outOffsetX, unsigned int outOffsetY, unsigned int outAlignedY)
{
	unsigned int diffIndex	= ((dataOffsetX+blockIdx.x) * dataAlignedY) + threadIdx.x + dataOffsetY;
	unsigned int outputIndex= ((outOffsetX+blockIdx.x) * outAlignedY) + threadIdx.x + outOffsetY;
	d_output[outputIndex] = d_intensities[diffIndex];
}

__global__ void d_preprocessDiffractions(real_t* d_intensities, bool squareRoot, real_t threshold, unsigned int intensitiesX, unsigned int intensitiesY)
{
	unsigned int intensityIndex = (blockIdx.x * intensitiesX) + ((blockIdx.y*blockDim.y) + threadIdx.y);
	intensityIndex = (intensityIndex * blockDim.x) + threadIdx.x;
	real_t value = d_intensities[intensityIndex];
	if(!isfinite(value))
		value = 0;
	value = (value<threshold)? 0 : value-threshold;
	d_intensities[intensityIndex] = squareRoot? sqrt_real_t(value) : value;
}

//
__host__ void h_calculateSTXM(const real_t* d_intensities, real_t* d_output,
								unsigned int scansX, unsigned int scansY, unsigned int scansAlignedY,
								unsigned int intensitiesX, unsigned int intensitiesY, unsigned int intensitiesAlignedY)
{
	dim3 grid(scansX, scansY, 1);
	dim3 block(intensitiesAlignedY, gh_iDivUp(GPUQuery::getInstance()->getGPUMaxThreads(), intensitiesX), 1);
	size_t shared_mem_size = (block.x <= 32) ? 2* block.x * sizeof(real_t) : block.x *  sizeof(real_t);

	switch (block.x)
	{
	case   8:	d_calculateSTXM<   8><<<grid, block, shared_mem_size>>>(d_intensities, d_output, scansAlignedY, intensitiesX, intensitiesY);
	break;
	case  16:	d_calculateSTXM<  16><<<grid, block, shared_mem_size>>>(d_intensities, d_output, scansAlignedY, intensitiesX, intensitiesY);
	break;
	case  32:	d_calculateSTXM<  32><<<grid, block, shared_mem_size>>>(d_intensities, d_output, scansAlignedY, intensitiesX, intensitiesY);
	break;
	case  64:	d_calculateSTXM<  64><<<grid, block, shared_mem_size>>>(d_intensities, d_output, scansAlignedY, intensitiesX, intensitiesY);
	break;
	case 128:	d_calculateSTXM< 128><<<grid, block, shared_mem_size>>>(d_intensities, d_output, scansAlignedY, intensitiesX, intensitiesY);
	break;
	case 256:	d_calculateSTXM< 256><<<grid, block, shared_mem_size>>>(d_intensities, d_output, scansAlignedY, intensitiesX, intensitiesY);
	break;
	case 512:	d_calculateSTXM< 512><<<grid, block, shared_mem_size>>>(d_intensities, d_output, scansAlignedY, intensitiesX, intensitiesY);
	break;
	case 1024:	d_calculateSTXM<1024><<<grid, block, shared_mem_size>>>(d_intensities, d_output, scansAlignedY, intensitiesX, intensitiesY);
	break;
	}
	cutilCheckMsg("d_calculateSTXM() execution failed!\n");
}

__host__ void h_symmetrizeDiffraction(const real_t* d_intensities, real_t* d_output,
					unsigned int dataOffsetX, unsigned int dataOffsetY, unsigned int dataAlignedY,
					unsigned int outOffsetX, unsigned int outOffsetY, unsigned int outAlignedY,
					unsigned int numX, unsigned int numY, cudaStream_t* stream)
{
	d_symmetrizeDiffraction<<<numX, numY, 0, (stream?*stream:0)>>>(d_intensities, d_output, dataOffsetX, dataOffsetY, dataAlignedY, outOffsetX, outOffsetY, outAlignedY);
	cutilCheckMsg("d_symmetrizeDiffraction() execution failed!\n");	
}

__host__ void h_preprocessDiffractions(real_t* d_intensities, bool squareRoot, real_t threshold,
					unsigned int intensitiesNum, unsigned int intensitiesX, unsigned int intensitiesY, unsigned int intensitiesAlignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), intensitiesAlignedY);
	dim3 grid(intensitiesNum, gh_iDivUp(intensitiesX,sliceNum), 1);
	dim3 block(intensitiesAlignedY, sliceNum, 1);
	d_preprocessDiffractions<<<grid, block>>>(d_intensities, squareRoot, threshold, intensitiesX, intensitiesY);
	cutilCheckMsg("d_preprocessDiffractions() execution failed!\n");	
}

#endif /* DIFFRACTIONSERNELS_CU_ */
