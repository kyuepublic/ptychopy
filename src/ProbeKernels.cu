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

#ifndef PROBEKERNELS_CU_
#define PROBEKERNELS_CU_

#include "Probe.cuh"
#include "utilities.h"
#include "reductions.cu"

#include <math.h>
#include <math_constants.h>

/* extern shared memory for dynamic allocation */
extern __shared__ real_t shared_array[];

__device__ real_t d_hypotOgrid(unsigned int x, unsigned int y, unsigned int N)
{
	real_t a =  x-(N*0.5);
	real_t b =  y-(N*0.5);
	return sqrt_real_t(a*a+b*b);
}

__device__ real_t d_calculateGaussian(unsigned int x, unsigned int y, unsigned int N, real_t window)
{
	real_t a = d_hypotOgrid(x,y,N);
	return exp(pow(window,-2)*a*a*-1);
}

__global__ void d_simulateGaussian(real_t* d_gauss, real_t window, unsigned int probeY)
{
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(threadIdx.x < probeY)
		d_gauss[probeIndex] = d_calculateGaussian(blockIdx.x, threadIdx.x, gridDim.x, window);
}

__global__ void d_initModesFromBase(complex_t* d_modes, real_t factor, real_t* d_intensities, 
									unsigned int probeX, unsigned int probeY)
{
	//unsigned int baseIndex = (blockIdx.x * probeX) + ((blockIdx.y*blockDim.y) + threadIdx.y);
	unsigned int row = (blockIdx.y*blockDim.y) + threadIdx.y;
	unsigned int modeIndex = ((row+((blockIdx.x+1) * probeX)) * blockDim.x) + threadIdx.x;
	unsigned int baseIndex = (row*blockDim.x) + threadIdx.x;

	if(row<probeX && threadIdx.x<probeY)
	{
		complex_t baseValue = mul_complex_t(d_modes[baseIndex], make_complex_t(factor,0));
		real_t baseIntensity = abs_complex_t(baseValue);
		d_modes[modeIndex] = baseValue;
		d_intensities[modeIndex] = baseIntensity * baseIntensity;
	}
}

__global__ void d_simulateProbe(complex_t* d_probeWavefront, real_t* d_phFunc, real_t window, real_t factor, unsigned int probeY)
{
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(threadIdx.x < probeY)
	{
		real_t sinFunc, cosFunc;
		real_t ortho_mode = d_calculateGaussian(blockIdx.x, threadIdx.x, gridDim.x, window) *
							sinpi_real_t(uint2real_t(threadIdx.x)/uint2real_t(gridDim.x)) *
							sinpi_real_t(uint2real_t(blockIdx.x)/uint2real_t(gridDim.x));
		sincospi_real_t(d_phFunc[probeIndex],&sinFunc,&cosFunc);
		cosFunc*=ortho_mode;
		sinFunc*=ortho_mode;
		d_probeWavefront[probeIndex] = make_complex_t(cosFunc==0?abs(cosFunc):cosFunc, sinFunc==0?abs(sinFunc):sinFunc);
	}
}

__global__ void d_initProbeSimulated(complex_t* d_probeWavefront, real_t* d_phFunc, real_t window, real_t factor, unsigned int probeY)
{
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(threadIdx.x < probeY)
	{
		real_t sinFunc, cosFunc;
		real_t probe = d_hypotOgrid(blockIdx.x, threadIdx.x, gridDim.x)*factor;
		probe = (probe!=0)? sinpi_real_t(probe)/(CUDART_PI*probe) : 1;
		probe = abs(probe*d_calculateGaussian(blockIdx.x, threadIdx.x, gridDim.x, window));

		sincospi_real_t(d_phFunc[probeIndex],&sinFunc,&cosFunc);
		d_probeWavefront[probeIndex] = make_complex_t(probe*cosFunc, probe*sinFunc);
	}
}

__global__ void d_initProbe(complex_t* d_probeWavefront, real_t window, real_t factor, unsigned int probeY)
{
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(threadIdx.x < probeY)
	{
		real_t probe = d_hypotOgrid(blockIdx.x, threadIdx.x, gridDim.x)*factor;
		probe = (probe!=0)? sinpi_real_t(probe)/(CUDART_PI*probe) : 1;
		probe = abs(probe*d_calculateGaussian(blockIdx.x, threadIdx.x, gridDim.x, window));

		d_probeWavefront[probeIndex] = make_complex_t(probe, 0.0);
	}
}

template<unsigned int threadNum>
__global__ void d_normalizeDiffractionIntensities(const real_t* d_intensities, real_t* d_output, real_t normalizeBy,
													unsigned int intensitiesNum, unsigned int alignedY)
{
	real_t* s_addends = (real_t*)shared_array;
	unsigned int outputIndex = (blockIdx.x * alignedY) + blockIdx.y;
	unsigned int tid = threadIdx.x;
	s_addends[threadIdx.x] = 0.0;

	while(tid<intensitiesNum)
	{
		s_addends[threadIdx.x] += d_intensities[(((tid*gridDim.x) + blockIdx.x) * alignedY) + blockIdx.y];
		tid += blockDim.x;
	}

	reduceToSum<real_t,threadNum>(s_addends,threadIdx.x);

	if(threadIdx.x == 0)
		d_output[outputIndex] = s_addends[0]*normalizeBy;
}

__global__ void d_probeModalSum(const complex_t* d_modes, complex_t* d_output, unsigned int modeNum, unsigned int x, unsigned int y)
{
	//unsigned int baseIndex = (blockIdx.x * probeX) + ((blockIdx.y*blockDim.y) + threadIdx.y);
	unsigned int modeIndex = (blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int outIndex = (modeIndex*blockDim.x) + threadIdx.x;

	if(threadIdx.x<y)
	{
		complex_t val = d_modes[outIndex];
		for(unsigned int i=1; i<modeNum; ++i)
			val = add_complex_t(val, d_modes[((modeIndex+(i*x))*blockDim.x) + threadIdx.x]);

		d_output[outIndex] = val;
	}
}

__global__ void d_updateProbe(complex_t* d_probe, const complex_t* d_objectArray, const complex_t* d_psi, const complex_t* d_psi_old,
								real_t* d_intensities, unsigned int offsetX, unsigned int offsetY, real_t c, unsigned int sampleY, unsigned int alignedObjectArrayY)
{
	unsigned int psiIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int objectIndex = ((offsetX+blockIdx.x) * alignedObjectArrayY) + threadIdx.x + offsetY;

	if(threadIdx.x < sampleY)
	{
		complex_t probeValue = add_complex_t(mul_complex_t( mul_complex_t(conj_complex_t(d_objectArray[objectIndex]),sub_complex_t(d_psi[psiIndex],d_psi_old[psiIndex])), make_complex_t(c,0.0)) , d_probe[psiIndex]);
		real_t intensity = abs_complex_t(probeValue);
		d_probe[psiIndex] = probeValue;
		d_intensities[psiIndex] = intensity*intensity;
	}
}

__global__ void d_updateProbeM(complex_t* d_probe, const complex_t* d_objectArray, const complex_t* d_psi, const complex_t* d_psi_old,
								real_t* d_intensities, unsigned int offsetX, unsigned int offsetY, real_t c, unsigned int modeNum,
								unsigned int sampleY, unsigned int alignedObjectArrayY)
{
	unsigned int objectIndex = ((offsetX+blockIdx.x) * alignedObjectArrayY) + threadIdx.x + offsetY;

	if(threadIdx.x < sampleY)
	{
		complex_t objectConj = conj_complex_t(d_objectArray[objectIndex]);
		for(unsigned int i=0; i<modeNum; ++i)
		{
			unsigned int psiIndex = ((blockIdx.x+(i*gridDim.x)) * blockDim.x) + threadIdx.x;
			complex_t probeValue = add_complex_t(mul_complex_t( mul_complex_t(objectConj,sub_complex_t(d_psi[psiIndex],d_psi_old[psiIndex])), make_complex_t(c,0.0)) , d_probe[psiIndex]);
			real_t intensity = abs_complex_t(probeValue);
			d_probe[psiIndex] = probeValue;
			d_intensities[psiIndex] = intensity*intensity;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__host__ void h_projectUtoV(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
								unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
{
	complex_t factor = div_complex_t(h_innerProduct(d_v, d_u, d_output, probeX, probeY, alignedProbeY) , h_innerProduct(d_u, d_u, d_output, probeX, probeY, alignedProbeY));
	h_multiply(d_u, factor, d_output, probeX, probeY, alignedProbeY);
	cutilCheckMsg("h_projectUtoV() execution failed!\n");
}

__host__ void h_initModesFromBase(complex_t* d_modes, unsigned int modesNum, real_t factor, real_t* d_intensities, 
								unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedProbeY);
	dim3 grid(modesNum-1, gh_iDivUp(probeX,sliceNum), 1);
	dim3 block(alignedProbeY, sliceNum, 1);
	d_initModesFromBase<<<grid, block>>>(d_modes, factor, d_intensities, probeX, probeY);
	cutilCheckMsg("d_initModesFromBase() execution failed!\n");
}

__host__ void h_simulateGaussian(real_t* d_gauss, real_t window,
					unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
{
	d_simulateGaussian<<<probeX, alignedProbeY>>>(d_gauss, window, probeY);
	cutilCheckMsg("d_simulateGaussian() execution failed!\n");
}

__host__ void h_simulateProbe(complex_t* d_probeWavefront, real_t* d_phFunc, real_t gaussWindow, real_t beamSize, real_t dx_s,
					unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
{
	d_simulateProbe<<<probeX, alignedProbeY>>>(d_probeWavefront, d_phFunc, gaussWindow, dx_s/beamSize, probeY);
	cutilCheckMsg("d_simulateProbe() execution failed!\n");
}

__host__ void h_initProbe(complex_t* d_probeWavefront, real_t* d_phFunc, real_t gaussWindow, real_t beamSize, real_t dx_s,
					unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY, bool simulated)
{
	if(simulated)
		d_initProbeSimulated<<<probeX, alignedProbeY>>>(d_probeWavefront, d_phFunc, gaussWindow, dx_s/beamSize, probeY);
	else
		d_initProbe<<<probeX, alignedProbeY>>>(d_probeWavefront, gaussWindow, dx_s/beamSize, probeY);
	cutilCheckMsg("d_initProbe() execution failed!\n");
}

__host__ void h_normalizeDiffractionIntensities(const real_t* d_intensities, real_t* d_output, unsigned int intensitiesNum,
					unsigned int intensitiesX, unsigned int intensitiesY, unsigned int intensitiesAlignedY)
{
	int roundedBlockSize = getReductionThreadNum(intensitiesNum);
	dim3 grid(intensitiesX, intensitiesY, 1);
	dim3 block(roundedBlockSize<=GPUQuery::getInstance()->getGPUMaxThreads()?roundedBlockSize:GPUQuery::getInstance()->getGPUMaxThreads(), 1, 1);
	size_t shared_mem_size = (block.x <= 32) ? 2* block.x * sizeof(real_t) : block.x *  sizeof(real_t);
	real_t normalizeBy = 1.0/(real_t)intensitiesNum;
	switch (block.x)
	{
	case   8:	d_normalizeDiffractionIntensities<   8><<<grid, block, shared_mem_size>>>(d_intensities, d_output, normalizeBy, intensitiesNum, intensitiesAlignedY);
	break;
	case  16:	d_normalizeDiffractionIntensities<  16><<<grid, block, shared_mem_size>>>(d_intensities, d_output, normalizeBy, intensitiesNum, intensitiesAlignedY);
	break;
	case  32:	d_normalizeDiffractionIntensities<  32><<<grid, block, shared_mem_size>>>(d_intensities, d_output, normalizeBy, intensitiesNum, intensitiesAlignedY);
	break;
	case  64:	d_normalizeDiffractionIntensities<  64><<<grid, block, shared_mem_size>>>(d_intensities, d_output, normalizeBy, intensitiesNum, intensitiesAlignedY);
	break;
	case 128:	d_normalizeDiffractionIntensities< 128><<<grid, block, shared_mem_size>>>(d_intensities, d_output, normalizeBy, intensitiesNum, intensitiesAlignedY);
	break;
	case 256:	d_normalizeDiffractionIntensities< 256><<<grid, block, shared_mem_size>>>(d_intensities, d_output, normalizeBy, intensitiesNum, intensitiesAlignedY);
	break;
	case 512:	d_normalizeDiffractionIntensities< 512><<<grid, block, shared_mem_size>>>(d_intensities, d_output, normalizeBy, intensitiesNum, intensitiesAlignedY);
	break;
	case 1024:	d_normalizeDiffractionIntensities<1024><<<grid, block, shared_mem_size>>>(d_intensities, d_output, normalizeBy, intensitiesNum, intensitiesAlignedY);
	break;
	}
	cutilCheckMsg("d_normalizeDiffractionIntensities() execution failed!\n");
}

__host__ void h_probeModalSum(const complex_t* d_probeModes, complex_t* d_output, unsigned int modesNum,
								unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedProbeY);
	dim3 grid(gh_iDivUp(probeX,sliceNum), 1, 1);
	dim3 block(alignedProbeY, sliceNum, 1);
	
	d_probeModalSum<<<grid, block>>>(d_probeModes, d_output, modesNum, probeX, probeY);
	cutilCheckMsg("d_probeModalSum() execution failed!\n");
}

__host__ void h_updateProbe(complex_t* d_probe, const complex_t* d_objectArray, const complex_t* d_psi, const complex_t* d_psi_old,
							real_t* d_intensities, unsigned int offsetX, unsigned int offsetY, real_t normalizationFactor,
							unsigned int modeNum, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY,
							unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY)
{
	if(modeNum>1)
		d_updateProbeM<<<probeX, alignedProbeY>>>(d_probe, d_objectArray, d_psi, d_psi_old, d_intensities, offsetX, offsetY,
														normalizationFactor, modeNum, probeY, alignedObjectArrayY);
	else
		d_updateProbe<<<probeX, alignedProbeY>>>(d_probe, d_objectArray, d_psi, d_psi_old, d_intensities, offsetX, offsetY,
														normalizationFactor, probeY, alignedObjectArrayY);
	cutilCheckMsg("h_updateProbe() execution failed!\n");
}

#endif /* PROBEKERNELS_CU_ */
