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
//Any publication using the package should cite fo
//Yue K, Deng J, Jiang Y, Nashed Y, Vine D, Vogt S.
//Ptychopy: GPU framework for ptychographic data analysis.
//X-Ray Nanoimaging: Instruments and Methods V 2021 .
//International Society for Optics and Photonics.
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
#ifndef SAMPLEKERNELS_CU_
#define SAMPLEKERNELS_CU_

#include "Sample.cuh"
#include "utilities.h"
#include <math.h>
#include <math_constants.h>

size_t g_texOffset = 0;
texture<complex_t, cudaTextureType2D, cudaReadModeElementType> g_objArrayTex;

__global__ void d_initRandomStatesSample(curandState *state, unsigned long seed)
{
	unsigned int posIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	/* Each thread gets same seed, a different sequence number,
	no offset */
	curand_init(seed, posIndex, 0, &state[posIndex]);
}

//__global__ void d_check1(curandState* randStates)
//{
//	unsigned int posIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
//	curandState localState;
//	localState = randStates[posIndex];
//
//
//	float sw=curand_uniform(&localState);
//
//
//	unsigned int sq1=1;
//}

__host__ void h_initRandomStatesSample(unsigned int meshX, unsigned int alignedY, curandState* devStates)
{
    d_initRandomStatesSample<<<meshX, alignedY>>>(devStates, time(NULL));

//    d_check1<<<meshX, alignedY>>>(devStates);

	cutilCheckMsg("h_initRandomStates() execution failed!\n");
}




__global__ void d_applyHammingToSample(complex_t* d_objectArray, real_t* d_sampleR, real_t* d_sampleI, unsigned int sampleY,
										unsigned int xOffset, unsigned int yOffset, unsigned int alignedObjectArrayY)
{
	unsigned int roiIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int objectIndex = ((xOffset+blockIdx.x) * alignedObjectArrayY) + threadIdx.x + yOffset;

	if(threadIdx.x < sampleY)
	{
		real_t sinFunc, cosFunc;
		real_t hammingSampleValue =  (0.53836-(0.46164*cospi_real_t(2.0*threadIdx.x/(blockDim.x-1)))) * (0.53836-(0.46164*cospi_real_t(2.0*blockIdx.x/(blockDim.x-1))));
		real_t realPart = d_sampleR[roiIndex];
		sincos_real_t(d_sampleI[roiIndex],&sinFunc,&cosFunc);
		d_objectArray[objectIndex] = mul_complex_t(make_complex_t(realPart*cosFunc, realPart*sinFunc), make_complex_t(hammingSampleValue, 0.0));
	}
}

template<bool enoughThreads>
__global__ void d_applyHammingToSample(const complex_t* d_sample, complex_t* d_output, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int objectIndex = (row * alignedSampleY) + col;

	if(row<sampleX && col<sampleY)
	{
		real_t hammingSampleValue =  (0.53836-(0.46164*cospi_real_t(2.0*col/(sampleY-1)))) * (0.53836-(0.46164*cospi_real_t(2.0*row/(sampleY-1))));
		d_output[objectIndex] = mul_complex_t( d_sample[objectIndex], make_complex_t(hammingSampleValue, 0.0));
	}
}

__global__ void d_simulatePSI(const complex_t* d_objectArray, const complex_t* d_probe, complex_t* d_psi, float nx, float ny,
								unsigned int alignedPsiY, unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY)
{
	unsigned int row = (blockIdx.x*blockDim.y)+threadIdx.y;
	unsigned int col = threadIdx.x;
	unsigned int outIndex = (row * alignedPsiY) + col;

	if(row<objectArrayX && col<objectArrayY)
		d_psi[outIndex] = mul_complex_t(tex2D(g_objArrayTex, col+ny+0.5f, row+nx+0.5f), d_probe[outIndex]);
}

template<bool useAbs, bool enoughThreads>
__global__ void d_getGradient(const complex_t* d_sample, complex_t* d_output, unsigned int sampleX, unsigned int sampleY,
								unsigned int alignedSampleY, bool applyHamming)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int objectIndex = (row * alignedSampleY) + col;

	if(row<sampleX && col<sampleY)
	{
		int left = col-1;
		if(left<0) left=col;
		left+= (row * blockDim.x);

		int right = col+1;
		if(right>=sampleY) right=col;
		right+= (row * blockDim.x);

		int up = row-1;
		if(up<0) up=row;
		up = (up * blockDim.x) + col;

		int down = row+1;
		if(down>=sampleX) down=row;
		down = (down * blockDim.x) + col;

		complex_t gradient;

		gradient.x =  useAbs?(abs_complex_t(d_sample[left])-abs_complex_t(d_sample[right]))* 0.5 :
							(atan2_real_t(d_sample[left].y, d_sample[left].x)-atan2_real_t(d_sample[right].y, d_sample[right].x))* 0.5;
		gradient.y =  useAbs?(abs_complex_t(d_sample[up])-abs_complex_t(d_sample[down]))* 0.5 :
							(atan2_real_t(d_sample[up].y, d_sample[up].x)-atan2_real_t(d_sample[down].y, d_sample[down].x))* 0.5;

		if(applyHamming)
		{
			real_t hamming =  (0.53836-(0.46164*cospi_real_t(2.0*col/(sampleY-1)))) * (0.53836-(0.46164*cospi_real_t(2.0*row/(sampleY-1))));
			gradient = mul_complex_t( gradient, make_complex_t(hamming, 0.0));
		}

		d_output[objectIndex] = make_complex_t(abs_complex_t(gradient), 0);
	}
}

template<bool enoughThreads>
__global__ void d_extractObjectArray(const complex_t* d_objectArray, complex_t* d_output, unsigned int sampleX, unsigned int sampleY,
										float offsetX, float offsetY, unsigned int alignedSampleY, unsigned int alignedObjectArrayY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	unsigned int outputIndex = (row * alignedSampleY) + col;

	if(row<sampleX && col<sampleY)
		d_output[outputIndex] = tex2D(g_objArrayTex, col+offsetY+0.5f, row+offsetX+0.5f);
}

__global__ void d_getObjectIntensities(const complex_t* d_objectArray, real_t* d_output, unsigned int sampleY,
										unsigned int offsetX, unsigned int offsetY, unsigned int alignedObjectArrayY, bool squared)
{
	unsigned int outputIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int objectIndex = ((offsetX+blockIdx.x) * alignedObjectArrayY) + threadIdx.x + offsetY;

	if(threadIdx.x < sampleY)
	{
		real_t temp = abs_complex_t(d_objectArray[objectIndex]);
		d_output[outputIndex] = squared?temp*temp:temp;
	}
}

template<bool enoughThreads>
__global__ void d_getObjectPhase(const complex_t* d_objectArray, real_t* d_output, unsigned int sampleY, unsigned int alignedSampleY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	unsigned int outputIndex = (row * alignedSampleY) + col;


	if(threadIdx.x < sampleY)
	{
		complex_t objectValue = d_objectArray[outputIndex];
		d_output[outputIndex] = (atan2_real_t(objectValue.y,objectValue.x)+CUDART_PI)/(2*CUDART_PI);
	}
}

__device__ complex_t constrainPhase(real_t mag, real_t arg)
{
	arg = arg<=0?arg:0;
	real_t sinFunc, cosFunc;
	sincos_real_t(arg,&sinFunc,&cosFunc);
	return make_complex_t(mag*cosFunc, mag*sinFunc);
}

__device__ complex_t constrainMag(real_t mag, real_t arg)
{
	mag = mag<=1?mag:1;
	real_t sinFunc, cosFunc;
	sincos_real_t(arg,&sinFunc,&cosFunc);
	return make_complex_t(mag*cosFunc, mag*sinFunc);
}

template<bool enoughThreads>
__global__ void d_constrainObject(complex_t* d_objectArray, bool phaseConstraint, unsigned int sampleY, unsigned int alignedSampleY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	unsigned int outputIndex = (row * alignedSampleY) + col;


	if(threadIdx.x < sampleY)
	{
		complex_t objectValue = d_objectArray[outputIndex];
		constrainMag(abs_complex_t(objectValue), atan2_real_t(objectValue.y,objectValue.x));
		if(phaseConstraint)
			objectValue = constrainPhase(abs_complex_t(objectValue), atan2_real_t(objectValue.y,objectValue.x));
		d_objectArray[outputIndex] = objectValue;
	}
}

template<bool phaseConstraint>
__global__ void d_updateObjectArray(complex_t* d_objectArray, const complex_t* d_probe, const complex_t* d_psi, const complex_t* d_psi_old, real_t* d_intensities,
										unsigned int offsetX, unsigned int offsetY, unsigned int roiOffsetX, unsigned int roiOffsetY,
										real_t c, unsigned int sampleY, unsigned int alignedObjectArrayY)
{
	unsigned int psiIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int objectRow = (offsetX+blockIdx.x);
	unsigned int objectCol = threadIdx.x + offsetY;
	unsigned int objectIndex = (objectRow*alignedObjectArrayY) + objectCol;

	if(threadIdx.x < sampleY)
	{
		complex_t objectValue = add_complex_t(d_objectArray[objectIndex], mul_complex_t( mul_complex_t(conj_complex_t(d_probe[psiIndex]), sub_complex_t(d_psi[psiIndex],d_psi_old[psiIndex])), make_complex_t(c,0.0)));
		real_t magnitude = abs_complex_t(objectValue);

		if(phaseConstraint)
			objectValue = constrainPhase(magnitude, atan2_real_t(objectValue.y,objectValue.x));

		d_objectArray[objectIndex] = objectValue;
		d_intensities[objectIndex] = magnitude*magnitude;
	}
}

template<bool phaseConstraint>
__global__ void d_updateObjectArrayM(complex_t* d_objectArray, const complex_t* d_probe, const complex_t* d_psi, const complex_t* d_psi_old, real_t* d_intensities,
									unsigned int offsetX, unsigned int offsetY, unsigned int roiOffsetX, unsigned int roiOffsetY,
									real_t c, unsigned int modeNum, unsigned int sampleY, unsigned int alignedObjectArrayY)
{
	unsigned int psiIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int objectRow = (offsetX+blockIdx.x);
	unsigned int objectCol = threadIdx.x + offsetY;
	unsigned int objectIndex = (objectRow*alignedObjectArrayY) + objectCol;

	if(threadIdx.x < sampleY)
	{
		complex_t objectValue = mul_complex_t(conj_complex_t(d_probe[psiIndex]), sub_complex_t(d_psi[psiIndex],d_psi_old[psiIndex]));
		for(unsigned int i=1; i<modeNum; ++i)
		{
			unsigned int modalPsiIndex = ((blockIdx.x+(i*gridDim.x)) * blockDim.x) + threadIdx.x;
			objectValue = add_complex_t(objectValue, mul_complex_t(conj_complex_t(d_probe[modalPsiIndex]), sub_complex_t(d_psi[modalPsiIndex],d_psi_old[modalPsiIndex])));
		}
		objectValue = add_complex_t(d_objectArray[objectIndex], mul_complex_t(objectValue, make_complex_t(c,0.0)));
		real_t magnitude = abs_complex_t(objectValue);

		if(phaseConstraint)
			objectValue = constrainPhase(magnitude, atan2_real_t(objectValue.y,objectValue.x));

		d_objectArray[objectIndex] = objectValue;
		d_intensities[objectIndex] = magnitude*magnitude;
	}
}

__global__ void d_postprocess(complex_t* d_objectArray, real_t factor, unsigned int sampleY,
										unsigned int offsetX, unsigned int offsetY, unsigned int alignedObjectArrayY)
{
	unsigned int objectIndex = ((offsetX+blockIdx.x) * alignedObjectArrayY) + threadIdx.x + offsetY;

	if(threadIdx.x < sampleY)
	{
		real_t value = abs_complex_t(d_objectArray[objectIndex]);
		value*=value;
		if(value<=factor)
			d_objectArray[objectIndex] = make_complex_t(0,0);
	}
}

template<bool enoughThreads>
__global__ void d_setObjectArray(complex_t* d_objectArray, const complex_t* d_roi, unsigned int roiX, unsigned int roiY, unsigned int alignedRoiY,
								unsigned int offsetX, unsigned int offsetY, unsigned int alignedObjectArrayY,
								unsigned int roiOffsetX, unsigned int roiOffsetY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	unsigned int roiIndex 	= ((roiOffsetX+row) * alignedRoiY)		+ col + roiOffsetY;
	unsigned int objectIndex= ((offsetX+row) * alignedObjectArrayY)	+ col + offsetY;

	complex_t temp=d_roi[roiIndex];

	if(row<roiX && col<roiY)
		d_objectArray[objectIndex] = d_roi[roiIndex];

}

__global__ void d_initRandObjectArray(complex_t* d_array, real_t* d_randarr1, real_t* d_randarr2, unsigned int sampleX,
		unsigned int sampleY, unsigned int alignedSampleY)
{
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(threadIdx.x < sampleY)
	{
		real_t expright=2*CUDART_PI*d_randarr2[index]*0.1;
		real_t expleft=d_randarr1[index];
		complex_t expTo=make_complex_t((expleft*cos_real_t(expright)), (expleft*sin_real_t(expright)));
		d_array[index]=expTo;
	}
}


template<bool enoughThreads>
__global__ void d_addSubsamples(complex_t* d_objectArray, const complex_t* d_neighbors,
								const uint2* d_nOffsets, const uint2* d_nDims,
								uint2 myOffset, unsigned int neighborNum,
								unsigned int sampleX, unsigned int objectArrayY,
								unsigned int alignedsampleY, unsigned int alignedObjectArrayY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int objectIndex = (row * alignedObjectArrayY) + col;

	if(col < objectArrayY)
	{
		complex_t accValue = d_objectArray[objectIndex];

		for(unsigned int i=0; i<neighborNum; ++i)
		{
			uint2 nOffset = d_nOffsets[i];
			uint2 nDims = d_nDims[i];
			nDims = make_uint2(nDims.x+nOffset.x, nDims.y+nOffset.y);
			uint2 myPos = make_uint2(row+myOffset.x, col+myOffset.y);
			if( myPos.x>=nOffset.x && myPos.y>=nOffset.y && myPos.x<nDims.x && myPos.y<nDims.y )
			{
				uint2 nPos = make_uint2(myPos.x-nOffset.x, myPos.y-nOffset.y);
				unsigned int nIndex = ((nPos.x+(i*sampleX)) * alignedsampleY) + nPos.y;
				accValue = add_complex_t(accValue, d_neighbors[nIndex]);
				accValue = mul_complex_t(accValue, make_complex_t(0.5,0.0));
			}
		}

		d_objectArray[objectIndex] = accValue;
	}
}

template<bool enoughThreads>
__global__ void d_mergeObjectArray(complex_t* d_objectArray, const complex_t* d_s1, const complex_t* d_s2,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY,
								unsigned int s2OffsetX, unsigned int s2OffsetY, unsigned int alignedS1Y, unsigned int alignedS2Y,
								unsigned int px, unsigned int py)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	if(row < objectArrayX)
	{
		unsigned int objectIndex = (row * alignedObjectArrayY) + col;
		unsigned int s1Index = (row * alignedS1Y) + col;
		unsigned int s2Index = ((s2OffsetX+(row-px)) * alignedS2Y) + (col-py) + s2OffsetY;

		if(col < objectArrayY)
		{
			if(px==0) px=objectArrayX;
			if(py==0) py=objectArrayY;
			d_objectArray[objectIndex] = (col<py && row<px) ? d_s1[s1Index] : d_s2[s2Index];
		}
	}
}

template<bool enoughThreads>
__global__ void d_crossPowerSpectrum(const complex_t* d_F1, const complex_t* d_F2, complex_t* d_cps,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		complex_t g1 =  d_F1[index];
		complex_t g2 =  conj_complex_t(d_F2[index]);
		d_cps[index] = div_complex_t( mul_complex_t(g1,g2), make_complex_t(abs_complex_t(g1)*abs_complex_t(g2), 0.0));
	}
}

////////////////////////// DM Kernels /////////////////////////////////////////////////////////////////////////////////////

__device__ cufftComplex d_calculatePsi(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
	DM_info* args = (DM_info*)callerInfo;
	unsigned int scanPosIndex = offset/(args->probeX*args->alignedProbeY);
	unsigned int row = (offset/args->alignedProbeY)%args->probeX;
	unsigned int col = offset%args->alignedProbeY;
	unsigned int probeIndex = (row*args->alignedProbeY) + col;
	float2 shift = args->d_scanPos[scanPosIndex];
	unsigned int objectIndex = ( (row+roundf(shift.x))*args->alignedObjectArrayY) + col + roundf(shift.y);

	complex_t psi = ((complex_t*)dataIn)[offset];
	complex_t psi_hat = mul_complex_t(args->d_probe[probeIndex], args->d_objectArray[objectIndex]);

	return (cufftComplex)sub_complex_t(mul_complex_t(psi_hat,make_complex_t(2,0)), psi);
}
__device__ 	COMPLEX_LOAD_CALLBACK d_calcPsiCB = d_calculatePsi;
			COMPLEX_LOAD_CALLBACK h_calcPsiCB = 0;

__device__ void d_modulusConstraint(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
	DM_info* args = (DM_info*)callerInfo;

	complex_t psi = ((complex_t)element);
	real_t det_mod = args->d_intensities[offset];

	real_t sinFunc, cosFunc;
	sincos_real_t(atan2_real_t(psi.y,psi.x),&sinFunc,&cosFunc);
	psi = mul_complex_t(make_complex_t(det_mod*cosFunc, det_mod*sinFunc), make_complex_t(1.0/(real_t)(args->probeX*args->probeY), 0));

	((cufftComplex*)dataOut)[offset] = (cufftComplex)psi;
}
__device__ 	COMPLEX_STORE_CALLBACK d_modConstraintCB = d_modulusConstraint;
			COMPLEX_STORE_CALLBACK h_modConstraintCB = 0;

__device__ void d_updateObject(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
	DM_info* args = (DM_info*)callerInfo;
	unsigned int scanPosIndex = offset/(args->probeX*args->alignedProbeY);
	unsigned int row = (offset/args->alignedProbeY)%args->probeX;
	unsigned int col = offset%args->alignedProbeY;
	unsigned int probeIndex = (row*args->alignedProbeY) + col;
	float2 shift = args->d_scanPos[scanPosIndex];
	unsigned int objectIndex = ( (row+roundf(shift.x))*args->alignedObjectArrayY) + col + roundf(shift.y);

	complex_t objectValue = args->d_objectArray[objectIndex];
	complex_t probeValue  = args->d_probe[probeIndex];
	complex_t psi = args->d_psi[offset];

	complex_t Pf = ((complex_t)element);
	complex_t Po = mul_complex_t(objectValue, probeValue);
	psi = sub_complex_t(add_complex_t(psi, Pf), mul_complex_t(Po,make_complex_t(2,0)));

	complex_t updatedValue = mul_complex_t(conj_complex_t(probeValue), psi);
	updatedValue = mul_complex_t(updatedValue, make_complex_t(args->d_normalizingFactors[1],0.0));
	//constrainMag(abs_complex_t(updatedValue), atan2_real_t(updatedValue.y,updatedValue.x));
	if(args->d_flags[0])
		updatedValue = constrainPhase(abs_complex_t(updatedValue), atan2_real_t(updatedValue.y,updatedValue.x));
	atomicAdd(&(args->d_objectArray[objectIndex].x), updatedValue.x);
	atomicAdd(&(args->d_objectArray[objectIndex].y), updatedValue.y);

	if(args->d_flags[1])
	{
		updatedValue = mul_complex_t(conj_complex_t(objectValue), psi);
		updatedValue = mul_complex_t(updatedValue, make_complex_t(args->d_normalizingFactors[0],0.0));
		atomicAdd(&(args->d_probe[probeIndex].x), updatedValue.x);
		atomicAdd(&(args->d_probe[probeIndex].y), updatedValue.y);
	}

	args->d_psi[offset] = psi;
	((cufftComplex*)dataOut)[offset] = (cufftComplex)psi;
}
__device__ 	COMPLEX_STORE_CALLBACK d_updateCB = d_updateObject;
			COMPLEX_STORE_CALLBACK h_updateCB = 0;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ void h_bindObjArrayToTex(complex_t* d_objectArray, unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<complex_t>();
	g_objArrayTex.addressMode[0] = cudaAddressModeClamp;
	g_objArrayTex.addressMode[1] = cudaAddressModeClamp;
	g_objArrayTex.filterMode = cudaFilterModeLinear;
	g_objArrayTex.normalized = false;
	cudaBindTexture2D(&g_texOffset, g_objArrayTex, d_objectArray, channelDesc, objectArrayY, objectArrayX, alignedObjectArrayY*sizeof(complex_t));
	cutilCheckMsg("h_bindObjectArrayToTex()::cudaBindTexture2D() failed!\n");
}

__host__ void h_unbindObjArrayTex()
{
	cudaUnbindTexture(g_objArrayTex);
	cutilCheckMsg("h_unbindTex()::cudaUnbindTexture() failed!\n");
}

__host__ void h_applyHammingToSample(complex_t* d_objectArray, real_t* d_sampleR, real_t* d_sampleI,
									unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
									unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY)
{
	unsigned int offsetX = (objectArrayX/2)-(sampleX/2);
	unsigned int offsetY = (objectArrayY/2)-(sampleY/2);
	d_applyHammingToSample<<<sampleX, alignedSampleY>>>(d_objectArray, d_sampleR, d_sampleI, sampleY, offsetX, offsetY, alignedObjectArrayY);
	cutilCheckMsg("h_applyHammingToSample() execution failed!\n");
}

__host__ void h_applyHammingToSample(const complex_t* d_sample, complex_t* d_output,
									unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(sampleX,alignedSampleY,grid,block);
	if(enoughThreads)	d_applyHammingToSample<true> <<<grid, block>>>(d_sample, d_output, sampleX, sampleY, alignedSampleY);
	else				d_applyHammingToSample<false><<<grid, block>>>(d_sample, d_output, sampleX, sampleY, alignedSampleY);
	cutilCheckMsg("h_applyHammingToSample() execution failed!\n");
}

__host__ void h_simulatePSI(const complex_t* d_objectArray, const complex_t* d_probe, complex_t* d_psi, float2 scanPos,
							unsigned int psiX, unsigned int psiY, unsigned int alignedPsiY, unsigned int objectArrayX,
							unsigned int objectArrayY, unsigned int alignedObjectArrayY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedPsiY);
	dim3 grid(gh_iDivUp(psiX, sliceNum), 1, 1);
	dim3 block(alignedPsiY, sliceNum, 1);

	d_simulatePSI<<<grid, block>>>(d_objectArray, d_probe, d_psi, scanPos.x, scanPos.y,
								 alignedPsiY, objectArrayX, objectArrayY, alignedObjectArrayY);
	cutilCheckMsg("h_simulatePSI() execution failed!\n");
}

__host__ void h_getPhaseGradient(const complex_t* d_sample, complex_t* d_output, unsigned int sampleX, unsigned int sampleY,
									unsigned int alignedSampleY, bool applyHamming)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(sampleX,alignedSampleY,grid,block);
	if(enoughThreads)	d_getGradient<false, true> <<<grid, block>>>(d_sample, d_output, sampleX, sampleY, alignedSampleY, applyHamming);
	else				d_getGradient<false, false><<<grid, block>>>(d_sample, d_output, sampleX, sampleY, alignedSampleY, applyHamming);
	cutilCheckMsg("h_getPhaseGradient() execution failed!\n");
}

__host__ void h_getMagnitudeGradient(const complex_t* d_sample, complex_t* d_output, unsigned int sampleX, unsigned int sampleY,
									unsigned int alignedSampleY, bool applyHamming)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(sampleX,alignedSampleY,grid,block);
	if(enoughThreads)	d_getGradient<true, true> <<<grid, block>>>(d_sample, d_output, sampleX, sampleY, alignedSampleY, applyHamming);
	else 				d_getGradient<true, false><<<grid, block>>>(d_sample, d_output, sampleX, sampleY, alignedSampleY, applyHamming);
	cutilCheckMsg("h_getMagnitudeGradient() execution failed!\n");
}

__host__ void h_extractObjectArray(complex_t* d_objectArray, complex_t* d_output, float offsetX, float offsetY,
									unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
									unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(sampleX,alignedSampleY,grid,block);
	if(enoughThreads)	d_extractObjectArray<true> <<<grid, block>>>(d_objectArray, d_output, sampleX, sampleY, offsetX, offsetY, alignedSampleY, alignedObjectArrayY);
	else				d_extractObjectArray<false><<<grid, block>>>(d_objectArray, d_output, sampleX, sampleY, offsetX, offsetY, alignedSampleY, alignedObjectArrayY);
	cutilCheckMsg("h_extractObjectArray() execution failed!\n");
}


__host__ void h_getObjectIntensities(complex_t* d_objectArray, real_t* d_output, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
										unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY, bool squared)
{
	d_getObjectIntensities<<<sampleX, alignedSampleY>>>(d_objectArray, d_output, sampleY, (objectArrayX/2)-(sampleX/2), (objectArrayY/2)-(sampleY/2), alignedObjectArrayY, squared);
	cutilCheckMsg("h_getObjectIntensities() execution failed!\n");
}

__host__ void h_getObjectPhase(complex_t* d_objectArray, real_t* d_output, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(sampleX,alignedSampleY,grid,block);
	if(enoughThreads)	d_getObjectPhase<true> <<<grid, block>>>(d_objectArray, d_output, sampleY, alignedSampleY);
	else				d_getObjectPhase<false><<<grid, block>>>(d_objectArray, d_output, sampleY, alignedSampleY);
	cutilCheckMsg("h_getObjectPhase() execution failed!\n");
}

__host__ void h_updateObjectArray(complex_t* d_objectArray, const complex_t* d_probe, const complex_t* d_psi, const complex_t* d_psi_old,
										real_t* d_intensities, unsigned int offsetX, unsigned int offsetY, real_t normalizationFactor,
										unsigned int modeNum, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
										unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY, bool phaseConstraint)
{
	unsigned int roiOffsetX =(objectArrayX/2)-(sampleX/2);
	unsigned int roiOffsetY =(objectArrayY/2)-(sampleY/2);
	if(phaseConstraint)
	{
		if(modeNum>1)
			d_updateObjectArrayM<true><<<sampleX, alignedSampleY>>>(d_objectArray, d_probe, d_psi, d_psi_old, d_intensities, offsetX, offsetY,
														roiOffsetX, roiOffsetY, normalizationFactor, modeNum, sampleY, alignedObjectArrayY);
		else
			d_updateObjectArray<true><<<sampleX, alignedSampleY>>>(d_objectArray, d_probe, d_psi, d_psi_old, d_intensities, offsetX, offsetY,
													roiOffsetX, roiOffsetY, normalizationFactor, sampleY, alignedObjectArrayY);
	}
	else
	{
			if(modeNum>1)
				d_updateObjectArrayM<false><<<sampleX, alignedSampleY>>>(d_objectArray, d_probe, d_psi, d_psi_old, d_intensities, offsetX, offsetY,
															roiOffsetX, roiOffsetY, normalizationFactor, modeNum, sampleY, alignedObjectArrayY);
			else
				d_updateObjectArray<false><<<sampleX, alignedSampleY>>>(d_objectArray, d_probe, d_psi, d_psi_old, d_intensities, offsetX, offsetY,
														roiOffsetX, roiOffsetY, normalizationFactor, sampleY, alignedObjectArrayY);
	}
	cutilCheckMsg("h_updateObjectArray() execution failed!\n");
}

__host__ void h_postprocess(complex_t* d_objectArray, real_t thresholdFactor, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY)
{
	d_postprocess<<<sampleX, alignedSampleY>>>(d_objectArray, thresholdFactor, sampleY, (objectArrayX/2)-(sampleX/2), (objectArrayY/2)-(sampleY/2), alignedObjectArrayY);
	cutilCheckMsg("h_postprocess() execution failed!\n");
}

__host__ void h_setObjectArray(complex_t* d_objectArray, const complex_t* d_roi,  unsigned int roiX, unsigned int roiY, unsigned int alignedROIY,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY, unsigned int roiOffsetX,
								unsigned int roiOffsetY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(roiX,alignedROIY,grid,block);

	if(enoughThreads) 	d_setObjectArray<true><<<grid, block>>>(d_objectArray, d_roi, roiX, roiY, alignedROIY,
											(objectArrayX/2)-(roiX/2), (objectArrayY/2)-(roiY/2),
											alignedObjectArrayY, roiOffsetX, roiOffsetY);
	else			 	d_setObjectArray<false><<<grid, block>>>(d_objectArray, d_roi, roiX, roiY, alignedROIY,
											(objectArrayX/2)-(roiX/2), (objectArrayY/2)-(roiY/2),
											alignedObjectArrayY, roiOffsetX, roiOffsetY);
	cutilCheckMsg("h_setObjectArray() execution failed!\n");
}

__host__ void h_initRandObjectArray(complex_t* d_array, real_t* d_randarr1, real_t* d_randarr2, unsigned int sampleX,
		unsigned int sampleY, unsigned int alignedSampleY)
{
	d_initRandObjectArray<<<sampleX, alignedSampleY>>>(d_array, d_randarr1, d_randarr2, sampleX,
			sampleY, alignedSampleY);
}

__host__ void h_addSubsamples(complex_t* d_objectArray, const complex_t* d_neighbors, const uint2* d_nOffset, const uint2* d_nDims,
								unsigned int sampleX, unsigned int alignedSampleY, uint2 myOffset, unsigned int neighborNum,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(objectArrayX,objectArrayY,grid,block);

	if(enoughThreads)	d_addSubsamples<true> <<<grid, block>>>(d_objectArray, d_neighbors, d_nOffset, d_nDims, myOffset, neighborNum,
																sampleX, objectArrayY, alignedSampleY, alignedObjectArrayY);
	else				d_addSubsamples<false><<<grid, block>>>(d_objectArray, d_neighbors, d_nOffset, d_nDims, myOffset, neighborNum,
																sampleX, objectArrayY, alignedSampleY, alignedObjectArrayY);
	cutilCheckMsg("h_addSubsamples() execution failed!\n");
}

__host__ void h_mergeObjectArray(complex_t* d_objectArray, const complex_t* d_s1, const complex_t* d_s2,
								unsigned int alignedS1Y, unsigned int alignedS2Y, unsigned int s2OffsetX, unsigned int s2OffsetY,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY, unsigned int px, unsigned int py)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(objectArrayX,alignedObjectArrayY,grid,block);
	if(enoughThreads) 	d_mergeObjectArray<true> <<<grid, block>>>(d_objectArray, d_s1, d_s2, objectArrayX, objectArrayY, alignedObjectArrayY,
											s2OffsetX, s2OffsetY, alignedS1Y, alignedS2Y, px, py);
	else				d_mergeObjectArray<false><<<grid, block>>>(d_objectArray, d_s1, d_s2, objectArrayX, objectArrayY, alignedObjectArrayY,
											s2OffsetX, s2OffsetY, alignedS1Y, alignedS2Y, px, py);
	cutilCheckMsg("h_mergeObjectArray() execution failed!\n");
}

__host__ void h_crossPowerSpectrum(const complex_t* d_F1, const complex_t* d_F2, complex_t* d_cps,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads) 	d_crossPowerSpectrum<true> <<<grid, block>>>(d_F1, d_F2, d_cps, x, y, alignedY);
	else				d_crossPowerSpectrum<false><<<grid, block>>>(d_F1, d_F2, d_cps, x, y, alignedY);
	cutilCheckMsg("d_crossPowerSpectrum() execution failed!\n");
}

__host__ void h_forwardPropagate(cufftHandle fftPlan, DM_info* args, complex_t* d_psi, bool phaseConstraint)
{
	if(!h_calcPsiCB)
	{
		cudaMemcpyFromSymbol(&h_calcPsiCB,d_calcPsiCB, sizeof(h_calcPsiCB));
		cutilCheckMsg("h_forwardPropagate():cudaMemcpyFromSymbol() execution failed!\n");
		cufftXtSetCallback(fftPlan, (void **)&h_calcPsiCB, CB_LD_COMPLEX, (void**)&args );
		cutilCheckMsg("h_forwardPropagate():cufftXtSetCallback() execution failed!\n");
	}
	if(!h_modConstraintCB)
	{
		cudaMemcpyFromSymbol(&h_modConstraintCB,d_modConstraintCB, sizeof(h_modConstraintCB));
		cutilCheckMsg("h_backPropagate():cudaMemcpyFromSymbol() execution failed!\n");
		cufftXtSetCallback(fftPlan, (void **)&h_modConstraintCB, CB_ST_COMPLEX, (void**)&args );
		cutilCheckMsg("h_backPropagate():cufftXtSetCallback() execution failed!\n");
	}

	COMPLEX2COMPLEX_FFT(fftPlan, d_psi, d_psi, CUFFT_FORWARD);
	cutilCheckMsg("h_forwardPropagate():fft() execution failed!\n");
}

__host__ void h_backPropagate(cufftHandle fftPlan, DM_info* args, complex_t* d_psi)
{
	if(!h_updateCB)
	{
		cudaMemcpyFromSymbol(&h_updateCB,d_updateCB, sizeof(h_updateCB));
		cutilCheckMsg("h_backPropagate():cudaMemcpyFromSymbol() execution failed!\n");
		cufftXtSetCallback(fftPlan, (void **)&h_updateCB, CB_ST_COMPLEX, (void**)&args );
		cutilCheckMsg("h_backPropagate():cufftXtSetCallback() execution failed!\n");
	}

	COMPLEX2COMPLEX_FFT(fftPlan, d_psi, d_psi, CUFFT_INVERSE);
	cutilCheckMsg("h_backPropagate():ifft() execution failed!\n");
}

__host__ void h_constrainObject(complex_t* d_objectArray, bool phaseConstraint, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(sampleX,alignedSampleY,grid,block);
	if(enoughThreads)	d_constrainObject<true> <<<grid, block>>>(d_objectArray, phaseConstraint, sampleY, alignedSampleY);
	else				d_constrainObject<false><<<grid, block>>>(d_objectArray, phaseConstraint, sampleY, alignedSampleY);
	cutilCheckMsg("h_constrainObject() execution failed!\n");
}

#endif /* SAMPLEKERNELS_CU_ */
