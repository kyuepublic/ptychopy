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

#ifndef UTILITIESKERNELS_CU_
#define UTILITIESKERNELS_CU_

#include "utilities.h"
#include "reductions.cu"

#include <math.h>
#include <math_constants.h>
#include <float.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>

/* extern shared memory for dynamic allocation */
extern __shared__ real_t shared_array[];

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
texture<float4, 1, cudaReadModeElementType> g_transferTex; // 1D transfer function texture
cudaArray *d_transferFuncArray = 0;

struct complexSum
{
	complex_t normalizeBy;
	complexSum(complex_t f=make_complex_t(1,0)) : normalizeBy(f)
	{}
  __host__ __device__ complex_t operator()(const complex_t&lhs, const complex_t&rhs) const 
  {return mul_complex_t(add_complex_t(lhs,rhs),normalizeBy);}
};

struct maxFloat2
{
	__host__ __device__ float2 operator()(float2 lhs, float2 rhs)
	{return make_float2(thrust::max(lhs.x, rhs.x), thrust::max(lhs.y, rhs.y));}
};

struct minFloat2
{
	__host__ __device__ float2 operator()(float2 lhs, float2 rhs)
	{return make_float2(thrust::min(lhs.x, rhs.x), thrust::min(lhs.y, rhs.y));}
};
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<unsigned int threadNum>
__global__ void d_reduceToSum(const complex_t* d_u, complex_t* d_output, unsigned int x1, unsigned int y1,
								unsigned int xNum, unsigned int yNum, unsigned int alignedY, bool enoughThreads)
{
	complex_t* s_addends = (complex_t*)shared_array;

	unsigned int row = blockIdx.x;//(blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int outIndex = enoughThreads? blockIdx.x : (blockIdx.y*gridDim.x) + blockIdx.x;
	unsigned int index = ((row+x1)*alignedY) + col + y1;

	if(row<xNum)
	{
		s_addends[threadIdx.x] = (col<yNum)? d_u[index] : make_complex_t(0,0);

		reduceToSumComplex<threadNum>(s_addends, threadIdx.x);

		if(threadIdx.x == 0)
			d_output[outIndex] = s_addends[0];
	}
}

template<unsigned int threadNum>
__global__ void d_reduceToSum(const real_t* d_u, real_t* d_output, unsigned int x1, unsigned int y1,
								unsigned int xNum, unsigned int yNum, unsigned int alignedY, bool enoughThreads)
{
	real_t* s_addends = (real_t*)shared_array;

	unsigned int row = blockIdx.x;//(blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int outIndex = enoughThreads? blockIdx.x : (blockIdx.y*gridDim.x) + blockIdx.x;
	unsigned int index = ((row+x1)*alignedY) + col + y1;

	if(row<xNum)
	{
		s_addends[threadIdx.x] = (col<yNum)? d_u[index] : 0;

		reduceToSum<real_t,threadNum>(s_addends, threadIdx.x);

		if(threadIdx.x == 0)
			d_output[outIndex] = s_addends[0];
	}
}

__global__ void d_complexSubtract(const complex_t* a, const complex_t* b, complex_t* result, unsigned int y)
{
	unsigned int index = (((blockIdx.x*blockDim.y)+threadIdx.y) * blockDim.x) + threadIdx.x;
	if(threadIdx.x < y)
		result[index] = sub_complex_t(a[index], b[index]);
}

template<bool enoughThreads>
__global__ void d_complexMultiply(const complex_t* a, const complex_t* b, complex_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY, real_t c,
								unsigned int axOffset, unsigned int ayOffset, unsigned int bxOffset,
								unsigned int byOffset)
{
	unsigned int aRow = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int aCol = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	unsigned int bRow = aRow + bxOffset;
	unsigned int bCol = aCol + byOffset;
	unsigned int bIndex = (bRow * alignedY) + bCol;

	aRow += axOffset;
	aCol += ayOffset;
	unsigned int aIndex = (aRow * alignedY) + aCol;

	if(max(aRow,bRow)<x && max(aCol,bCol)<y)
	{
		complex_t temp = mul_complex_t(a[aIndex], b[bIndex]);
		result[bIndex] = mul_complex_t(temp, make_complex_t(c, 0));
	}
}

template<bool enoughThreads>
__global__ void d_complexMultiply(const complex_t* a, complex_t b, complex_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY, real_t c)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		complex_t temp = mul_complex_t(a[index], b);
		result[index] = mul_complex_t(temp, make_complex_t(c, 0));
	}
}

template<bool enoughThreads>
__global__ void d_complexMultiply(const complex_t* a, real_t c, complex_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
		result[index] = mul_complex_t(a[index], make_complex_t(c, 0.0));
}

template<bool enoughThreads>
__global__ void d_realToRGBA(const real_t* a, real_t c, float4* result, unsigned int X, unsigned int Y, unsigned int alignedY, float transferOffset, float transferScale)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;
	unsigned int oIndex = (row * Y) + col;

	if(row<X && col<Y)
	{
		float normalizedV = (float) (a[index]*c);
		result[oIndex] = tex1D(g_transferTex, (normalizedV-transferOffset)*transferScale);
	}
}

template<bool enoughThreads>
__global__ void d_realToGray(const real_t* a, real_t c, float* result, unsigned int X, unsigned int Y, unsigned int alignedY, bool outAligned)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;
	unsigned int oIndex = (row * Y) + col;

	if(row<X && col<Y)
	{
		float normalizedV = (float) (a[index]*c);
		result[outAligned?index:oIndex] = normalizedV;
	}
}

template<unsigned char op, bool enoughThreads>
__global__ void d_complexToDouble(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		real_t temp = 0;
		switch(op)
		{
		case 'a': temp = abs_complex_t(a[index]); break;
		case 'p': temp = atan2_real_t(a[index].y, a[index].x); break;
		case 'r': temp = real_complex_t(a[index]); break;
		case 'i': temp = imag_complex_t(a[index]); break;
		default:  temp = 0; break;
		}
		result[index] = squared? temp*temp: temp;
	}
}

__device__ bool letFloat(const real_t* beamstopMask, unsigned int index, const real_t saturationValue, const real_t diffValue)
{
	bool toFloat = beamstopMask? beamstopMask[index]<0.99:false;
	toFloat = toFloat || (diffValue-saturationValue)>=0;
	return toFloat;
}

__device__ complex_t modulusConstraint(complex_t psi, real_t det_mod)
{
	real_t sinFunc, cosFunc;
	sincos_real_t(atan2_real_t(psi.y,psi.x),&sinFunc,&cosFunc);
	return make_complex_t(det_mod*cosFunc, det_mod*sinFunc);
}

__global__ void d_adjustFFT(const complex_t* d_psi, complex_t* d_output, const real_t* d_det_mod, const real_t* d_mask,
							const real_t saturationValue, unsigned int y, real_t normalizeBy)
{
	unsigned int row = (blockIdx.x*blockDim.y)+threadIdx.y;
	unsigned int col = threadIdx.x;
	unsigned int psiIndex = (row * blockDim.x) + col;
	if(col < y)
	{
		complex_t psi = d_psi[psiIndex];
		real_t diffValue = d_det_mod[psiIndex];
		if(diffValue>=saturationValue)
		{
			printf("diffValue is %f, saturationValue is %f \n", diffValue, saturationValue);
			printf(" row is %u, column is %u, complex_t psi x is %f, psi y is %f \n", row, col, psi.x, psi.y);
		}

		bool toFloat = letFloat(d_mask,psiIndex,saturationValue, diffValue);
//		d_output[psiIndex] = toFloat?psi:mul_complex_t(modulusConstraint(psi, diffValue), make_complex_t(normalizeBy, 0.0));
		d_output[psiIndex] = mul_complex_t(toFloat?psi:modulusConstraint(psi, diffValue), make_complex_t(normalizeBy, 0.0));
	}
}

__global__ void d_adjustModalFFT(const complex_t* d_psi, complex_t* d_output, const real_t* d_det_mod, const real_t* d_mask,
								const real_t saturationValue, unsigned int modeNum, unsigned int x, unsigned int y, real_t normalizeBy)
{
	unsigned int modeIndex = (blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int detIndex = (modeIndex*blockDim.x) + threadIdx.x;

	if(modeIndex<x && threadIdx.x<y)
	{
		real_t modalSum = 0, avdata = d_det_mod[detIndex];

		for(unsigned int i=0; i<modeNum; ++i)
		{
			unsigned int psiIndex = ((modeIndex+(i*x))*blockDim.x) + threadIdx.x;
			real_t psiFFtAbs = abs_complex_t(d_psi[psiIndex]);
			modalSum += psiFFtAbs * psiFFtAbs;
		}

		modalSum = rsqrt_real_t(modalSum);

		for(unsigned int i=0; i<modeNum; ++i)
		{
			unsigned int psiIndex = ((modeIndex+(i*x))*blockDim.x) + threadIdx.x;
			bool toFloat = letFloat(d_mask, detIndex, saturationValue, avdata);
//			d_output[psiIndex] = toFloat?d_psi[psiIndex]:mul_complex_t(d_psi[psiIndex], make_complex_t(avdata*modalSum*normalizeBy, 0.0));
			d_output[psiIndex] = mul_complex_t(d_psi[psiIndex], make_complex_t((toFloat?1:avdata)*modalSum*normalizeBy, 0.0));
		}
	}
}

template<unsigned int threadNum>
__global__ void d_calculateER(const complex_t* d_psi, const real_t* d_detMod, real_t* d_output,
								unsigned int x, unsigned int y, unsigned int alignedY, unsigned int modeNum, bool enoughThreads)
{
	real_t* s_addends = (real_t*)shared_array;

	unsigned int row = blockIdx.x;//(blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int outIndex = enoughThreads? blockIdx.x : (blockIdx.y*gridDim.x) + blockIdx.x;
	unsigned int index = (row*alignedY) + col;

	if(row<x)
	{
		complex_t modalSum = make_complex_t(0,0);

		for(unsigned int i=0; i<modeNum; ++i)
		{
			unsigned int psiIndex = ((row+(i*x))*alignedY) + col;
			modalSum = add_complex_t(modalSum, d_psi[psiIndex]);
		}

		s_addends[threadIdx.x] = (col<y)? abs_real_t( d_detMod[index] - abs_complex_t(modalSum) ) : 0;

		reduceToSum<real_t, threadNum>(s_addends, threadIdx.x);

		if(threadIdx.x == 0)
			d_output[outIndex] = s_addends[0];
	}
}

template<bool enoughThreads>
__global__ void d_realSpaceER(const complex_t* d_GT, const complex_t* d_obj, real_t* d_output,
								unsigned int qx,  unsigned int qy,
								unsigned int outX, unsigned int outY,
								unsigned int x1, unsigned int y1, unsigned int alignedY1,
								unsigned int x2, unsigned int y2, unsigned int alignedY2)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	unsigned int outIndex = (row*outY) + col;
	unsigned int gtIndex = ((row+qx)*alignedY1) + col + qy;
	unsigned int objIndex = ((row+qx)*alignedY2) + col + qy;

	if(row<outX && col<outY)
	{
		complex_t gtVal = d_GT[gtIndex];
		complex_t objVal = d_obj[objIndex];
		real_t diff = abs_complex_t(gtVal) - abs_complex_t(objVal);
		d_output[outIndex] = diff*diff;
	}
}

template<typename T, bool enoughThreads>
__global__ void d_shiftY(const T* d_objectArray, T* d_output, float nx,
	unsigned int offset, unsigned int X, unsigned int Y, unsigned int alignedY)
{
	unsigned int xIndex = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int yIndex = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	unsigned int objectArrayIndex = (xIndex * alignedY) + yIndex;
	T saved = d_objectArray[objectArrayIndex];

	if(xIndex<X && yIndex<Y)
	{
		int offsetY = yIndex - (roundf(nx) - offset);
		if(offsetY < 0) offsetY += Y;
		if(offsetY >= Y) offsetY -= Y;
		offsetY += (xIndex * alignedY);

		__syncthreads();

		d_output[offsetY] = saved;
	}
}

template<typename T, bool enoughThreads>
__global__ void d_shiftX(const T* d_objectArray, T* d_output, float ny,
	unsigned int offset, unsigned int X, unsigned int Y, unsigned int alignedY)
{
	unsigned int xIndex = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int yIndex = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;

	if(xIndex<X && yIndex<Y)
	{
		unsigned int objectArrayIndex = (xIndex * alignedY) + yIndex;
		T saved = d_objectArray[objectArrayIndex];

		int offsetX = xIndex - (roundf(ny) - offset);
		if(offsetX < 0) offsetX += X;
		if(offsetX >= X) offsetX -= X;
		offsetX = (offsetX * alignedY) + yIndex;

		__syncthreads();

		d_output[offsetX] = saved;
	}
}

template<typename T>
__global__ void d_mirrorY(const T* d_objectArray, T* d_output, unsigned int objectArrayY)
{
	unsigned int objectArrayIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	T saved = d_objectArray[objectArrayIndex];

	if(threadIdx.x < objectArrayY)
	{
		unsigned int mirrorIndex = (blockIdx.x * blockDim.x) + (objectArrayY-threadIdx.x);
		d_output[--mirrorIndex] = saved;
	}
}

template<typename T>
__global__ void d_rot90(const T* src, T* dst, unsigned int rows, unsigned int cols, unsigned int pitch)
{
	unsigned int row = (blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int col = threadIdx.x;
	unsigned int tid = row * pitch + col;
	unsigned int tid_out = (rows-col-1) * pitch + row;

	//saved[threadIdx.x*blockDim.y+threadIdx.y] = srcDst[tid];
	if(row<rows && col<cols)
		dst[tid_out] = src[tid];//saved[threadIdx.x*blockDim.y+threadIdx.y];
}

template<unsigned int threadNum>
__global__ void d_innerProduct(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
								real_t oneOverN, unsigned int y)
{
	complex_t* s_addends = (complex_t*)shared_array;
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	complex_t value = (threadIdx.x<y)?mul_complex_t( conj_complex_t(d_u[probeIndex]), d_v[probeIndex]): make_complex_t(0,0);
	s_addends[threadIdx.x] = make_complex_t(value.x*oneOverN,value.y*oneOverN);
	
	reduceToSumComplex<threadNum>(s_addends,threadIdx.x);	

	if(threadIdx.x == 0)
		d_output[blockIdx.x] = s_addends[0];
}

template<typename T>
__global__ void d_modalSum(const T* d_modes, T* d_output, unsigned int modeNum, unsigned int x, unsigned int y, bool sqaureRoot)
{
	//unsigned int baseIndex = (blockIdx.x * probeX) + ((blockIdx.y*blockDim.y) + threadIdx.y);
	unsigned int modeIndex = (blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int outIndex = (modeIndex*blockDim.x) + threadIdx.x;

	if(threadIdx.x < y)
	{
		T val = d_modes[outIndex];
		for(unsigned int i=1; i<modeNum; ++i)
			val += d_modes[((modeIndex+(i*x))*blockDim.x) + threadIdx.x];

		d_output[outIndex] = sqaureRoot? sqrt_real_t(val) : val;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__host__ int getReductionThreadNum(int size) {return (int) rint( pow(2.0f, (int)ceil( log2( (float) size) ) ) );}

__host__ void h_initColorTransferTexture()
{
	// create transfer function texture
    float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1);
	cutilCheckMsg("h_initColorTransferTexture() cudaMallocArray execution failed!\n");
    cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice);
	cutilCheckMsg("h_initColorTransferTexture() cudaMemcpyToArray execution failed!\n");

    g_transferTex.filterMode = cudaFilterModeLinear;
    g_transferTex.normalized = true;    // access with normalized texture coordinates
    g_transferTex.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates

    // Bind the array to the texture
    cudaBindTextureToArray(&g_transferTex, d_transferFuncArray, &channelDesc2);
	cutilCheckMsg("h_initColorTransferTexture() cudaBindTextureToArray execution failed!\n");
}

__host__ void h_freeColorTransferTexture()
{
	if(d_transferFuncArray)
	{
		cudaUnbindTexture(&g_transferTex);
		cutilCheckMsg("h_freeColorTransferTexture()::cudaUnbindTexture() execution failed!\n");
		cudaFreeArray(d_transferFuncArray);
		cutilCheckMsg("h_init3DTexture()::cudaFreeArray() execution failed!\n");
	}
}

template<typename T>
void h_reduceToSum(const T* a, thrust::device_vector<T>& out, unsigned int x1, unsigned int x2, unsigned int y1, unsigned int y2, unsigned int alignedY)
{
	unsigned int xNum = x2-x1;
	unsigned int yNum = y2-y1;

	unsigned int maxThreads =  GPUQuery::getInstance()->getGPUMaxThreads();
	unsigned int reductionThreads = getReductionThreadNum(yNum);
	dim3 grid;
	dim3 block;
	bool enoughThreads = true;
	if(reductionThreads<=maxThreads)
	{
		grid = dim3(xNum, 1, 1);
		block = dim3(reductionThreads, 1, 1);
		out.resize(xNum);
	}
	else
	{
		enoughThreads = false;
		unsigned int sliceNum = gh_iDivUp(reductionThreads, maxThreads);
		grid = dim3(xNum, sliceNum, 1);
		block = dim3(maxThreads, 1, 1);
		out.resize(xNum*sliceNum);
	}
	unsigned int threadNum = block.x * block.y;
	size_t shared_mem_size = (threadNum <= 32) ? 2* threadNum * sizeof(T) : threadNum *  sizeof(T);

	switch (threadNum)
	{
	case   8:	d_reduceToSum<   8><<<grid, block, shared_mem_size>>>(a, thrust::raw_pointer_cast(out.data()), x1, y1, xNum, yNum, alignedY, enoughThreads);
	break;
	case  16:	d_reduceToSum<  16><<<grid, block, shared_mem_size>>>(a, thrust::raw_pointer_cast(out.data()), x1, y1, xNum, yNum, alignedY, enoughThreads);
	break;
	case  32:	d_reduceToSum<  32><<<grid, block, shared_mem_size>>>(a, thrust::raw_pointer_cast(out.data()), x1, y1, xNum, yNum, alignedY, enoughThreads);
	break;
	case  64:	d_reduceToSum<  64><<<grid, block, shared_mem_size>>>(a, thrust::raw_pointer_cast(out.data()), x1, y1, xNum, yNum, alignedY, enoughThreads);
	break;
	case 128:	d_reduceToSum< 128><<<grid, block, shared_mem_size>>>(a, thrust::raw_pointer_cast(out.data()), x1, y1, xNum, yNum, alignedY, enoughThreads);
	break;
	case 256:	d_reduceToSum< 256><<<grid, block, shared_mem_size>>>(a, thrust::raw_pointer_cast(out.data()), x1, y1, xNum, yNum, alignedY, enoughThreads);
	break;
	case 512:	d_reduceToSum< 512><<<grid, block, shared_mem_size>>>(a, thrust::raw_pointer_cast(out.data()), x1, y1, xNum, yNum, alignedY, enoughThreads);
	break;
	case 1024:	d_reduceToSum<1024><<<grid, block, shared_mem_size>>>(a, thrust::raw_pointer_cast(out.data()), x1, y1, xNum, yNum, alignedY, enoughThreads);
	break;
	}
	cutilCheckMsg("d_reduceToSum() execution failed!\n");
}

__host__ complex_t h_complexSum(const complex_t* a, unsigned int x1, unsigned int x2, unsigned int y1, unsigned int y2, unsigned int alignedY)
{
	thrust::device_vector<complex_t> output;
	h_reduceToSum<complex_t>(a, output, x1, x2, y1, y2, alignedY);
	return thrust::reduce(output.begin(), output.end(), make_complex_t(0,0), complexSum());
}

__host__ real_t h_realSum(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY)
{
	thrust::device_ptr<real_t> devPtr_a = thrust::device_pointer_cast(a);
	return thrust::reduce(devPtr_a, devPtr_a+(x*alignedY));
}

__host__ real_t h_realSum(const real_t* a, unsigned int x1, unsigned int x2, unsigned int y1, unsigned int y2, unsigned int alignedY)
{
	thrust::device_vector<real_t> output;
	h_reduceToSum<real_t>(a, output, x1, x2, y1, y2, alignedY);
	return thrust::reduce(output.begin(), output.end());
}

__host__ float2 h_maxFloat2(float2* a, unsigned int x, unsigned int y, unsigned int alignedY)
{
	thrust::device_ptr<float2> devPtr_a = thrust::device_pointer_cast(a);
	return thrust::reduce(devPtr_a, devPtr_a+(x*alignedY), make_float2(FLT_MIN,FLT_MIN), maxFloat2());
}

__host__ float2 h_minFloat2(float2* a, unsigned int x, unsigned int y, unsigned int alignedY)
{
	thrust::device_ptr<float2> devPtr_a = thrust::device_pointer_cast(a);
	return thrust::reduce(devPtr_a, devPtr_a+(x*alignedY), make_float2(FLT_MAX,FLT_MAX), minFloat2());
}

__host__ void h_subtract(const complex_t* a, const complex_t* b, complex_t* result,
                                        unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	d_complexSubtract<<<grid, block>>>(a, b, result, y);
	cutilCheckMsg("d_complexSubtract() execution failed!\n");
}

__host__ void h_multiply(const complex_t* a, const complex_t* b, complex_t* result,	unsigned int x, unsigned int y, unsigned int alignedY,
						bool normalize, unsigned int axOffset, unsigned int ayOffset, unsigned int bxOffset, unsigned int byOffset)
{
	unsigned int maxThreads = GPUQuery::getInstance()->getGPUMaxThreads();
	unsigned int blockOffset = max(axOffset,bxOffset);
	if(blockOffset<x && max(ayOffset,byOffset)<y)
	{
		if (alignedY <= maxThreads)
		{
			unsigned int sliceNum = gh_iDivDown(maxThreads, alignedY);
			dim3 grid(gh_iDivUp(x-blockOffset, sliceNum), 1, 1);
			dim3 block(alignedY, sliceNum, 1);
			d_complexMultiply<true><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1,
														axOffset, ayOffset, bxOffset, byOffset);
		}
		else
		{
			unsigned int sliceNum = gh_iDivUp(alignedY, maxThreads);
			dim3 grid(x-blockOffset, sliceNum, 1);
			dim3 block(maxThreads, 1, 1);
			d_complexMultiply<false><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1,
														axOffset, ayOffset, bxOffset, byOffset);
		}
	}
	cutilCheckMsg("d_complexMultiply() execution failed!\n");
}

__host__ void h_multiply(const complex_t* a, const complex_t& b, complex_t* result,
	unsigned int x, unsigned int y, unsigned int alignedY, bool normalize)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_complexMultiply<true> <<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1);
	else 				d_complexMultiply<false><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1);
	cutilCheckMsg("d_complexMultiply() execution failed!\n");
}
__host__ void h_normalize(complex_t* a, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_complexMultiply<true> <<<grid, block>>>(a, factor, a, x, y, alignedY);
	else				d_complexMultiply<false><<<grid, block>>>(a, factor, a, x, y, alignedY);
	cutilCheckMsg("d_complexMultiply() execution failed\n");
}

__host__ void h_normalize(const complex_t* a, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_complexMultiply<true> <<<grid, block>>>(a, factor, result, x, y, alignedY);
	else				d_complexMultiply<false><<<grid, block>>>(a, factor, result, x, y, alignedY);
	cutilCheckMsg("d_complexMultiply() execution failed\n");
}

__host__ real_t h_realMax(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY)
{
	thrust::device_ptr<real_t> devPtr_a = thrust::device_pointer_cast(a);
	return thrust::reduce(devPtr_a, devPtr_a+(x*alignedY), DBL_MIN, thrust::maximum<real_t>() );
}

__host__ void h_realToRGBA(const real_t* d_arr, float4* d_output, unsigned int x, unsigned int y, unsigned int alignedY, 
							real_t factor, float tf, float ts)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_realToRGBA<true> <<<grid, block>>>(d_arr, factor, d_output, x, y, alignedY, tf, ts);
	else				d_realToRGBA<false><<<grid, block>>>(d_arr, factor, d_output, x, y, alignedY, tf, ts);
	cutilCheckMsg("h_realToRGBA() execution failed\n");
}

__host__ void h_realToGray(const real_t* d_arr, float* d_output, unsigned int x, unsigned int y, unsigned int alignedY, 
							real_t factor, bool outAligned)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_realToGray<true> <<<grid, block>>>(d_arr, factor, d_output, x, y, alignedY, outAligned);
	else 				d_realToGray<false><<<grid, block>>>(d_arr, factor, d_output, x, y, alignedY, outAligned);
	cutilCheckMsg("h_realToGray() execution failed\n");
}

__host__ void h_normalize(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY)
{
	thrust::device_ptr<real_t> devPtr_a = thrust::device_pointer_cast(a);
	thrust::constant_iterator<real_t> maxValue(h_realMax(a,x,y,alignedY));
	thrust::transform(devPtr_a, devPtr_a+(x*alignedY), maxValue, devPtr_a, thrust::divides<real_t>());
	cutilCheckMsg("h_normalize() execution failed\n");
}

__host__ void h_realComplexAbs(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_complexToDouble<'a', true> <<<grid, block>>>(a, result, x, y, alignedY, squared);
	else				d_complexToDouble<'a', false><<<grid, block>>>(a, result, x, y, alignedY, squared);
	cutilCheckMsg("h_realComplexReal() execution failed\n");
}

__host__ void h_realComplexPhase(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_complexToDouble<'p', true> <<<grid, block>>>(a, result, x, y, alignedY, squared);
	else				d_complexToDouble<'p', false><<<grid, block>>>(a, result, x, y, alignedY, squared);
	cutilCheckMsg("h_realComplexPhase() execution failed\n");
}

__host__ void h_realComplexReal(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_complexToDouble<'r', true> <<<grid, block>>>(a, result, x, y, alignedY, squared);
	else				d_complexToDouble<'r', false><<<grid, block>>>(a, result, x, y, alignedY, squared);
	cutilCheckMsg("h_realComplexReal() execution failed\n");
}

__host__ void h_realComplexImag(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_complexToDouble<'i', true> <<<grid, block>>>(a, result, x, y, alignedY, squared);
	else				d_complexToDouble<'i', false><<<grid, block>>>(a, result, x, y, alignedY, squared);
	cutilCheckMsg("h_realComplexImag() execution failed\n");
}

__host__ void h_adjustFFT(const complex_t* d_psi, complex_t* d_output, const real_t* d_det_mod, const real_t* d_mask,
						const real_t saturationValue,  unsigned int modeNum, unsigned int x, unsigned int y, unsigned int alignedY, bool normalize)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	if(modeNum>1)
		d_adjustModalFFT<<<grid, block>>>(d_psi, d_output, d_det_mod, d_mask, saturationValue, modeNum, x, y, normalize?1.0/(real_t)(x*y):1);
	else
		d_adjustFFT<<<grid, block>>>(d_psi, d_output, d_det_mod, d_mask, saturationValue, y, normalize?1.0/(real_t)(x*y):1);
	cutilCheckMsg("h_adjustFFT() execution failed!\n");
}

__host__ real_t h_calculateER(const complex_t* d_psi, const real_t* d_det_mod, unsigned int modeNum,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	thrust::device_vector<real_t> output;

	unsigned int maxThreads =  GPUQuery::getInstance()->getGPUMaxThreads();
	unsigned int reductionThreads = getReductionThreadNum(y);
	dim3 grid;
	dim3 block;
	bool enoughThreads = true;
	if(reductionThreads<=maxThreads)
	{
		grid = dim3(x, 1, 1);
		block = dim3(reductionThreads, 1, 1);
		output.resize(x);
	}
	else
	{
		enoughThreads = false;
		unsigned int sliceNum = gh_iDivUp(reductionThreads, maxThreads);
		grid = dim3(x, sliceNum, 1);
		block = dim3(maxThreads, 1, 1);
		output.resize(x*sliceNum);
	}
	unsigned int threadNum = block.x * block.y;
	size_t shared_mem_size = (threadNum <= 32) ? 2* threadNum * sizeof(real_t) : threadNum *  sizeof(real_t);

	switch (threadNum)
	{
	case   8:	d_calculateER<   8><<<grid, block, shared_mem_size>>>(d_psi, d_det_mod, thrust::raw_pointer_cast(output.data()), x, y, alignedY, modeNum, enoughThreads);
	break;
	case  16:	d_calculateER<  16><<<grid, block, shared_mem_size>>>(d_psi, d_det_mod, thrust::raw_pointer_cast(output.data()), x, y, alignedY, modeNum, enoughThreads);
	break;
	case  32:	d_calculateER<  32><<<grid, block, shared_mem_size>>>(d_psi, d_det_mod, thrust::raw_pointer_cast(output.data()), x, y, alignedY, modeNum, enoughThreads);
	break;
	case  64:	d_calculateER<  64><<<grid, block, shared_mem_size>>>(d_psi, d_det_mod, thrust::raw_pointer_cast(output.data()), x, y, alignedY, modeNum, enoughThreads);
	break;
	case 128:	d_calculateER< 128><<<grid, block, shared_mem_size>>>(d_psi, d_det_mod, thrust::raw_pointer_cast(output.data()), x, y, alignedY, modeNum, enoughThreads);
	break;
	case 256:	d_calculateER< 256><<<grid, block, shared_mem_size>>>(d_psi, d_det_mod, thrust::raw_pointer_cast(output.data()), x, y, alignedY, modeNum, enoughThreads);
	break;
	case 512:	d_calculateER< 512><<<grid, block, shared_mem_size>>>(d_psi, d_det_mod, thrust::raw_pointer_cast(output.data()), x, y, alignedY, modeNum, enoughThreads);
	break;
	case 1024:	d_calculateER<1024><<<grid, block, shared_mem_size>>>(d_psi, d_det_mod, thrust::raw_pointer_cast(output.data()), x, y, alignedY, modeNum, enoughThreads);
	break;
	}
	cutilCheckMsg("h_calculateER() execution failed!\n");

	return thrust::reduce(output.begin(), output.end())/modeNum;
}

__host__ real_t h_calculateER(const complex_t* d_GT, const complex_t* d_obj,
								unsigned int sx, unsigned int sy, unsigned int qx, unsigned int qy,
								unsigned int x1, unsigned int y1, unsigned int alignedY1,
								unsigned int x2, unsigned int y2, unsigned int alignedY2)
{
	thrust::device_vector<real_t> output(sx*sy);

	dim3 grid, block;
	bool enoughThreads = calcGrids(sx,sy,grid,block);

	if(enoughThreads)	d_realSpaceER<true> <<<grid, block>>>(	d_GT, d_obj, thrust::raw_pointer_cast(output.data()),
																qx, qy, sx, sy, x1, y1, alignedY1, x2, y2, alignedY2);
	else				d_realSpaceER<false><<<grid, block>>>(	d_GT, d_obj, thrust::raw_pointer_cast(output.data()),
																qx, qy, sx, sy, x1, y1, alignedY1, x2, y2, alignedY2);
	cutilCheckMsg("d_realSpaceER() execution failed\n");

	return sqrt(thrust::reduce(output.begin(), output.end()))/output.size();
}

__host__ void h_shiftFFT(real_t* d_data, real_t* d_temp, unsigned int x, unsigned int y, unsigned int alignedY, cudaStream_t* stream)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);

	d_shiftX<real_t, true><<<grid, block,0,(stream?*stream:0)>>>(d_data, d_temp, (float)x/2.0, 0, x, y, alignedY);
	cutilCheckMsg("h_shiftFFT() execution failed!\n");

	cudaDeviceSynchronize();

	d_shiftY<real_t, true><<<grid, block,0,(stream?*stream:0)>>>(d_temp, d_data, (float)y/2.0, 0, x, y, alignedY);
	cutilCheckMsg("h_shiftFFT() execution failed!\n");
}

__host__ void h_realRotate90(const real_t* d_data, real_t* d_out, unsigned int x, unsigned int y, unsigned int alignedY, unsigned int times, cudaStream_t* stream)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	switch(times % 4)
	{
	case 0: break;
	case 1:
		d_rot90<real_t><<<grid,block,0,(stream?*stream:0)>>>(d_data, d_out, x, y, alignedY);
		break;
	case 2:
		//d_mirrorY<real_t><<<x, alignedY, alignedY*sizeof(real_t)>>>(d_data, d_data, y);
		break;
	case 3:
		break;
	}
	cutilCheckMsg("h_realRotate90() execution failed!\n");
}

__host__ complex_t h_innerProduct(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int reductionThreads = getReductionThreadNum(alignedY);
	dim3 grid(x, 1, 1);
	dim3 block(reductionThreads, 1, 1);
	size_t shared_mem_size = (block.x <= 32) ? 2* block.x * sizeof(complex_t) : block.x *  sizeof(complex_t);

	switch (block.x)
	{
	case   8:	d_innerProduct<   8><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0/(real_t)(x*y), y);
	break;
	case  16:	d_innerProduct<  16><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0/(real_t)(x*y), y);
	break;
	case  32:	d_innerProduct<  32><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0/(real_t)(x*y), y);
	break;
	case  64:	d_innerProduct<  64><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0/(real_t)(x*y), y);
	break;
	case 128:	d_innerProduct< 128><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0/(real_t)(x*y), y);
	break;
	case 256:	d_innerProduct< 256><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0/(real_t)(x*y), y);
	break;
	case 512:	d_innerProduct< 512><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0/(real_t)(x*y), y);
	break;
	case 1024:	d_innerProduct<1024><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0/(real_t)(x*y), y);
	break;
	}
	cutilCheckMsg("d_innerProduct() execution failed!\n");

	thrust::device_ptr<complex_t> devPtr = thrust::device_pointer_cast(d_output);
	return thrust::reduce(devPtr, devPtr+x, make_complex_t(0,0), complexSum());
}

__host__ void h_realModalSum(const real_t* d_modes, real_t* d_output, unsigned int modesNum,
								unsigned int x, unsigned int y, unsigned int alignedY, bool sqaureRoot)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x,sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	
	d_modalSum<real_t><<<grid, block>>>(d_modes, d_output, modesNum, x, y, sqaureRoot);
	cutilCheckMsg("d_modalSum() execution failed!\n");
}

__host__ int2 h_realArgMax2D(real_t* d_ncc, unsigned int x, unsigned int y, unsigned int alignedY, unsigned char dir)
{
	thrust::device_ptr<real_t> ncc_wrapper = thrust::device_pointer_cast(d_ncc);
	int maxIndex = thrust::max_element(ncc_wrapper, ncc_wrapper+(x*alignedY)) - ncc_wrapper;
	cutilCheckMsg("h_realArgMax2D():thrust::max_element() execution failed!\n");

	int2 peak;
	peak.x = maxIndex / alignedY;
	peak.y = maxIndex % alignedY;

	peak.x = (dir == 'h' && (peak.x >= (x/2)))? peak.x - x: peak.x;
	peak.y = (dir == 'v' && (peak.y >= (y/2)))? peak.y - y: peak.y;

	//printf("Registration point (%d,%d)...\n", peak.x, peak.y);
	return peak;
}

__host__ void h_realComplexModulate(const complex_t* d_array1, complex_t* d_array2, int2& peak,
										unsigned int x, unsigned int y, unsigned int alignedY, unsigned char dir)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)
	{
		if(dir == 'h' && peak.x!=0)
		{
			d_shiftX<complex_t, true><<<grid, block>>>(d_array2, d_array2, -(float)peak.x, 0, x, y, alignedY);
			cutilCheckMsg("h_hMatchArrays()::shiftX() execution failed!\n");
			peak.x = 0;
		}
		else if(dir == 'v' && peak.y!=0)
		{
			d_shiftY<complex_t, true><<<grid, block>>>(d_array2, d_array2, -(float)peak.y, 0, x, y, alignedY);
			cutilCheckMsg("h_vMatchArrays()::shiftY() execution failed!\n");
			peak.y=0;
		}
	}
	else
	{
		if(dir == 'h' && peak.x!=0)
		{
			d_shiftX<complex_t, false><<<grid, block>>>(d_array2, d_array2, -(float)peak.x, 0, x, y, alignedY);
			cutilCheckMsg("h_hMatchArrays()::shiftX() execution failed!\n");
			peak.x = 0;
		}
		else if(dir == 'v' && peak.y!=0)
		{
			d_shiftY<complex_t, false><<<grid, block>>>(d_array2, d_array2, -(float)peak.y, 0, x, y, alignedY);
			cutilCheckMsg("h_vMatchArrays()::shiftY() execution failed!\n");
			peak.y=0;
		}

	}

	complex_t m1 = h_complexSum(d_array1, peak.x, x, peak.y, y, alignedY);
	complex_t m2 = h_complexSum(d_array2, 0, x-peak.x, 0, y-peak.y, alignedY);
	complex_t ratio = div_complex_t(m1,m2);
	h_multiply(d_array2, ratio, d_array2, x, y, alignedY, false);
}

#endif /* UTILITIESKERNELS_CU_ */