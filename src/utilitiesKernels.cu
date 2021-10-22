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
//Any publication using the package should cite for
//Yue K, Deng J, Jiang Y, Nashed Y, Vine D, Vogt S.
//Ptychopy: GPU framework for ptychographic data analysis.
//X-Ray Nanoimaging: Instruments and Methods V 2021 .
//International Society for Optics and Photonics.
//
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

#define EPS 1e-3

//#include <cub/device/device_reduce.cuh>
//using namespace cub;

/* extern shared memory for dynamic allocation */
extern __shared__ real_t shared_array[];

// ~10800 is the maximum of const memory
const unsigned int MAX_IND_READ = 3000;
//__constant__ unsigned int gC_ind_read[MAX_IND_READ];
__constant__ unsigned int gC_pos_X[MAX_IND_READ];
__constant__ unsigned int gC_pos_Y[MAX_IND_READ];

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

//__global__ void d_check(real_t* d_data)
//{
//
//	unsigned int Index = (blockIdx.x * blockDim.x) + threadIdx.x;
//
////	real_t temp=d_data[Index];
////	unsigned int sq1=1;
//}

//__global__ void d_checkcomplex(complex_t* d_data)
//{
//
//	unsigned int Index = (blockIdx.x * blockDim.x) + threadIdx.x;
//
////	complex_t temp=d_data[Index];
////	unsigned int sq1=1;
//}

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

__global__ void d_subtract(const real_t* a, const real_t* b, real_t* result, unsigned int y)
{
	unsigned int index = (((blockIdx.x*blockDim.y)+threadIdx.y) * blockDim.x) + threadIdx.x;
	if(threadIdx.x < y)
		result[index] = a[index]-b[index];
}

__global__ void d_addFactorDivide(real_t* a, real_t* result, real_t factor, unsigned int y)
{
	unsigned int index = (((blockIdx.x*blockDim.y)+threadIdx.y) * blockDim.x) + threadIdx.x;
	if(threadIdx.x < y)
	{
		real_t tmp=a[index];
		result[index]=tmp/(tmp+factor);
	}
}

__global__ void d_object_sum_update_Gfun(complex_t* a, real_t* b, complex_t* result, real_t factor, unsigned int y)
{
	unsigned int index = (((blockIdx.x*blockDim.y)+threadIdx.y) * blockDim.x) + threadIdx.x;
	if(threadIdx.x < y)
	{
		complex_t tmp= make_complex_t((b[index]+factor), 0);
		result[index]=div_complex_t(a[index], tmp);
	}
}

__global__ void d_addFactor(complex_t* a, complex_t* result, complex_t factor, unsigned int y)
{
	unsigned int index = (((blockIdx.x*blockDim.y)+threadIdx.y) * blockDim.x) + threadIdx.x;
	if(threadIdx.x < y)
	{
		result[index]=add_complex_t(a[index], factor);
	}
}

__global__ void d_addFactor(real_t* a, real_t* result, real_t factor, unsigned int y)
{
	unsigned int index = (((blockIdx.x*blockDim.y)+threadIdx.y) * blockDim.x) + threadIdx.x;
	if(threadIdx.x < y)
	{
		result[index]=a[index]+factor;
	}
}

template<bool enoughThreads>
__global__ void d_multiplyConju(const complex_t* a, const complex_t* b, complex_t* result,
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
		complex_t temp = mul_complex_t(a[aIndex], conj_complex_t(b[bIndex]));
		result[bIndex] = mul_complex_t(temp, make_complex_t(c, 0));
	}
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
__global__ void d_complexMultiply(const real_t* a, const complex_t* b, complex_t* result,
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
		result[bIndex] = mul_complex_t(make_complex_t(a[aIndex], 0), b[bIndex]);
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
__global__ void d_multiply(const real_t* a, real_t b, real_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY, real_t c)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		result[index] = a[index]*b;
	}
}

template<bool enoughThreads>
__global__ void d_mul_rca_mulc_rcr(complex_t* a, complex_t* b, complex_t* c, real_t* weight_proj,
		unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		complex_t temp1=mul_complex_t(a[index],b[index]);
		float sum2denom=abs_complex_t(temp1);

		complex_t temp3=mul_complex_t(c[index], conj_complex_t(temp1));
		float sum2nom=real_complex_t(temp3);
		weight_proj[index]=0.1*sum2nom/(sum2denom*sum2denom);
	}

}

// Only has one row of factor col from 1 to alignedy
template<bool enoughThreads>
__global__ void d_multiplyPage(const complex_t* a, complex_t* b, complex_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY, unsigned int pagex)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	// bcol=col
	unsigned int brow=row%pagex;
	unsigned int aindex = (row * alignedY) + col;
	unsigned int bindex = (brow * alignedY) + col;


	if(row<x && col<y)
	{
		result[aindex] = mul_complex_t(a[aindex], b[bindex]);
	}
}

template<bool enoughThreads>
__global__ void d_multiplyAbsConjuRealWhole(const complex_t* a, complex_t* b, complex_t* c, real_t* result1,
		real_t* result2, unsigned int x, unsigned int y, unsigned int alignedY,
		unsigned int pagex)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		complex_t img = mul_complex_t(a[index], b[index]);
		real_t temp = abs_complex_t(img);
		result1[index] = temp*temp;
		img = mul_complex_t(c[index], conj_complex_t(img));
		result2[index]=real_complex_t(img);
	}
}

template<bool enoughThreads>
__global__ void d_multiplyAbsConjuReal(const complex_t* a, complex_t* b, complex_t* c, real_t* result1,
		real_t* result2, unsigned int x, unsigned int y, unsigned int alignedY,
		unsigned int pagex)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	// bcol=col
	unsigned int brow=row%pagex;
	unsigned int aindex = (row * alignedY) + col;
	unsigned int bindex = (brow * alignedY) + col;


	if(row<x && col<y)
	{
		complex_t img = mul_complex_t(a[aindex], b[bindex]);
		real_t temp = abs_complex_t(img);
		result1[aindex] = temp*temp;

		img = mul_complex_t(c[aindex], conj_complex_t(img));
		result2[aindex]=real_complex_t(img);
	}
}



// Only has one row of factor col from 1 to alignedy
template<bool enoughThreads>
__global__ void d_multiplyRow(const complex_t* a, real_t* b, complex_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY, real_t c)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		complex_t temp = mul_complex_t(a[index], make_complex_t(b[col], 0));
		result[index] = temp;
	}
}
// the factor is from 0 to x;
template<bool enoughThreads>
__global__ void d_multiplyColum(const complex_t* a, real_t* b, complex_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY, real_t c)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		complex_t temp = mul_complex_t(a[index], make_complex_t(b[row], 0));
		result[index] = temp;
	}
}




//function [AA1,AA2,AA4, Atb1,Atb2] = ...
//            get_optimal_step_lsq(chi,dO,dP,O,P, lambda)
//     % fast kernel for estimation of optimal probe and object steps
//    dOP = dO.*P;
//    dPO = dP.*O;
//    cdOP = conj(dOP);
//    cdPO = conj(dPO);
//
//    AA1 = real(dOP .* cdOP)+lambda;
//    AA2 = (dOP .* cdPO);
//    AA4 = real(dPO .* cdPO)+lambda;
//    Atb1 = real(cdOP .* chi);
//    Atb2 = real(cdPO .* chi);
//end


template<bool enoughThreads>
__global__ void d_get_optimal_step_lsq(complex_t* chi, complex_t* object_update_proj, complex_t* dPO, complex_t* probe, real_t lambda,
		real_t* AA1, complex_t* AA2, real_t* AA4, real_t* Atb1, real_t* Atb2,
		unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		complex_t dOP=mul_complex_t(object_update_proj[index], probe[index]);
		complex_t cdOP=conj_complex_t(dOP);
		complex_t cdPO=conj_complex_t(dPO[index]);

		AA1[index]=real_complex_t(mul_complex_t(dOP, cdOP))+lambda;
		AA2[index]=mul_complex_t(dOP, cdPO);
		AA4[index] = real_complex_t(mul_complex_t(dPO[index], cdPO))+lambda;
		Atb1[index]=real_complex_t(mul_complex_t(cdOP, chi[index]));
		Atb2[index]=real_complex_t(mul_complex_t(cdPO, chi[index]));
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
	{
		complex_t temp=mul_complex_t(a[index], make_complex_t(c, 0.0));
		result[index] = temp;
	}

//	int temp=1;
}

template<bool enoughThreads>
__global__ void d_realMultiply(real_t* a, real_t* b, real_t* result,
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
		result[aIndex]=a[aIndex]*b[bIndex];
	}
}

template<bool enoughThreads>
__global__ void d_realMultiply(real_t* a, real_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		real_t temp=a[index]*result[index];
		result[index] = temp;
	}
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

template<bool enoughThreads>
__global__ void d_realComplexExp(const real_t* src, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;
	if(row<x && col<y)
	{
		complex_t temp=make_complex_t((cos_real_t(src[index]*factor)), (sin_real_t(src[index]*factor)));
		result[index]=temp;
	}
}


template<bool enoughThreads>
__global__ void d_realsquareRoot(real_t* d_arr, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
//		real_t temp = 0;
//		temp= sqrt_real_t(d_arr[index]);
		result[index]=sqrt_real_t(d_arr[index]);
	}
}

template<bool enoughThreads>
__global__ void d_square(real_t* d_arr, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(row<x && col<y)
	{
		real_t tmp=d_arr[index];
		result[index]=tmp*tmp;
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
//		if(diffValue>=saturationValue)
//		{
//			printf("diffValue is %f, saturationValue is %f \n", diffValue, saturationValue);
//			printf(" row is %u, column is %u, complex_t psi x is %f, psi y is %f \n", row, col, psi.x, psi.y);
//		}

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
__global__ void d_imshift_fft(T* d_data, unsigned int midx, unsigned int midy, float radNo1, float radNo2,
		unsigned int X, unsigned int Y, unsigned int alignedY)
{

	unsigned int xIndex = threadIdx.x;
	unsigned int yIndex = (blockIdx.x*blockDim.y) + threadIdx.y;

	if(xIndex<Y && yIndex<X)
	{
		unsigned int objectArrayIndex = (yIndex * alignedY) + xIndex;
		T saved = d_data[objectArrayIndex];

		float xgridindex=xIndex;
		float ygridindex=yIndex;

		if (xIndex < midx)
			xgridindex+=midx;
		else
			xgridindex-=midx;

		if (yIndex < midy)
			ygridindex+=midy;
		else
			ygridindex-=midy;

			xgridindex=radNo1*(xgridindex/X-0.5);
			ygridindex=radNo2*(ygridindex/Y-0.5);

			real_t sumInitx=2*CUDART_PI*xgridindex;
			real_t sumInity=2*CUDART_PI*ygridindex;
			real_t costx=cos_real_t(sumInitx);
			real_t sintx=-1*sin_real_t(sumInitx);
			real_t costy=cos_real_t(sumInity);
			real_t sinty=-1*sin_real_t(sumInity);
			complex_t tempmulx = make_complex_t(costx, sintx);
			complex_t tempmuly = make_complex_t(costy, sinty);

			d_data[objectArrayIndex]=mul_complex_t(saved,mul_complex_t(tempmulx, tempmuly));
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

template<unsigned int threadNum>
__global__ void d_innerProductOne(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
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




__global__ void d_innerProductModes(complex_t* d_u, complex_t* d_v, complex_t* d_factor,
		unsigned int index, unsigned int ModeNumber, unsigned int probeX, unsigned int probeY, unsigned int offset)
{
	unsigned int row = (blockIdx.x*blockDim.y) + threadIdx.y;
//	unsigned int modeIndex = ((row+((blockIdx.x) * probeX)) * blockDim.x) + threadIdx.x;
	unsigned int baseIndex = (row*blockDim.x) + threadIdx.x;

	if(row<probeX && threadIdx.x<probeY)
	{
		complex_t value=make_complex_t(0, 0);
		for(int i=0; i< ModeNumber; i++)
		{
			value = add_complex_t(value, mul_complex_t(d_u[baseIndex+offset*i], d_factor[index+i*ModeNumber]));
		}
//		complex_t value=add_complex_t(mul_complex_t(d_u[baseIndex], d_factor[index]), mul_complex_t(d_u[baseIndex+offset], d_factor[index+5]));
//		value = add_complex_t(value, mul_complex_t(d_u[baseIndex+offset*2], d_factor[index+2*5]));
//		value = add_complex_t(value, mul_complex_t(d_u[baseIndex+offset*3], d_factor[index+3*5]));
//		value = add_complex_t(value, mul_complex_t(d_u[baseIndex+offset*4], d_factor[index+4*5]));
		d_v[baseIndex]=value;
	}
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

__global__ void d_modalSumComplex(const complex_t* d_modes, complex_t* d_output, unsigned int modeNum, unsigned int x, unsigned int y, bool sqaureRoot)
{
	unsigned int modeIndex = (blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int outIndex = (modeIndex*blockDim.x) + threadIdx.x;

	if(threadIdx.x < y)
	{
		complex_t val = d_modes[outIndex];
		for(unsigned int i=1; i<modeNum; ++i)
			val=add_complex_t(val, d_modes[((modeIndex+(i*x))*blockDim.x) + threadIdx.x]);

		d_output[outIndex]=val;
	}
}

__global__ void d_complexSum(complex_t* d_leftArr, complex_t* d_rightArr, complex_t* d_result, real_t leftFactor, real_t rightFactor, unsigned int x, unsigned int y,
		unsigned int alignedY)
{
	unsigned int row = (blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int col = threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(threadIdx.x < y)
	{
		complex_t leftOp=mul_complex_t(d_leftArr[index], make_complex_t(leftFactor,0));
		complex_t rightOp=mul_complex_t(d_rightArr[index], make_complex_t(rightFactor,0));
		d_result[index]=add_complex_t(leftOp, rightOp);
	}
}

__global__ void d_realSum(real_t* d_leftArr, real_t* d_rightArr, real_t* d_result, real_t leftFactor, real_t rightFactor, unsigned int x, unsigned int y,
		unsigned int alignedY)
{
	unsigned int row = (blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int col = threadIdx.x;
	unsigned int index = (row * alignedY) + col;

	if(threadIdx.x < y)
	{
		d_result[index]=d_leftArr[index]*leftFactor+d_rightArr[index]*rightFactor;
	}
}

// The first row and the first column combine to a new matrix by adding the duplicated elments of each line
// 26*256 26*256
__global__ void d_realSingleSum(real_t* d_leftArr, real_t* d_rightArr, real_t* d_result, unsigned int x, unsigned int y,
		unsigned int alignedY)
{
	unsigned int row = (blockIdx.x*blockDim.y) + threadIdx.y;
	unsigned int col = threadIdx.x;

	unsigned int leftIndex= col;
//	unsigned int rightindex = row * alignedY;
	unsigned int rightindex= row;
	unsigned int index = (row * alignedY) + col;
	if(threadIdx.x < y)
	{
		d_result[index]=d_leftArr[leftIndex]+d_rightArr[rightindex];
	}
}

template<bool enoughThreads>
__global__ void d_extractArrReal(real_t* d_objectArray, real_t* d_output, unsigned int sampleX, unsigned int sampleY,
										float offsetX, float offsetY, unsigned int alignedSampleY, unsigned int alignedObjectArrayY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	unsigned int outputIndex = (row * alignedSampleY) + col;
	unsigned int inputIndex=(row+offsetX)*alignedObjectArrayY+col+offsetY;

	if(row<sampleX && col<sampleY)
	{
		d_output[outputIndex] = d_objectArray[inputIndex];
	}
}

template<bool enoughThreads>
__global__ void d_extractArrComplex(complex_t* d_objectArray, complex_t* d_output, unsigned int sampleX, unsigned int sampleY,
										float offsetX, float offsetY, unsigned int alignedSampleY, unsigned int alignedObjectArrayY)
{
	unsigned int row = enoughThreads? (blockIdx.x*blockDim.y) + threadIdx.y : blockIdx.x;
	unsigned int col = enoughThreads? threadIdx.x : (blockIdx.y*blockDim.x) + threadIdx.x;

	unsigned int outputIndex = (row * alignedSampleY) + col;
	unsigned int inputIndex=(row+offsetX)*alignedObjectArrayY+col+offsetY;

	if(row<sampleX && col<sampleY)
	{
		d_output[outputIndex] = d_objectArray[inputIndex];
	}
}

__global__ void d_addToArray_r(float * sarray, float* larray, unsigned int* pos_X, unsigned int* posY,
		unsigned int  Np_px, unsigned int Np_py, unsigned int  Np_ox, unsigned int  Np_oy,
		unsigned int  Npos, unsigned int alignedObjectY, const bool isFlat)
{
    // Location in a 3D matrix
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    int idy= blockIdx.y * blockDim.y + threadIdx.y;
    int id = blockIdx.z * blockDim.z + threadIdx.z;
    if ( idx < Np_px && idy < Np_py && id < Npos)
    {
        int idz = id;  // go only through some of the indices
        int id_large =  alignedObjectY*(pos_X[idz]+idx)+(posY[idz]+idy);
        int id_small = Np_py*idx + idy ;
//        if (!isFlat)
//            id_small = id_small + Np_px*Np_py*idz ;
        if (!isFlat)
        	id_small = id_small + Np_px*alignedObjectY*idz ;

      atomicAdd(&larray[id_large] ,sarray[id_small]);
    }
}

__global__ void d_addToArray_c(complex_t * sarray, complex_t* larray, unsigned int* pos_X, unsigned int* posY,
		unsigned int  Np_px, unsigned int Np_py, unsigned int Np_ox, unsigned int  Np_oy,
		unsigned int  Npos, unsigned int alignedObjectY, unsigned int alignedProbeY, const bool isFlat)
{
    // Location in a 3D matrix
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    int idy= blockIdx.y * blockDim.y + threadIdx.y;
    int id = blockIdx.z * blockDim.z + threadIdx.z;
    if ( idx < Np_px && idy < Np_py && id < Npos)
    {
        int idz = id;  // go only through some of the indices
        int id_large =  alignedObjectY*(pos_X[idz]+idx)+(posY[idz]+idy);
        int id_small = Np_py*idx + idy ;
        if (!isFlat)
        	id_small = id_small + Np_px*alignedProbeY*idz ;
//            id_small = id_small + Np_px*Np_py*idz ;
      atomicAdd(&larray[id_large].x ,sarray[id_small].x);
      atomicAdd(&larray[id_large].y ,sarray[id_small].y);
    }
}

__global__ void d_readFromArray_c(complex_t * sarray, const complex_t * larray, /*unsigned int* ind_read,*/ unsigned int* pos_X, unsigned int* pos_Y,
		unsigned int Np_px, unsigned int Np_py, unsigned int Np_pz, unsigned int Np_ox, unsigned int Np_oy,
		unsigned int alignedObjectY, unsigned int alignedProbeY, unsigned int Npos)  {
    // Location in a 3D matrix
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    int idy= blockIdx.y * blockDim.y + threadIdx.y;
    int id = blockIdx.z * blockDim.z + threadIdx.z;

    if ( idx < Np_px & idy < Np_py & id < Npos)
    {
//        int idz = ind_read[id];  // go only through some of the indices
    	int idz = id;
        int id_large = alignedObjectY*(pos_X[idz]+idx)+pos_Y[idz]+idy;
//        int id_large = pos_X[idz]+idx + Np_ox*(pos_Y[idz]+idy);
//        int id_small = idx + Np_px*idy + Np_px*Np_py*idz ;
        int id_small = alignedProbeY*idx + idy + Np_px*alignedProbeY*idz ;
//        sarray[id_small].x = larray[id_large].x ;
//        sarray[id_small].y = larray[id_large].y ;
        sarray[id_small]= larray[id_large];
    }
}

__global__ void d_readFromArray_r(real_t * sarray, const real_t * larray, /*unsigned int* ind_read,*/ unsigned int* pos_X, unsigned int* pos_Y,
		unsigned int Np_px, unsigned int Np_py, unsigned int Np_pz, unsigned int Np_ox, unsigned int Np_oy,
		unsigned int alignedObjectY, unsigned int alignedProbeY, unsigned int Npos)
{
    // Location in a 3D matrix
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    int idy= blockIdx.y * blockDim.y + threadIdx.y;
    int id = blockIdx.z * blockDim.z + threadIdx.z;

    if ( idx < Np_px & idy < Np_py & id < Npos)
    {
//        int idz = ind_read[id];  // go only through some of the indices
    	int idz = id;
        int id_large = alignedObjectY*(pos_X[idz]+idx)+pos_Y[idz]+idy;
//        int id_large = pos_X[idz]+idx + Np_ox*(pos_Y[idz]+idy);
//        int id_small = idx + Np_px*idy + Np_px*Np_py*idz ;
        int id_small = alignedProbeY*idx + idy + Np_px*alignedProbeY*idz ;
//        sarray[id_small].x = larray[id_large].x ;
//        sarray[id_small].y = larray[id_large].y ;
        sarray[id_small]= larray[id_large];
    }
}

__global__ void d_readFromArray_r_fast(real_t * sarray, const real_t * larray,
		unsigned int Np_px, unsigned int Np_py, unsigned int Np_pz, unsigned int Np_ox, unsigned int Np_oy,
		unsigned int alignedObjectY, unsigned int alignedProbeY, unsigned int Npos)
{
    // Location in a 3D matrix
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    int idy= blockIdx.y * blockDim.y + threadIdx.y;
    int id = blockIdx.z * blockDim.z + threadIdx.z;

    if ( idx < Np_px & idy < Np_py & id < Npos)
    {
//        int idz = gC_ind_read[id]-1;  // go only through some of the indices
    	int idz = id;
//        int id_large =    gC_pos_X[idz]+idx + Np_ox*(gC_pos_Y[idz]+idy);
    	int id_large =    alignedObjectY*(gC_pos_X[idz]+idx) + gC_pos_Y[idz]+idy;
//        int id_small = idx + Np_px*idy + Np_px*Np_py*idz ;
    	int id_small = alignedProbeY*idx + idy + Np_px*alignedProbeY*idz ;
        sarray[id_small] =  larray[id_large];
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
//	thrust::device_ptr<real_t> devPtr_a = thrust::device_pointer_cast(a);
//	return thrust::reduce(devPtr_a, devPtr_a+(x*alignedY));

	double sum=h_realSum(a, 0, x, 0, y, alignedY);

//	real_t sum = h_realSumCUB(a, x, y, alignedY);

	return sum;
}

__host__ real_t h_mean2(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY)
{

	double sum=h_realSum(a, 0, x, 0, y, alignedY);

//	double sum=h_realSumCUB(a, x, y, alignedY);

	return sum/(x*y);
}

__host__ real_t h_realSum(const real_t* a, unsigned int x1, unsigned int x2, unsigned int y1, unsigned int y2, unsigned int alignedY)
{
	thrust::device_vector<real_t> output;
	h_reduceToSum<real_t>(a, output, x1, x2, y1, y2, alignedY);
	return thrust::reduce(output.begin(), output.end());

}

//__host__ real_t h_realSumCUB(real_t* d_in, unsigned int x, unsigned int y, unsigned int alignedY)
//{
//
//	real_t* d_out;
//	cudaMalloc((void **)&d_out, sizeof(real_t));
//
//    // Request and allocate temporary storage
//    void            *d_temp_storage = NULL;
//    size_t          temp_storage_bytes = 0;
//    int num_items=x*alignedY;
//
//    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
//    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
//    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
//    cudaDeviceSynchronize();
//
//    real_t sum=0;
//    cudaMemcpy(&sum, d_out, sizeof(real_t), cudaMemcpyDeviceToHost);
//
//	cudaFree(d_out);
//	cudaFree(d_temp_storage);
////    printf("sum: %15e.\n", sum);
//
//    cutilCheckMsg("h_realSumCUB() execution failed!\n");
//
//    return sum;
//}

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

__host__ real_t h_maxFloat(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY)
{
//	thrust::device_ptr<real_t> devPtr_a = thrust::device_pointer_cast(a);
//	return thrust::reduce(devPtr_a, devPtr_a+(x*alignedY), make_float2(FLT_MIN,FLT_MIN), maxFloat2());
	thrust::device_ptr<real_t> devPtr_a = thrust::device_pointer_cast(a);
//	thrust::device_vector<real_t> devPtr_a(devPtr);

	thrust::device_vector<real_t>::iterator iter = thrust::max_element(devPtr_a, devPtr_a+(x*alignedY));
	real_t max_val = *iter;
	return max_val;
}

//__host__ float2 h_subtractFloat2(const float2* a, const float* b,
//                                        unsigned int x, unsigned int y, unsigned int alignedY)
//{
////	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
////	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
////	dim3 block(alignedY, sliceNum, 1);
//
//	d_float2Subtract<<<x, alignedY>>>(a, b, result, y);
//	cutilCheckMsg("d_complexSubtract() execution failed!\n");
//}
//
//__global__ void d_float2Subtract(const float2* a, const float* b, complex_t* result, unsigned int y)
//{
//	unsigned int index = (((blockIdx.x*blockDim.y)+threadIdx.y) * blockDim.x) + threadIdx.x;
//	if(threadIdx.x < y)
//		result[index] = sub_complex_t(a[index], b[index]);
//
//	unsigned int posIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
//
//}

__host__ void h_subtract(const complex_t* a, const complex_t* b, complex_t* result,
                                        unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	d_complexSubtract<<<grid, block>>>(a, b, result, y);
	cutilCheckMsg("d_complexSubtract() execution failed!\n");
}

__host__ void h_subtract(const real_t* a, const real_t* b, real_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	d_subtract<<<grid, block>>>(a, b, result, y);
	cutilCheckMsg("d_complexSubtract() execution failed!\n");
}

__host__ void h_addFactorDivide(real_t* a, real_t* result, real_t factor,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	d_addFactorDivide<<<grid, block>>>(a, result, factor, y);
	cutilCheckMsg("d_complexSubtract() execution failed!\n");
}

__host__ void h_object_sum_update_Gfun(complex_t* a, real_t* b, complex_t* result, real_t factor,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	d_object_sum_update_Gfun<<<grid, block>>>(a, b, result, factor, y);
	cutilCheckMsg("d_complexSubtract() execution failed!\n");
}

void h_addFactor(complex_t* a, complex_t* result, complex_t factor,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	d_addFactor<<<grid, block>>>(a, result, factor, y);
	cutilCheckMsg("d_addFactor() execution failed!\n");
}

void h_addFactor(real_t* a, real_t* result, real_t factor,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	d_addFactor<<<grid, block>>>(a, result, factor, y);
	cutilCheckMsg("d_addFactor() execution failed!\n");
}

__host__ void h_square(real_t* a, real_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	d_square<true><<<grid, block>>>(a, result, x, y, alignedY);
//	d_square<true><<<grid, block>>>(d_arr, d_result, x, y, alignedY);
	cutilCheckMsg("d_complexSubtract() execution failed!\n");
}

__host__ void h_multiplyConju(complex_t* a, complex_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
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
			d_multiplyConju<true><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1,
														axOffset, ayOffset, bxOffset, byOffset);
		}
		else
		{
			unsigned int sliceNum = gh_iDivUp(alignedY, maxThreads);
			dim3 grid(x-blockOffset, sliceNum, 1);
			dim3 block(maxThreads, 1, 1);
			d_multiplyConju<false><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1,
														axOffset, ayOffset, bxOffset, byOffset);
		}
	}
	cutilCheckMsg("d_multiplyConju() execution failed!\n");

}
__host__ void h_multiply(complex_t* a, complex_t* b, complex_t* result,	unsigned int x, unsigned int y, unsigned int alignedY,
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

__host__ void h_multiply(real_t* a, complex_t* b, complex_t* result,	unsigned int x, unsigned int y, unsigned int alignedY,
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

__host__ void h_multiply(real_t* a, real_t* b, real_t* result,	unsigned int x, unsigned int y, unsigned int alignedY,
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
			d_realMultiply<true><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1,
														axOffset, ayOffset, bxOffset, byOffset);
		}
		else
		{
			unsigned int sliceNum = gh_iDivUp(alignedY, maxThreads);
			dim3 grid(x-blockOffset, sliceNum, 1);
			dim3 block(maxThreads, 1, 1);
			d_realMultiply<false><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1,
														axOffset, ayOffset, bxOffset, byOffset);
		}
	}
	cutilCheckMsg("d_realMultiply() execution failed!\n");
}

__host__ void h_checkCache(	thrust::device_vector<real_t>& m_factors,
thrust::host_vector<bool>& m_cachedFlags,
thrust::host_vector<real_t>& m_cachedFactors, thrust::device_vector<bool>& m_flags, real_t objMax, real_t probeMax,
		bool phaseConstraint,bool updateProbe, bool updateProbeModes, bool RMS)
{
	bool passedFlags[3] = {phaseConstraint, updateProbe, updateProbeModes};
	for(size_t i=0; i<m_cachedFlags.size();++i)
		if(m_cachedFlags[i]!=passedFlags[i])
		{
			m_cachedFlags[i]=passedFlags[i];
			m_flags[i] = m_cachedFlags[i];
		}
	real_t passedFactors[2] = {1.0/objMax, 1.0/probeMax};
	for(size_t i=0; i<m_cachedFactors.size();++i)
	{
		if(fabs(m_cachedFactors[i]-passedFactors[i])>EPS)
		{
			m_cachedFactors[i]=passedFactors[i];
			m_factors[i] = m_cachedFactors[i];
		}
	}
}


__host__ void h_multiply(const complex_t* a, const complex_t& b, complex_t* result,
	unsigned int x, unsigned int y, unsigned int alignedY, bool normalize)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_complexMultiply<true> <<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1);
	else 				d_complexMultiply<false><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1);
	cutilCheckMsg("h_multiply() execution failed!\n");
}

__host__ void h_multiply(const real_t* a, const real_t& b, real_t* result,
		unsigned int x, unsigned int y, unsigned int alignedY, bool normalize)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_multiply<true> <<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1);
	else 				d_multiply<false><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1);
	cutilCheckMsg("h_multiply() execution failed!\n");
}

__host__ void h_multiplyPage(complex_t* a, complex_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
		unsigned int pagex, unsigned int axOffset, unsigned int ayOffset, unsigned int bxOffset, unsigned int byOffset)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_multiplyPage<true> <<<grid, block>>>(a, b, result, x, y, alignedY, pagex);
	else 				d_multiplyPage<false><<<grid, block>>>(a, b, result, x, y, alignedY, pagex);
	cutilCheckMsg("h_multiplyPage() execution failed!\n");
}

__host__ void h_multiplyAbsConjuRealWhole(complex_t* a, complex_t* b, complex_t* c, real_t* result1, real_t* result2, unsigned int x, unsigned int y, unsigned int alignedY,
		unsigned int pagex)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_multiplyAbsConjuRealWhole<true> <<<grid, block>>>(a, b, c, result1, result2, x, y, alignedY, pagex);
	else 				d_multiplyAbsConjuRealWhole<false><<<grid, block>>>(a, b, c, result1, result2, x, y, alignedY, pagex);
	cutilCheckMsg("h_multiplyRow() execution failed!\n");
}

__host__ void h_multiplyAbsConjuReal(complex_t* a, complex_t* b, complex_t* c, real_t* result1, real_t* result2, unsigned int x, unsigned int y, unsigned int alignedY,
		unsigned int pagex)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_multiplyAbsConjuReal<true> <<<grid, block>>>(a, b, c, result1, result2, x, y, alignedY, pagex);
	else 				d_multiplyAbsConjuReal<false><<<grid, block>>>(a, b, c, result1, result2, x, y, alignedY, pagex);
	cutilCheckMsg("h_multiplyRow() execution failed!\n");
}

__host__ void h_multiplyRow(complex_t* a, real_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
		bool normalize, unsigned int axOffset, unsigned int ayOffset, unsigned int bxOffset, unsigned int byOffset)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_multiplyRow<true> <<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1);
	else 				d_multiplyRow<false><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1);
	cutilCheckMsg("h_multiplyRow() execution failed!\n");
}

__host__ void h_multiplyColumn(complex_t* a, real_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
		bool normalize, unsigned int axOffset, unsigned int ayOffset, unsigned int bxOffset, unsigned int byOffset)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_multiplyColum<true> <<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1);
	else 				d_multiplyColum<false><<<grid, block>>>(a, b, result, x, y, alignedY, normalize?1.0/(real_t)(x*y):1);
	cutilCheckMsg("h_multiplyColumn() execution failed!\n");
}

__host__ void h_get_optimal_step_lsq(complex_t* chi,complex_t* object_update_proj, complex_t* dPO, complex_t* probe, real_t lambda,
		real_t* AA1, complex_t* AA2, real_t* AA4, real_t* Atb1, real_t* Atb2, unsigned int x, unsigned int y, unsigned int alignedY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_get_optimal_step_lsq<true> <<<grid, block>>>(chi, object_update_proj, dPO, probe, lambda,
			AA1, AA2, AA4, Atb1, Atb2, x, y, alignedY);
	else 				d_get_optimal_step_lsq<false><<<grid, block>>>(chi, object_update_proj, dPO, probe, lambda,
			AA1, AA2, AA4, Atb1, Atb2, x, y, alignedY);
	cutilCheckMsg("h_get_optimal_step_lsq() execution failed!\n");

}

__host__ void h_mul_rca_mulc_rcr(complex_t* obj_proj_i, complex_t* modes_i, complex_t* chi_i, real_t* weight_proj,
		unsigned int x, unsigned int y, unsigned int alignedY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_mul_rca_mulc_rcr<true> <<<grid, block>>>(obj_proj_i, modes_i, chi_i, weight_proj, x, y, alignedY);
	else 				d_mul_rca_mulc_rcr<false><<<grid, block>>>(obj_proj_i, modes_i, chi_i, weight_proj, x, y, alignedY);
	cutilCheckMsg("h_mul_rca_mulc_rcr() execution failed!\n");

}

__host__ void h_multiplyReal(real_t* a, real_t* result,
	unsigned int x, unsigned int y, unsigned int alignedY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_realMultiply<true> <<<grid, block>>>(a, result, x, y, alignedY);
	else 				d_realMultiply<false> <<<grid, block>>>(a, result, x, y, alignedY);
	cutilCheckMsg("h_multiplyReal() execution failed!\n");
}



__host__ void h_normalize(complex_t* a, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_complexMultiply<true> <<<grid, block>>>(a, factor, a, x, y, alignedY);
	else				d_complexMultiply<false><<<grid, block>>>(a, factor, a, x, y, alignedY);
	cutilCheckMsg("h_normalize() execution failed\n");
}

__host__ void h_normalize(const complex_t* a, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_complexMultiply<true> <<<grid, block>>>(a, factor, result, x, y, alignedY);
	else				d_complexMultiply<false><<<grid, block>>>(a, factor, result, x, y, alignedY);
	cutilCheckMsg("h_normalize() execution failed\n");
}

__host__ void h_normalize(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY)
{
	thrust::device_ptr<real_t> devPtr_a = thrust::device_pointer_cast(a);
	thrust::constant_iterator<real_t> maxValue(h_realMax(a,x,y,alignedY));
	thrust::transform(devPtr_a, devPtr_a+(x*alignedY), maxValue, devPtr_a, thrust::divides<real_t>());
	cutilCheckMsg("h_normalize() execution failed\n");
}

__host__ void h_normalize(real_t* a, real_t factor, unsigned int x, unsigned int y, unsigned int alignedY)
{
	thrust::device_ptr<real_t> devPtr_a = thrust::device_pointer_cast(a);
	thrust::constant_iterator<real_t> factorValue(factor);
	thrust::transform(devPtr_a, devPtr_a+(x*alignedY), factorValue, devPtr_a, thrust::divides<real_t>());
	cutilCheckMsg("h_normalize() execution failed\n");
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

__host__ real_t h_norm2Mat(real_t* d_arr, real_t* d_result, unsigned int x, unsigned int y, unsigned int alignedY)
{

	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_square<true><<<grid, block>>>(d_arr, d_result, x, y, alignedY);
	else				d_square<false><<<grid, block>>>(d_arr, d_result, x, y, alignedY);
	cutilCheckMsg("h_realComplexReal() execution failed\n");

	real_t result=h_realSum(d_result, x, y, alignedY);
//	real_t result=h_realSum(d_result, 0, x, 0, y, alignedY);
	real_t xresult=sqrt_real_t(result/(x*y));

	return xresult;
}

__host__ real_t h_norm2Mat(complex_t* d_arr, real_t* d_result, unsigned int x, unsigned int y, unsigned int alignedY)
{
	h_realComplexAbs(d_arr, d_result, x, y, alignedY, true);
	real_t result=h_realSum(d_result, x, y, alignedY);
//	real_t result=h_realSum(d_result, 0, x, 0, y, alignedY);
	real_t xresult=sqrt_real_t(result/(x*y));
	return xresult;
}

__host__ void h_squareRoot(real_t* d_arr, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_realsquareRoot<true><<<grid, block>>>(d_arr, result, x, y, alignedY);
	else				d_realsquareRoot<false><<<grid, block>>>(d_arr, result, x, y, alignedY);
	cutilCheckMsg("h_realComplexReal() execution failed\n");
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

__host__ void h_realComplexExp(const real_t* src, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(x,alignedY,grid,block);
	if(enoughThreads)	d_realComplexExp<true> <<<grid, block>>>(src, result, x, y, alignedY, factor);
	else				d_realComplexExp<false><<<grid, block>>>(src, result, x, y, alignedY, factor);
	cutilCheckMsg("realComplexExp() execution failed\n");
}

__host__ void h_set_projections(real_t* p_object, real_t* proj, unsigned int* p_positions_x, unsigned int* p_positions_y,
		unsigned int objectX, unsigned int objectY, unsigned int alignedObjectY, unsigned int probeX, unsigned int probeY, unsigned int Npos)
{
    int const threadsPerBlockEachDim =  32;
    int const blocksPerGrid_M = (probeX + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_N = (probeY + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_O = Npos;

    dim3 const dimBlock(blocksPerGrid_M, blocksPerGrid_N, blocksPerGrid_O);
    dim3 const dimThread(threadsPerBlockEachDim, threadsPerBlockEachDim, 1);
    bool isFlat=true;

    d_addToArray_r<<<dimBlock, dimThread>>>(proj, p_object, p_positions_x, p_positions_y ,probeX , probeY, objectX, objectY, Npos, alignedObjectY, isFlat);
}

__host__ void h_set_projections(complex_t* p_object, complex_t* proj, unsigned int* p_positions_x, unsigned int* p_positions_y,
		unsigned int objectX, unsigned int objectY, unsigned int alignedObjectY, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY,
		unsigned int Npos, bool isFlat)
{
    int const threadsPerBlockEachDim =  32;
    int const blocksPerGrid_M = (probeX + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_N = (probeY + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_O = Npos;

    dim3 const dimBlock(blocksPerGrid_M, blocksPerGrid_N, blocksPerGrid_O);
    dim3 const dimThread(threadsPerBlockEachDim, threadsPerBlockEachDim, 1);

    d_addToArray_c<<<dimBlock, dimThread>>>(proj, p_object, p_positions_x, p_positions_y ,probeX , probeY, objectX, objectY, Npos, alignedObjectY, alignedProbeY, isFlat);
}

__host__ void h_get_projections(const complex_t* p_object, complex_t* proj, unsigned int* p_positions_x, unsigned int* p_positions_y,
		unsigned int objectX, unsigned int objectY, unsigned int alignedObjectY, unsigned int probeX, unsigned int probeY, unsigned int probeZ,
		unsigned int alignedProbeY, unsigned int Npos)
{

    int const threadsPerBlockEachDim =  32;
    int const blocksPerGrid_M = (probeX + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_N = (probeY + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_O = Npos;

    dim3 const dimBlock(blocksPerGrid_M, blocksPerGrid_N, blocksPerGrid_O);
    dim3 const dimThread(threadsPerBlockEachDim, threadsPerBlockEachDim, 1);

    d_readFromArray_c<<<dimBlock, dimThread>>>(proj, p_object, p_positions_x, p_positions_y , probeX, probeY, probeZ,
    		objectX, objectY, alignedObjectY, alignedProbeY, Npos);

}

__host__ void h_get_projections(const real_t* p_object, real_t* proj, unsigned int* p_positions_x, unsigned int* p_positions_y,
		unsigned int objectX, unsigned int objectY, unsigned int alignedObjectY, unsigned int probeX, unsigned int probeY, unsigned int probeZ,
		unsigned int alignedProbeY, unsigned int Npos)
{

    int const threadsPerBlockEachDim =  32;
    int const blocksPerGrid_M = (probeX + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_N = (probeY + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_O = Npos;

    dim3 const dimBlock(blocksPerGrid_M, blocksPerGrid_N, blocksPerGrid_O);
    dim3 const dimThread(threadsPerBlockEachDim, threadsPerBlockEachDim, 1);

    d_readFromArray_r<<<dimBlock, dimThread>>>(proj, p_object, p_positions_x, p_positions_y , probeX, probeY, probeZ,
    		objectX, objectY, alignedObjectY, alignedProbeY, Npos);

//    if(Npos<MAX_IND_READ)
//    {
//        cudaMemcpyToSymbol(gC_pos_X, p_positions_x, Npos*sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
//        cudaMemcpyToSymbol(gC_pos_Y, p_positions_y, Npos*sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
//        d_readFromArray_r_fast<<<dimBlock, dimThread>>>(proj, p_object, probeX, probeY, probeZ,
//        		objectX, objectY, alignedObjectY, alignedProbeY, Npos);
//    }
//    else
//    {
//        d_readFromArray_r<<<dimBlock, dimThread>>>(proj, p_object, p_positions_x, p_positions_y , probeX, probeY, probeZ,
//        		objectX, objectY, alignedObjectY, alignedProbeY, Npos);
//    }


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

	cudaDeviceSynchronize();

//	d_check<<<x, y>>>(d_data);


}

__host__ void h_shiftFFTy(real_t* d_data, real_t* d_temp, unsigned int x, unsigned int y, unsigned int alignedY, cudaStream_t* stream)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);

	d_shiftY<real_t, true><<<grid, block,0,(stream?*stream:0)>>>(d_data, d_temp, (float)y/2.0, 0, x, y, alignedY);
	cutilCheckMsg("h_shiftFFT() execution failed!\n");

	cudaDeviceSynchronize();

}

__host__ void h_shiftFFTtmp(complex_t* d_probe, complex_t* d_tempprobe, complex_t* d_copyprobe, unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);

	d_shiftX<complex_t, true><<<grid, block,0,0>>>(d_copyprobe, d_tempprobe, (float)x/2.0, 0, x, y, alignedY);
	cutilCheckMsg("h_shiftFFTtmp() execution failed!\n");

	cudaDeviceSynchronize();

	d_shiftY<complex_t, true><<<grid, block,0,0>>>(d_tempprobe, d_probe, (float)y/2.0, 0, x, y, alignedY);
	cutilCheckMsg("h_shiftFFTtmp() execution failed!\n");

	cudaDeviceSynchronize();

//	d_checkcomplex<<<x, y>>>(d_probe);


}

__host__ void h_shiftFFTtwo(complex_t* d_probe, complex_t* d_tempprobe, unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);

	d_shiftX<complex_t, true><<<grid, block,0,0>>>(d_probe, d_tempprobe, (float)x/2.0, 0, x, y, alignedY);
	cutilCheckMsg("h_shiftFFTtmp() execution failed!\n");

	cudaDeviceSynchronize();

	d_shiftY<complex_t, true><<<grid, block,0,0>>>(d_tempprobe, d_probe, (float)y/2.0, 0, x, y, alignedY);
	cutilCheckMsg("h_shiftFFTtmp() execution failed!\n");

	cudaDeviceSynchronize();

//	d_checkcomplex<<<x, y>>>(d_probe);
}

__host__ void imshift_fft(complex_t* d_probe, unsigned int x, unsigned int y, unsigned int alignedY, float radNo1, float radNo2)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x, sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);

	d_imshift_fft<<<grid, block>>>(d_probe, x/2, y/2, radNo1, radNo2, x, y, alignedY);

//	d_checkcomplex<<<x, y>>>(d_probe);

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

__host__ complex_t h_innerProductOne(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
								unsigned int x, unsigned int y, unsigned int alignedY)
{
	unsigned int reductionThreads = getReductionThreadNum(alignedY);
	dim3 grid(x, 1, 1);
	dim3 block(reductionThreads, 1, 1);
	size_t shared_mem_size = (block.x <= 32) ? 2* block.x * sizeof(complex_t) : block.x *  sizeof(complex_t);

	switch (block.x)
	{
	case   8:	d_innerProductOne<   8><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0, y);
	break;
	case  16:	d_innerProductOne<  16><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0, y);
	break;
	case  32:	d_innerProductOne<  32><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0, y);
	break;
	case  64:	d_innerProductOne<  64><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0, y);
	break;
	case 128:	d_innerProductOne< 128><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0, y);
	break;
	case 256:	d_innerProductOne< 256><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0, y);
	break;
	case 512:	d_innerProductOne< 512><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0, y);
	break;
	case 1024:	d_innerProductOne<1024><<<grid, block, shared_mem_size>>>(d_u, d_v, d_output, 1.0, y);
	break;
	}
	cutilCheckMsg("d_innerProduct() execution failed!\n");

	thrust::device_ptr<complex_t> devPtr = thrust::device_pointer_cast(d_output);
	complex_t result = thrust::reduce(devPtr, devPtr+x, make_complex_t(0,0), complexSum());

	return result;
}

__host__ void h_innerProductModes(complex_t* d_u, complex_t* d_v, complex_t* d_factor, unsigned int index,
		unsigned int modesNum, unsigned int x, unsigned int y, unsigned int alignedY)
{
//	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
//	dim3 grid(modesNum, gh_iDivUp(x,sliceNum), 1);
//	dim3 block(alignedY, sliceNum, 1);

	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x,sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);

	unsigned int offset=x*alignedY;
	d_innerProductModes<<<grid, block>>>(d_u, d_v, d_factor, index, modesNum, x, y, offset);
	cutilCheckMsg("d_innerProductModes() execution failed!\n");
}

__host__ void h_extracSubArrReal(real_t* d_objectArray, real_t* d_output, unsigned int offsetX, unsigned int offsetY,
		unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
		unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY)
{

	dim3 grid, block;
	bool enoughThreads = calcGrids(sampleX,alignedSampleY,grid,block);

	if(enoughThreads)	d_extractArrReal<true> <<<grid, block>>>(d_objectArray, d_output, sampleX, sampleY, offsetX, offsetY, alignedSampleY, alignedObjectArrayY);
	else				d_extractArrReal<false><<<grid, block>>>(d_objectArray, d_output, sampleX, sampleY, offsetX, offsetY, alignedSampleY, alignedObjectArrayY);
	cutilCheckMsg("h_extractObjectArray() execution failed!\n");
//		d_check<<<sampleX, alignedSampleY>>>(d_output);
}

__host__ void h_extracSubArrComplex(complex_t* d_objectArray, complex_t* d_output, unsigned int offsetX, unsigned int offsetY,
		unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
		unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY)
{
	dim3 grid, block;
	bool enoughThreads = calcGrids(sampleX,alignedSampleY,grid,block);

	if(enoughThreads)	d_extractArrComplex<true> <<<grid, block>>>(d_objectArray, d_output, sampleX, sampleY, offsetX, offsetY, alignedSampleY, alignedObjectArrayY);
	else				d_extractArrComplex<false><<<grid, block>>>(d_objectArray, d_output, sampleX, sampleY, offsetX, offsetY, alignedSampleY, alignedObjectArrayY);
	cutilCheckMsg("h_extractObjectArray() execution failed!\n");
}

__host__ void h_realModalSum(const real_t* d_modes, real_t* d_output, unsigned int modesNum,
								unsigned int x, unsigned int y, unsigned int alignedY, bool sqaureRoot)
{
	// Along the z direction it must be a 3d array
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x,sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);
	
	d_modalSum<real_t><<<grid, block>>>(d_modes, d_output, modesNum, x, y, sqaureRoot);
	cutilCheckMsg("d_modalSum() execution failed!\n");
}

__host__ void h_realModalSum(const complex_t* d_modes, complex_t* d_output, unsigned int modesNum,
								unsigned int x, unsigned int y, unsigned int alignedY, bool sqaureRoot)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x,sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);

	d_modalSumComplex<<<grid, block>>>(d_modes, d_output, modesNum, x, y, sqaureRoot);
	cutilCheckMsg("d_modalSum() execution failed!\n");
}

__host__ void h_complexSum(complex_t* d_leftArr, complex_t* d_rightArr, complex_t* d_result, real_t leftFactor, real_t rightFactor, unsigned int x, unsigned int y,
								unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x,sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);

	d_complexSum<<<grid, block>>>(d_leftArr, d_rightArr, d_result, leftFactor, rightFactor, x, y, alignedY);
	cutilCheckMsg("d_modalSum() execution failed!\n");
}

__host__ void h_realSum(real_t* d_leftArr, real_t* d_rightArr, real_t* d_result, real_t leftFactor, real_t rightFactor, unsigned int x, unsigned int y,
								unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x,sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);

	d_realSum<<<grid, block>>>(d_leftArr, d_rightArr, d_result, leftFactor, rightFactor, x, y, alignedY);
	cutilCheckMsg("d_realSum() execution failed!\n");
}

__host__ void h_realSingleSum(real_t* d_leftArr, real_t* d_rightArr, real_t* d_result, unsigned int x, unsigned int y,
								unsigned int alignedY)
{
	unsigned int sliceNum = gh_iDivDown(GPUQuery::getInstance()->getGPUMaxThreads(), alignedY);
	dim3 grid(gh_iDivUp(x,sliceNum), 1, 1);
	dim3 block(alignedY, sliceNum, 1);

	d_realSingleSum<<<grid, block>>>(d_leftArr, d_rightArr, d_result, x, y, alignedY);
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
