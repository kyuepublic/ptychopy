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

__global__ void d_switchprobe(complex_t* d_u, complex_t* d_v,
		 unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
{
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(threadIdx.x < probeY)
		d_u[probeIndex] = d_v[probeIndex];

//	complex_t value = d_u[probeIndex];
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

__global__ void d_initVarMOdes(complex_t* d_varModes, real_t* d_randarr1, real_t* d_randarr2, unsigned int probeX,
		unsigned int probeY, unsigned int alignedProbeY)
{
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(threadIdx.x < probeY)
	{
		real_t expright=2*CUDART_PI*d_randarr2[probeIndex];
		real_t expleft=d_randarr1[probeIndex];
		complex_t expTo=make_complex_t((expleft*cos_real_t(expright)), (expleft*sin_real_t(expright)));
		d_varModes[probeIndex]=expTo;
	}
}

__global__ void d_norm2(complex_t* d_varModes, real_t* d_result, unsigned int probeX,
		unsigned int probeY, unsigned int alignedProbeY)
{
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(threadIdx.x < probeY)
	{
		complex_t tmp=d_varModes[probeIndex];
		real_t realcom=real_complex_t(tmp);
		real_t imagcom=imag_complex_t(tmp);
		real_t tempResult=realcom*realcom+imagcom*imagcom;
		d_result[probeIndex]=tempResult;
	}
}

//__global__ void d_preCalillum(complex_t* d_modes, complex_t* d_aprobe2, unsigned int probeX,
//		unsigned int probeY, unsigned int alignedProbeY)
//{
//	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
//	if(threadIdx.x < probeY)
//	{
//		complex_t tmp=d_modes[probeIndex];
//		real_t realcom=real_complex_t(tmp);
//		real_t imagcom=imag_complex_t(tmp);
//		real_t tempResult=realcom*realcom+imagcom*imagcom;
//		d_aprobe2[probeIndex]=tempResult;
//	}
//}

//__global__ void d_normalizeVariProbe(complex_t* d_extramodes, real_t factor, unsigned int probeX,
//		unsigned int probeY, unsigned int alignedProbeY)
//{
//	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
//	if(threadIdx.x < probeY)
//	{
//		complex_t tmp=d_varModes[probeIndex];
//
//		real_t realcom=real_complex_t(tmp);
//		real_t imagcom=imag_complex_t(tmp);
//		real_t tempResult=realcom*realcom+imagcom*imagcom;
//		d_result[probeIndex]=tempResult;
//	}
//}


//
//__global__ void addToArray_r( float const * sarray, float * larray,cuint16* ind_read, cuint16* pos_X, cuint16* posY,
//                                cuint Np_px,cuint Np_py, cuint Np_pz,cuint Np_ox, cuint Np_oy,
//                                cuint Npos,const bool isFlat)  {
//    // Location in a 3D matrix
//    int idx= blockIdx.x * blockDim.x + threadIdx.x;
//    int idy= blockIdx.y * blockDim.y + threadIdx.y;
//    int id = blockIdx.z * blockDim.z + threadIdx.z;
//    if ( idx < Np_px & idy < Np_py & id < Npos)
//    {
//        int idz = ind_read[id]-1;  // go only through some of the indices
//        int id_large =  pos_X[idz]+idx + Np_ox*(posY[idz]+idy);
//        int id_small = idx + Np_px*idy ;
//        if (!isFlat)
//            id_small = id_small + Np_px*Np_py*idz ;
//      atomicAdd(&larray[id_large] ,sarray[ id_small ]);
//      //  larray[id_large] += sarray[ id_small ];
//    }
//}

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

__global__ void d_initProbeMLH(complex_t* d_probeWavefront, real_t startValue, real_t stepsize, real_t fzpValue,
		real_t D_FZP, real_t D_H, real_t fu, real_t startFU, real_t fustepsize, real_t expkDz, complex_t kzvalue, unsigned int probeY, complex_t* d_pfValue)
{
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(threadIdx.x < probeY)
	{
		float2 initialValues;

		initialValues.x = startValue + stepsize*threadIdx.x; // X matrix euqals to lxvalues.
		initialValues.y = startValue + stepsize*blockIdx.x; // Y matrix

		real_t suminitV=initialValues.x*initialValues.x + initialValues.y*initialValues.y;

		real_t sumInit = fzpValue*suminitV;
		real_t cost=cos_real_t(sumInit);
		real_t sint = sin_real_t(sumInit);

		// Calculate the C H T value
		real_t sqrtInit = sqrt_real_t(suminitV);
		real_t Cvalue = sqrtInit <= D_FZP?1:0;
		real_t Hvalue = sqrtInit >= D_H?1:0;
		complex_t Tvalue = make_complex_t(cost, sint);
		complex_t Invalue = mul_complex_t(mul_complex_t(make_complex_t(Cvalue, 0.0), Tvalue), make_complex_t(Hvalue, 0.0));

//		float2 lxVaules;
//		lxVaules.x = startValue+stepsize*threadIdx.x;
//		lxVaules.y = startValue+stepsize*blockIdx.x;

		// Every elmentwise square plus then make the complex

		float2 luValues;
		if(threadIdx.x < probeY/2)
			luValues.x = startFU + fustepsize*threadIdx.x;
		else
			luValues.x = -fu/2 + fustepsize*(threadIdx.x-probeY/2);

		if(blockIdx.x < gridDim.x/2)
			luValues.y = startFU + fustepsize*blockIdx.x;
		else
			luValues.y = -fu/2 + fustepsize*(blockIdx.x-gridDim.x/2);

		real_t tempsumkDzuv=luValues.x*luValues.x+luValues.y*luValues.y;
		real_t sumkDzuv=expkDz*tempsumkDzuv;
		complex_t tempkDzuv = make_complex_t(cos_real_t(sumkDzuv), sin_real_t(sumkDzuv));
		complex_t pf_complex=mul_complex_t(kzvalue, tempkDzuv);
		d_pfValue[probeIndex]=pf_complex;

		real_t sumkDzxy=expkDz*suminitV;
		complex_t expsumkDzxy=make_complex_t(cos_real_t(sumkDzxy), sin_real_t(sumkDzxy));
		complex_t kern_complex=mul_complex_t(Invalue, expsumkDzxy);

		unsigned int sq1=(gridDim.x*blockDim.x+probeY)/2;
		unsigned int sq2=(gridDim.x*blockDim.x-probeY)/2;
		// Finish kern matrix
		// u,v to x,y, assume and implement z > 0 TODO z < 0

		// Start to do fftshift clockwise 1 2 3 4

		if (threadIdx.x < probeY/2)
		    {
		        if (blockIdx.x < gridDim.x/2)
		        {
		            // First Quad
		        	d_probeWavefront[probeIndex + sq1]=kern_complex;
		        }
		        else
		        {
		        	// Fourth Quad
		        	d_probeWavefront[probeIndex - sq2]=kern_complex;
		        }
		    }
		    else
		    {
		        if (blockIdx.x < gridDim.x/2)
		        {
		            // Second Quad
		        	d_probeWavefront[probeIndex + sq2]=kern_complex;
		        }
		        else
		        {
		        	// Third Quad
		        	d_probeWavefront[probeIndex - sq1]=kern_complex;
		        }
		    }

//		__syncthreads();
//		kern_complex=d_probeWavefront[probeIndex];
//		sq1=1;
//		real_t probe = d_hypotOgrid(blockIdx.x, threadIdx.x, gridDim.x)*factor;
//		probe = (probe!=0)? sinpi_real_t(probe)/(CUDART_PI*probe) : 1;
//		probe = abs(probe*d_calculateGaussian(blockIdx.x, threadIdx.x, gridDim.x, window));
//
//		d_probeWavefront[probeIndex] = make_complex_t(probe, 0.0);
	}
}

__global__ void d_endProbeMLH(complex_t* d_probeWavefront,complex_t* d_pfValue, unsigned int probeY)
{
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
//	complex_t tempWave=d_probeWavefront[probeIndex];
//	complex_t temppf=d_pfValue[probeIndex];

	complex_t sumOut=mul_complex_t(d_probeWavefront[probeIndex], d_pfValue[probeIndex]);

	d_pfValue[probeIndex]=sumOut;
}


__global__ void  d_shiftEnd(complex_t* d_probeWavefront,complex_t* d_pfValue, unsigned int probeY)
{
	unsigned int probeIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned int sq1=(gridDim.x*blockDim.x+probeY)/2;
	unsigned int sq2=(gridDim.x*blockDim.x-probeY)/2;

//	complex_t temp = d_pfValue[probeIndex];

	if (threadIdx.x < probeY/2)
	    {
	        if (blockIdx.x < gridDim.x/2)
	        {
	            // First Quad
	        	d_probeWavefront[probeIndex + sq1]=d_pfValue[probeIndex];
	        }
	        else
	        {
	        	// Fourth Quad
	        	d_probeWavefront[probeIndex - sq2]=d_pfValue[probeIndex];
	        }
	    }
	    else
	    {
	        if (blockIdx.x < gridDim.x/2)
	        {
	            // Second Quad
	        	d_probeWavefront[probeIndex + sq2]=d_pfValue[probeIndex];
	        }
	        else
	        {
	        	// Third Quad
	        	d_probeWavefront[probeIndex - sq1]=d_pfValue[probeIndex];
	        }
	    }

}


//__global__ void d_check1(real_t* d_data)
//{
//
//	unsigned int Index = (blockIdx.x * blockDim.x) + threadIdx.x;
//
//	real_t temp=d_data[Index];
//	unsigned int sq1=1;
//}

//__global__ void d_checkcomplex1(complex_t* d_data)
//{
//
//	unsigned int Index = (blockIdx.x * blockDim.x) + threadIdx.x;
//
//	complex_t temp=d_data[Index];
//	unsigned int sq1=1;
//}



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


__global__ void addToArray_c(float2 * sarray, float2 * larray,  unsigned int * pos_X, unsigned int * posY,
		unsigned int  Np_px, unsigned int Np_py, unsigned int  Np_ox, unsigned int  Np_oy,
		unsigned int  Npos, const bool isFlat)
{
    // Location in a 3D matrix id==ind_read
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    int idy= blockIdx.y * blockDim.y + threadIdx.y;
    int id = blockIdx.z * blockDim.z + threadIdx.z;
    if ( idx < Np_px && idy < Np_py && id < Npos)
    {
//        int idz = ind_read[id]-1;  // go only through some of the indices
    	int idz = id;
        int id_large =  pos_X[idz]+idx + Np_ox*(posY[idz]+idy);
        int id_small = idx + Np_px*idy ;
        if (!isFlat)
            id_small = id_small + Np_px*Np_py*idz ;
        atomicAdd(&larray[id_large].x ,sarray[ id_small ].x);
        atomicAdd(&larray[id_large].y ,sarray[ id_small ].y);
    }
}

__global__ void addToArray_r(float * sarray, float* larray, unsigned int * pos_X, unsigned int * posY,
		unsigned int  Np_px, unsigned int Np_py, unsigned int  Np_ox, unsigned int  Np_oy,
		unsigned int  Npos, unsigned int obalignedProbeY, const bool isFlat)
{
    // Location in a 3D matrix
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    int idy= blockIdx.y * blockDim.y + threadIdx.y;
    int id = blockIdx.z * blockDim.z + threadIdx.z;
    if ( idx < Np_px && idy < Np_py && id < Npos)
    {
    	//int idz = ind_read[id]-1;
        int idz = id;  // go only through some of the indices
//        int id_large =  pos_X[idz]+idx + Np_ox*(posY[idz]+idy);
//        int id_large =  pos_X[idz]+idx + obalignedProbeY*(posY[idz]+idy);
        int id_large =  obalignedProbeY*(pos_X[idz]+idx) + (posY[idz]+idy);
//        int id_small = idx + Np_px*idy ;
        int id_small = Np_py*idx + idy ;

//        if (!isFlat)
//            id_small = id_small + Np_px*Np_py*idz ;
      atomicAdd(&larray[id_large] ,sarray[ id_small ]);
      //  larray[id_large] += sarray[ id_small ];
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

__host__ complex_t h_Orth(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
							unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
{
	complex_t factor = h_innerProductOne(d_u, d_v, d_output, probeX, probeY, alignedProbeY);
//	unsigned int temp=1;

	return factor;


}

__host__ void h_orthro(complex_t* d_u, complex_t* d_v, complex_t* d_factor, unsigned int index,
		unsigned int modesNum, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
{
	h_innerProductModes(d_u, d_v, d_factor, index, modesNum, probeX, probeY, alignedProbeY);
}

__host__ void h_switchprobe(complex_t* d_u, complex_t* d_v,
		unsigned int modesNum, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
{
	d_switchprobe<<<probeX, alignedProbeY>>>(d_u, d_v, probeX, probeY, alignedProbeY);
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

__host__ void h_initVarModes(complex_t* d_varModes, real_t* d_randarr1, real_t* d_randarr2, unsigned int probeX,
		unsigned int probeY, unsigned int alignedProbeY)
{
	d_initVarMOdes<<<probeX, alignedProbeY>>>(d_varModes, d_randarr1, d_randarr2, probeX,
			 probeY, alignedProbeY);
}

__host__ real_t h_norm2(complex_t* d_extramodes, real_t* d_result, unsigned int probeX,
		unsigned int probeY, unsigned int alignedProbeY)
{
	d_norm2<<<probeX, alignedProbeY>>>(d_extramodes, d_result, probeX,
			 probeY, alignedProbeY);
	real_t result=h_realSum(d_result, probeX, probeY, alignedProbeY);
	real_t xresult=sqrt_real_t(result/(probeX*probeY));

	return xresult;
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

__host__ void h_initProbeMLH(complex_t* d_probeWavefront, double dx_fzp, double fzpValue, double D_FZP, double D_H, double lambda, double fl, double beamSize, complex_t* d_pfValue,
					unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY, bool simulated)
{
//	if(simulated)
//		d_initProbeSimulated<<<probeX, alignedProbeY>>>(d_probeWavefront, d_phFunc, gaussWindow, dx_s/beamSize, probeY);
//	else
	double stepsize = dx_fzp*probeX/(probeX-1);
	double startValue = -dx_fzp*probeX/2; // -dxy*M/2
	double k = 2*CUDART_PI/lambda;
	double z = fl + beamSize;
	double kz=k*z;

	double cost=cos(kz);
	double sint=sin(kz);


	complex_t kzvalue = make_complex_t(cost, sint);

	double expkDz=k/(2*z);


	double fu = lambda*(fl+beamSize)/dx_fzp;

	double fustepsize = fu/(probeX-1);
	double startFU = -fu/2 + fustepsize*(probeX/2); // Rightside start FU

//	real_t testvv = 2.1*(3/2);// 2.1*1 or

	d_initProbeMLH<<<probeX, alignedProbeY>>>(d_probeWavefront, startValue, stepsize, fzpValue*CUDART_PI, D_FZP/2, D_H/2, fu, startFU, fustepsize, expkDz, kzvalue, probeY, d_pfValue);

//	d_check<<<probeX, alignedProbeY>>>(d_probeWavefront);

	cutilCheckMsg("d_initProbeMLH() execution failed!\n");



}

__host__ void h_endProbeMLH(complex_t* d_probeWavefront, complex_t* d_pfValue,
					unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY, bool simulated)
{

	d_endProbeMLH<<<probeX, alignedProbeY>>>(d_probeWavefront, d_pfValue, probeY);

	d_shiftEnd<<<probeX, alignedProbeY>>>(d_probeWavefront, d_pfValue, probeY);

	cutilCheckMsg("d_endProbeMLH() execution failed!\n");

//	d_check<<<probeX, alignedProbeY>>>(d_probeWavefront);
}





//__host__ void check(real_t* d_array, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
//{
//	d_check1<<<probeX, alignedProbeY>>>(d_array);
//}


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

__host__ void h_normalizeVariProbe(complex_t* d_extramodes, double factor,
					unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY)
{
	h_normalize(d_extramodes, probeX, probeY, alignedProbeY, 1.0/factor);

//	d_check<<<probeX, alignedProbeY>>>(d_extramodes);
}

__host__ void h_preCalillum(complex_t* d_modes, real_t* d_result, real_t* p_object, unsigned int Npos, uint2 Np_o, unsigned int* p_positions_x,
		unsigned int* p_positions_y, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY, unsigned int obalignedProbeY,
		complex_t* d_objet, real_t* d_tmpObjResult)
{

	//d_result=p_obj_proj=aprobe2 probex size
	d_norm2<<<probeX, alignedProbeY>>>(d_modes, d_result, probeX,
			 probeY, alignedProbeY);

	//illum_sum_0{ll} = set_projections(Gzeros(self.Np_o), Garray(aprobe2), 1,ind, cache);
    // Choose a reasonably sized number of threads in each dimension for the block.
    int const threadsPerBlockEachDim =  32;
    // Compute the thread block and grid sizes based on the board dimensions.
    int const blocksPerGrid_M = (probeX + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_N = (probeY + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    int const blocksPerGrid_O = Npos;

    dim3 const dimBlock(blocksPerGrid_M, blocksPerGrid_N, blocksPerGrid_O);
    dim3 const dimThread(threadsPerBlockEachDim, threadsPerBlockEachDim, 1);
    bool isFlat=true;

    addToArray_r<<<dimBlock, dimThread>>>(d_result, p_object, p_positions_x, p_positions_y , probeX, probeY, Np_o.x, Np_o.y, Npos, obalignedProbeY, isFlat);
//    d_check1<<<Np_o.x, obalignedProbeY>>>(p_object);

    // sum2(illum_sum_0{ll}
    real_t resultillum=h_realSum(p_object, Np_o.x, Np_o.y, obalignedProbeY);

    //abs(self.object{ll}).^2
	d_norm2<<<Np_o.x, obalignedProbeY>>>(d_objet, d_tmpObjResult, Np_o.x,
			Np_o.y, obalignedProbeY);
	//abs(self.object{ll}).^2 .* illum_sum_0{ll}
	h_multiplyReal(p_object, d_tmpObjResult, Np_o.x, Np_o.y, obalignedProbeY);
	real_t resultobjillum=h_realSum(d_tmpObjResult, Np_o.x, Np_o.y, obalignedProbeY);

	real_t object_norm=sqrt_real_t(resultobjillum/resultillum);
	h_normalize(d_objet, Np_o.x, Np_o.y, obalignedProbeY, (1.0/object_norm));

//	d_checkcomplex1<<<Np_o.x, obalignedProbeY>>>(d_objet);

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

__host__ void h_initVarProbe(complex_t* d_probe, complex_t* d_tempprobe, complex_t* d_initprobe, unsigned int x, unsigned int y, unsigned int alignedY, float randNo1, float randNo2)
{

	h_shiftFFTtmp(d_probe, d_tempprobe, d_initprobe, x, y, alignedY);

	// TODO Generate random number afterwards
//	float ranNo1=0.0431;
//	float randNo2=-0.4012;

	imshift_fft(d_probe, x, y, alignedY, randNo1, randNo2);

	// Shift the probe

	h_shiftFFTtwo(d_probe, d_tempprobe, x, y, alignedY);

}





#endif /* PROBEKERNELS_CU_ */
