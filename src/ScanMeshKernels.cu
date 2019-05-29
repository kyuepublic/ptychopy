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

#ifndef SCANMESHKERNELS_CU_
#define SCANMESHKERNELS_CU_

#include "ScanMesh.cuh"
#include "utilities.h"


template<int type>
__global__ void d_generateMesh(float2* d_positions, unsigned int meshY, real_t stepSizeX, real_t stepSizeY, real_t one_over_dxs,
								unsigned int jitterRadius, bool mirrorX, bool mirrorY, curandState* randStates, float2 driftCoeff)
{
	unsigned int posIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	curandState localState;

	if(jitterRadius>0)
		localState = randStates[posIndex];

	if(threadIdx.x < meshY)
	{
		float2 jitter;
		jitter.x = (jitterRadius>0) ? jitterRadius - (curand_uniform(&localState) * 2.0f * jitterRadius) : 0;
		jitter.y = (jitterRadius>0) ? jitterRadius - (curand_uniform(&localState) * 2.0f * jitterRadius) : 0;
		
		float2 initialValues;
		if(type == 1) //Cartesian Grid
		{
			float meshIndex = uint2float(blockIdx.x);
			float meshDim 	= uint2float(gridDim.x-1)*0.5f;
			initialValues.x = ( (mirrorX?(meshDim-meshIndex):(meshIndex-meshDim)) * stepSizeX );

			meshIndex 		= uint2float(threadIdx.x);
			meshDim 		= uint2float(meshY-1)*0.5f;
			initialValues.y = ( (mirrorY?(meshDim-meshIndex):(meshIndex-meshDim)) * stepSizeY );
		}
		else if(type == 2) //List
			initialValues = d_positions[posIndex];
		else if(type == 3) //Spiral
		{
			real_t sqrtIndex = sqrt_real_t((real_t)posIndex);
			initialValues.x = sqrtIndex * cos_real_t(4.0*sqrtIndex) * stepSizeX;
			initialValues.y = sqrtIndex * sin_real_t(4.0*sqrtIndex) * stepSizeY;
		}

		initialValues.x += blockIdx.x*driftCoeff.x;
		initialValues.y += threadIdx.x*driftCoeff.y;

		d_positions[posIndex] = make_float2((initialValues.x*one_over_dxs)+ jitter.x,
											(initialValues.y*one_over_dxs)+ jitter.y );
	}

	if(jitterRadius>0)
		randStates[posIndex] = localState;
}

__global__ void d_initRandomStates(curandState *state, unsigned long seed)
{
	unsigned int posIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	/* Each thread gets same seed, a different sequence number,
	no offset */
	curand_init(seed, posIndex, 0, &state[posIndex]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ void h_generateCartesianMesh(float2* d_positions, unsigned int meshX, unsigned int meshY, unsigned int alignedY,
										real_t stepSizeX, real_t stepSizeY, real_t dx_s, unsigned int jitterRadius,
										bool mirrorX, bool mirrorY, curandState* randStates, float2 driftCoeff)
{
	d_generateMesh<1><<<meshX, alignedY>>>(d_positions, meshY, stepSizeX, stepSizeY, 1.0/dx_s, jitterRadius, mirrorX, mirrorY, randStates, driftCoeff);
	cutilCheckMsg("h_generateCartesianMesh() execution failed!\n");
}

__host__ void h_generateListMesh(float2* d_positions, unsigned int meshX, unsigned int meshY, unsigned int alignedY,
										real_t stepSize, real_t dx_s, unsigned int jitterRadius, curandState* randStates, float2 driftCoeff)
{
	d_generateMesh<2><<<meshX, alignedY>>>(d_positions, meshY, stepSize, 0, 1.0/dx_s, jitterRadius, false, false, randStates, driftCoeff);
	cutilCheckMsg("h_generateListMesh() execution failed!\n");
}

__host__ void h_generateSpiralMesh(float2* d_positions, unsigned int meshX, unsigned int meshY, unsigned int alignedY,
										real_t stepSizeX, real_t stepSizeY, real_t dx_s, unsigned int jitterRadius,
										curandState* randStates, float2 driftCoeff)
{
	d_generateMesh<3><<<meshX, alignedY>>>(d_positions, meshY, stepSizeX, stepSizeY, 1.0/dx_s, jitterRadius, false, false, randStates, driftCoeff);
	cutilCheckMsg("h_generateCartesianMesh() execution failed!\n");
}

__host__ void h_initRandomStates(unsigned int meshX, unsigned int alignedY, curandState* devStates)
{ 
    d_initRandomStates<<<meshX, alignedY>>>(devStates, time(NULL));
	cutilCheckMsg("h_initRandomStates() execution failed!\n");
}

#endif /* SCANMESHKERNELS_CU_ */
