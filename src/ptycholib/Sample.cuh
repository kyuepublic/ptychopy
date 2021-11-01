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
#ifndef SAMPLE_CUH_
#define SAMPLE_CUH_

#include "CudaFFTPlan.h"
#include <curand.h>
#include <curand_kernel.h>


__host__ void h_initRandomStatesSample(unsigned int, unsigned int, curandState*);


__host__ void h_bindObjArrayToTex(complex_t* d_objectArray, unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY);
__host__ void h_unbindObjArrayTex();
__host__ void h_applyHammingToSample(complex_t* d_objectArray, real_t* d_sampleR, real_t* d_sampleI,
								unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY);
__host__ void h_applyHammingToSample(const complex_t* d_sample, complex_t* d_output,
								unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY);
__host__ void h_simulatePSI(const complex_t* d_objectArray, const complex_t* d_probe, complex_t* d_psi, float2 scanPos,
								unsigned int psiX, unsigned int psiY, unsigned int alignedPsiY, unsigned int objectArrayX,
								unsigned int objectArrayY, unsigned int alignedObjectArrayY);
__host__ void h_getPhaseGradient(const complex_t* d_sample, complex_t* d_output,
								unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY, bool applyHamming);
__host__ void h_getMagnitudeGradient(const complex_t* d_sample, complex_t* d_output,
								unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY, bool applyHamming);
__host__ void h_extractObjectArray(complex_t* d_objectArray, complex_t* d_output, float offsetX, float offsetY,
								unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY);
__host__ void h_getObjectIntensities(complex_t* d_objectArray, real_t* d_output, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY, bool squared=false);
__host__ void h_getObjectPhase(complex_t* d_objectArray, real_t* d_output, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY);
__host__ void h_updateObjectArray(complex_t* d_objectArray, const complex_t* d_probe, const complex_t* d_psi, const complex_t* d_psi_old,
								real_t* d_intensities, unsigned int offsetX, unsigned int offsetY, real_t normalizationFactor,
								unsigned int modeNum, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY, bool phaseConstraint);
__host__ void h_postprocess(complex_t* d_objectArray, real_t thresholdFactor, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY);
__host__ void h_setObjectArray(complex_t* d_objectArray, const complex_t* d_roi,  unsigned int roiX, unsigned int roiY, unsigned int alignedROIY,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY,
								unsigned int roiOffsetX=0, unsigned int roiOffsetY=0);

__host__ void h_initRandObjectArray(complex_t* d_array, real_t* d_randarr1, real_t* d_randarr2, unsigned int sampleX,
		unsigned int sampleY, unsigned int alignedSampleY);

__host__ void h_updateStencils(char* d_myStencil, const uint2* d_nOffset, const uint2* d_nDims,
								uint2 myOffset, unsigned int neighborNum, unsigned int objectArrayX,
								unsigned int objectArrayY, unsigned int alignedObjectArrayY);
__host__ void h_addSubsamples(complex_t* d_objectArray, const complex_t* d_neighbors, const uint2* d_nOffset, const uint2* d_nDims,
								unsigned int sampleX, unsigned int alignedSampleY, uint2 myOffset, unsigned int neighborNum,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY);
__host__ void h_mergeObjectArray(complex_t* d_objectArray, const complex_t* d_s1, const complex_t* d_s2,
								unsigned int alignedS1Y, unsigned int alignedS2Y, unsigned int s2OffsetX, unsigned int s2OffsetY,
								unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY, unsigned int px, unsigned int py);
__host__ void h_crossPowerSpectrum(const complex_t* d_F1, const complex_t* d_F2, complex_t* d_cps,
								unsigned int x, unsigned int y, unsigned int alignedY);

struct DM_info
{
	complex_t* d_objectArray;
	complex_t* d_probe;
	complex_t* d_psi;
	const real_t* d_intensities;
	const float2* d_scanPos;
	const bool* d_flags;
	const real_t* d_normalizingFactors;
	unsigned int intensitiesNum;
	unsigned int modeNum;
	unsigned int probeX;
	unsigned int probeY;
	unsigned int alignedProbeY;
	unsigned int objectArrayX;
	unsigned int objectArrayY;
	unsigned int alignedObjectArrayY;

	DM_info(complex_t* O, complex_t* P, complex_t* PSI, const real_t* I, const float2* R, const bool* F, const real_t* NF,
				unsigned int N, unsigned int K, unsigned int pX, unsigned int pY, unsigned int pAY,
				unsigned int oX, unsigned int oY, unsigned int oAY): 	d_objectArray(O), d_probe(P), d_psi(PSI), d_intensities(I),
																		d_scanPos(R), d_flags(F), d_normalizingFactors(NF),
																		intensitiesNum(N), modeNum(K),
																		probeX(pX), probeY(pY), alignedProbeY(pAY),
																		objectArrayX(oX), objectArrayY(oY), alignedObjectArrayY(oAY)
	{}
};

__host__ void h_forwardPropagate(cufftHandle fftPlan, DM_info* args, complex_t* d_psi, bool phaseConstraint);
__host__ void h_backPropagate(cufftHandle fftPlan, DM_info* args, complex_t* d_psi);
__host__ void h_constrainObject(complex_t* d_objectArray, bool phaseConstraint, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY);

#endif /* SAMPLE_CUH_ */
