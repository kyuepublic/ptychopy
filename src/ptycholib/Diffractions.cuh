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


#ifndef DIFFRACTIONS_CUH_
#define DIFFRACTIONS_CUH_

#include "datatypes.h"

__host__ void h_calculateSTXM(const real_t* d_intensities, real_t* d_output,
				unsigned int scansX, unsigned int scansY, unsigned int scansAlignedY,
				unsigned int intensitiesX, unsigned int intensitiesY, unsigned int intensitiesAlignedY);

__host__ void h_symmetrizeDiffraction(const real_t* d_intensities, real_t* d_output,
				unsigned int dataOffsetX, unsigned int dataOffsetY, unsigned int dataAlignedY,
				unsigned int outOffsetX, unsigned int outOffsetY, unsigned int outAlignedY,
				unsigned int numX, unsigned int numY, cudaStream_t* stream=0);

__host__ void h_preprocessDiffractions(real_t* d_intensities, bool squareRoot, real_t threshold,
				unsigned int intensitiesNum, unsigned int intensitiesX, unsigned int intensitiesY, unsigned int intensitiesAlignedY);

__host__ void h_squareRootDiffractions(real_t* d_intensities,
					unsigned int intensitiesNum, unsigned int intensitiesX, unsigned int intensitiesY, unsigned int intensitiesAlignedY);

__host__ void h_modulus_amplitude(real_t* modF, real_t* aPsi, real_t* result, real_t R_offset, unsigned int x, unsigned int y, unsigned int alignedY);

#endif /* DIFFRACTIONS_CUH_ */
