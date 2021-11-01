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
#ifndef PROBE_CUH_
#define PROBE_CUH_

#include <cuComplex.h>
#include <curand.h>
#include "datatypes.h"

__host__ void h_projectUtoV(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
							unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY);

__host__ complex_t h_Orth(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
							unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY);
__host__ void h_orthro( complex_t* d_u,  complex_t* d_v, complex_t* d_factor, unsigned int index,
		unsigned int modesNum, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY);

__host__ void h_switchprobe( complex_t* d_u,  complex_t* d_v,
		unsigned int modesNum, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY);

__host__ void h_initModesFromBase(complex_t* d_modes, unsigned int modesNum, real_t factor, real_t* d_intensities, 
							unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY);

__host__ void h_initVarModes(complex_t* d_varModes, real_t* d_randarr1, real_t* d_randarr2, unsigned int probeX,
		unsigned int probeY, unsigned int alignedProbeY);
__host__ real_t h_norm2(complex_t* d_extramodes, real_t* d_result, unsigned int probeX,
		unsigned int probeY, unsigned int alignedProbeY);


__host__ void h_simulateProbe(complex_t* d_probeWavefront, real_t* d_phFunc, real_t gaussWindow, real_t beamSize, real_t dx_s,
							unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY);
__host__ void h_initProbe(complex_t* d_probeWavefront, real_t* d_phFunc, real_t gaussWindow, real_t beamSize, real_t dx_s,
							unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY, bool simulated);

__host__ void h_initProbeMLH(complex_t* d_probeWavefront, double dx_fzp, double fzpValue, double D_FZP, double D_H, double lambda, double fl, double beamSize, complex_t* d_pfValue,
							 unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY, bool simulated);

__host__ void h_endProbeMLH(complex_t* d_probeWavefront, complex_t* d_pfValue,
					unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY, bool simulated);

__host__ void check(real_t* d_array, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY);



__host__ void h_normalizeDiffractionIntensities(const real_t* d_intensities, real_t* d_output, unsigned int intensitiesNum,
							unsigned int intensitiesX, unsigned int intensitiesY, unsigned int intensitiesAlignedY);
__host__ void h_probeModalSum(const complex_t* d_probeModes, complex_t* d_output, unsigned int modesNum,
							unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY);

__host__ void h_normalizeVariProbe(complex_t* d_extramodes, double factor,
							unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY);

__host__ void h_preCalillum(complex_t* d_modes, real_t* d_result, real_t* p_object, unsigned int Npos, uint2 Np_o,
		unsigned int* p_positions_x, unsigned int* p_positions_y, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY, unsigned int obalignedProbeY,
		complex_t* d_objet, real_t* d_tmpObjResult);

__host__ void h_updateProbe(complex_t* d_probe, const complex_t* d_objectArray, const complex_t* d_psi, const complex_t* d_psi_old,
							real_t* d_intensities, unsigned int offsetX, unsigned int offsetY, real_t normalizationFactor,
							unsigned int modeNum, unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
							unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY);



__host__ void h_initVarProbe(complex_t* d_probe, complex_t* d_tempprobe, complex_t* d_initprobe, unsigned int x, unsigned int y, unsigned int alignedY, float randNo1, float randNo2);



#endif /* PROBE_CUH_ */

