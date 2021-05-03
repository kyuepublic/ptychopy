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

#ifndef DM_H_
#define DM_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "IPhasingMethod.h"
#include "Sample.cuh"

template<typename T> class Cuda3DArray;

class DM : public IPhasingMethod
{
private:
	thrust::device_vector<float2> m_scanPositions;
	thrust::device_vector<bool> m_flags;
	thrust::device_vector<real_t> m_factors;
	thrust::host_vector<bool> m_cachedFlags;
	thrust::host_vector<real_t> m_cachedFactors;

	Cuda3DArray<complex_t>* m_psi;
	Cuda3DArray<complex_t>* m_psiOld;
	DM_info* m_argsDevicePtr;
	cufftHandle m_forwardFFTPlan;
	cufftHandle m_inverseFFTPlan;

	void checkCache(real_t objMax, real_t probeMax,
					bool phaseConstraint,bool updateProbe, bool updateProbeModes, bool RMS);

public:
	DM();
	virtual ~DM();

	virtual void initMem(IPtychoScanMesh* scanMesh, uint2 probeSize);
	virtual real_t iteration(Diffractions*, Probe*, Sample*, IPtychoScanMesh*,std::vector< std::vector<real_t> >& fourierErrors, const std::vector<float2>&,
								bool phaseConstraint,bool updateProbe,
								bool updateProbeModes, unsigned int iter, bool RMS=false);
	virtual void endPhasing();
};

#endif /* DM_H_ */
