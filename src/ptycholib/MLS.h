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

#ifndef MLS_H_
#define MLS_H_

#include "IPhasingMethod.h"

template<typename T> class Cuda3DArray;


class MLS : public IPhasingMethod
{
private:
	Cuda3DArray<complex_t>* m_psi;
	Cuda3DArray<complex_t>* m_psi_old;

	Cuda3DArray<complex_t>* dPO;
	Cuda3DArray<real_t>* AA1;
	Cuda3DArray<complex_t>* AA2;
	Cuda3DArray<real_t>* AA4;
	Cuda3DArray<real_t>* Atb1;
	Cuda3DArray<real_t>* Atb2;
public:
	MLS();
	virtual ~MLS();

	virtual void initMem(IPtychoScanMesh* scanMesh, uint2 probeSize);
	virtual real_t iteration(Diffractions*, Probe*, Sample*, IPtychoScanMesh*, std::vector< std::vector<real_t> >& fourierErrors, const std::vector<float2>&, bool phaseConstraint,
			bool updateProbe, bool updateProbeModes, unsigned int iter, bool RMS=false);

	virtual void endPhasing();
};

#endif /* MLS_H_ */