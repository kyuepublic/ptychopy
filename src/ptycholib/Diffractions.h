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


#ifndef DIFFRACTIONS_H_
#define DIFFRACTIONS_H_

#include "DiffractionLoader.h"
#include "IPtychoScanMesh.h"
#include <vector_types.h>
#include <vector>

template<typename T> class Cuda3DArray;
template<typename T> class Cuda3DElement;
class ICuda2DArray;
struct PreprocessingParams;

class Diffractions
{
private:
	Cuda3DArray<real_t>* m_patterns;
	CudaSmartPtr m_beamstopMask;
	std::vector<real_t> m_squaredSums;

public:
	Cuda3DArray<real_t>* img;

public:
	Diffractions();
	~Diffractions();

	void initMem(IPtychoScanMesh* scanMesh, uint2 probeSize);
	int loadarr(double*** diffarr, const ExperimentParams* eParams, const PreprocessingParams* pParams); // Load diff from python numpy array

	void clearPatterns();
	int load(const char*, const std::vector<unsigned int>& indeces, unsigned int fStart, const PreprocessingParams* p);
	void simulate(const std::vector<float2>& scanPositions, uint2 offset,
					Cuda3DElement<complex_t> probeMode, const CudaSmartPtr& objectArray);
	void simulateMLs(const std::vector<float2>& scanPositions, uint2 offset,
					Cuda3DElement<complex_t> probeMode, const CudaSmartPtr& objectArray, float2 minima);
	void dumpSTXM(const ICuda2DArray*, const char*) const;
	void fillSquaredSums();
	const Cuda3DArray<real_t>* getPatterns()	const {return m_patterns;}
	const real_t* getBeamstopMask();
	real_t getSquaredSum(unsigned int i)	const {return i<m_squaredSums.size()?m_squaredSums[i]:1;}
	std::vector<real_t> getTotalSquaredSum() {return m_squaredSums;}

	void get_fourier_error(Cuda3DArray<real_t>* apsi, std::vector<int> ind_read, std::vector<real_t>& fourierErrors);

	void modulus_constraint(Cuda3DArray<real_t>* apsi, std::vector<int> ind_read, std::vector < Cuda3DArray<complex_t>* > psivec,
			 int W=0, int R_offset=1);

	void squaredRoot();
};

#endif /* DIFFRACTIONS_H_ */
