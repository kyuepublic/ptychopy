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

#ifndef PROBE_H_
#define PROBE_H_

#include "CudaSmartPtr.h"
//#include "IRenderable.h"
#include "datatypes.h"
#include <vector_types.h>

template<typename T> class Cuda3DArray;

struct curandGenerator_st;

class Probe
{
private:
	Cuda3DArray<complex_t>* m_modes;
	Cuda3DArray<real_t>* m_intensities;
	real_t m_maxIntensity;
    curandGenerator_st* m_randomGenerator;
    bool m_modesInitialized;

    CudaSmartPtr generateRandKernel(unsigned int x, unsigned int y);
	void initProbeModes();

public:
	Probe(unsigned int size, unsigned int modes=1);
	~Probe();

	void clear();
	bool init(const Cuda3DArray<real_t>* diffractions, real_t beamSize, real_t dx_s, const char* filename=0);
	void simulate(real_t beamSize, real_t dx_s, bool addNoise=false);
	void orthogonalize();
	void normalize(CudaSmartPtr d_tmpComplex=CudaSmartPtr());
	real_t getMaxIntensity()				const {return m_maxIntensity;}
	Cuda3DArray<complex_t>* getModes()		const {return m_modes;}
	Cuda3DArray<real_t>* getIntensities()	const {return m_intensities;}
	void updateProbeEstimate(	const ICuda2DArray* object, const Cuda3DArray<complex_t>* psi, 
								const Cuda3DArray<complex_t>* psi_old, unsigned int qx, unsigned int qy,
								real_t objectMaxIntensity);
	void beginModalReconstruction();
	void endModalReconstruction();
	void updateIntensities(bool useSum=false);
	void updateMaxIntensity(bool useSum=false);

	//For rendering purposes
	void fillResources();
	unsigned int getWidth() const;
	unsigned int getHeight()const;
	void toRGBA(float4*,const char*,float,float);
	void toGray(float*,const char*,bool=false);
};	

#endif /* PROBE_H_ */


