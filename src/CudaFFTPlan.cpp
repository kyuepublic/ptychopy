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

#include "CudaFFTPlan.h"
#include "utilities.h"
#include "Cuda2DArray.hpp"

using namespace std;


CudaFFTPlan::~CudaFFTPlan()
{
	for(cufftMap::iterator it=m_R2CPlans.begin(); it!=m_R2CPlans.end(); ++it)
		cufftDestroy(it->second);
	m_R2CPlans.clear();

	for(cufftMap::iterator it=m_C2RPlans.begin(); it!=m_C2RPlans.end(); ++it)
		cufftDestroy(it->second);
	m_C2RPlans.clear();

	for(cufftMap::iterator it=m_C2CPlans.begin(); it!=m_C2CPlans.end(); ++it)
		cufftDestroy(it->second);
	m_C2CPlans.clear();
}

cufftHandle CudaFFTPlan::getR2CPlan(const ICuda2DArray* arr, unsigned int batch, bool cache)
{return getPlan(&m_R2CPlans, REAL2COMPLEX, arr->getX(), arr->getY(), arr->getAlignedY(), batch, cache);}

cufftHandle CudaFFTPlan::getC2RPlan(const ICuda2DArray* arr, unsigned int batch, bool cache)
{return getPlan(&m_C2RPlans, COMPLEX2REAL, arr->getX(), arr->getY(), arr->getAlignedY(), batch, cache);}

cufftHandle CudaFFTPlan::getC2CPlan(const ICuda2DArray* arr, unsigned int batch, bool cache)
{return getPlan(&m_C2CPlans, COMPLEX2COMPLEX, arr->getX(), arr->getY(), arr->getAlignedY(), batch, cache);}

cufftHandle CudaFFTPlan::getPlan(cufftMap* activeMap, cufftType fftType, unsigned int fftH,
									unsigned int fftW, unsigned int pitch, unsigned int batch, bool cache)
{
	if(cache)
	{
		fftKey key(fftH,fftW,batch);

		cufftMap::iterator it = activeMap->find(key);
		if(it == activeMap->end())
		{
			cufftHandle fftPlan = createPlan(fftType,fftH,fftW,pitch,batch);
			it = activeMap->insert(pair<fftKey,cufftHandle>(key,fftPlan)).first;
		}
		return it->second;
	}
	else
		return createPlan(fftType,fftH,fftW,pitch,batch);
}

cufftHandle CudaFFTPlan::createPlan(cufftType fftType, unsigned int fftH,unsigned int fftW,
									unsigned int pitch, unsigned int batch)
{
	cufftHandle fftPlan;
	int dims[] = {fftH,fftW};
	int embed[] = {fftH,pitch};

	cufftPlanMany(&fftPlan, 2, dims, embed,1,fftH*pitch, embed,1,fftH*pitch, fftType, batch);
	cutilCheckMsg("CudaFFTPlan::getPlan() FFT Plan creation failed!\n");
	return fftPlan;
}
