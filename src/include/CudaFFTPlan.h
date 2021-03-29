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

#ifndef CUDAFFTPLAN_H_
#define CUDAFFTPLAN_H_

#include <map>
#include <cufft.h>
#include <cufftXt.h>
#include "datatypes.h"
#include "Singleton.h"

#ifdef USE_SINGLE_PRECISION
	#define REAL2COMPLEX CUFFT_R2C
	#define COMPLEX2REAL CUFFT_C2R
	#define COMPLEX2COMPLEX CUFFT_C2C

	#define REAL2COMPLEX_FFT cufftExecR2C
	#define COMPLEX2REAL_FFT cufftExecC2R
	#define COMPLEX2COMPLEX_FFT cufftExecC2C

	#define REAL_LOAD_CALLBACK cufftCallbackLoadR
	#define COMPLEX_LOAD_CALLBACK cufftCallbackLoadC
	#define REAL_STORE_CALLBACK cufftCallbackStoreR
	#define COMPLEX_STORE_CALLBACK cufftCallbackStoreC

	#define CB_LD_REAL CUFFT_CB_LD_REAL
	#define CB_LD_COMPLEX CUFFT_CB_LD_COMPLEX
	#define CB_ST_REAL CUFFT_CB_ST_REAL
	#define CB_ST_COMPLEX CUFFT_CB_ST_COMPLEX
#else
	#define REAL2COMPLEX CUFFT_D2Z
	#define COMPLEX2REAL CUFFT_Z2D
	#define COMPLEX2COMPLEX CUFFT_Z2Z

	#define REAL2COMPLEX_FFT cufftExecD2Z
	#define COMPLEX2REAL_FFT cufftExecZ2D
	#define COMPLEX2COMPLEX_FFT cufftExecZ2Z

	#define REAL_LOAD_CALLBACK cufftCallbackLoadD
	#define COMPLEX_LOAD_CALLBACK cufftCallbackLoadZ
	#define REAL_STORE_CALLBACK cufftCallbackStoreD
	#define COMPLEX_STORE_CALLBACK cufftCallbackStoreZ

	#define CB_LD_REAL CUFFT_CB_LD_REAL_DOUBLE
	#define CB_LD_COMPLEX CUFFT_CB_LD_COMPLEX_DOUBLE
	#define CB_ST_REAL CUFFT_CB_ST_REAL_DOUBLE
	#define CB_ST_COMPLEX CUFFT_CB_ST_COMPLEX_DOUBLE
#endif

class ICuda2DArray;

struct fftKey
{
	std::pair<unsigned int, unsigned int> fftDims;
	unsigned int batchNum;

	fftKey() : fftDims(0,0), batchNum(0)
	{}
	fftKey(unsigned int x, unsigned int y, unsigned int b) : fftDims(x,y), batchNum(b)
	{}

	bool operator<(const fftKey& param) const
	{return batchNum<param.batchNum || (batchNum==param.batchNum&&fftDims<param.fftDims);}
};

typedef std::map<fftKey,cufftHandle> cufftMap;

class CudaFFTPlan
{
private:
	cufftMap m_R2CPlans;
	cufftMap m_C2RPlans;
	cufftMap m_C2CPlans;

	cufftHandle createPlan(cufftType fftType, unsigned int fftH,unsigned int fftW,
						unsigned int pitch, unsigned int batch);
	cufftHandle getPlan(cufftMap* activeMap, cufftType fftType, unsigned int fftH,
						unsigned int fftW, unsigned int pitch, unsigned int batch, bool cache);
public:
	CudaFFTPlan(){}
	~CudaFFTPlan();

	cufftHandle getR2CPlan(const ICuda2DArray* arr, unsigned int batch=1, bool cache=true);
	cufftHandle getC2RPlan(const ICuda2DArray* arr, unsigned int batch=1, bool cache=true);
	cufftHandle getC2CPlan(const ICuda2DArray* arr, unsigned int batch=1, bool cache=true);
};

typedef Singleton<CudaFFTPlan> FFTPlanner;

#endif /* CUDAFFTPLAN_H_ */
