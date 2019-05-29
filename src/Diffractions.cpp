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

#include "Diffractions.h"
#include "Diffractions.cuh"
#include "Sample.cuh"
#include "CudaFFTPlan.h"
#include "Cuda2DArray.hpp"
#include "ThreadPool.h"
#include "Parameters.h"
#include "FileManager.h"

#include <vector>

using namespace std;

Diffractions::Diffractions() : m_patterns(0)
{}

Diffractions::~Diffractions()
{clearPatterns();}

void Diffractions::clearPatterns()
{
	if(m_patterns) delete m_patterns;
	m_patterns = 0;
}

void Diffractions::simulate(const vector<float2>& scanPositions, uint2 offset,
							Cuda3DElement<complex_t> probeMode, const CudaSmartPtr& objectArray)
{
	clearPatterns();
	m_patterns= new Cuda3DArray<real_t>(scanPositions.size(), probeMode.getDimensions());

	const complex_t* probeWavefront = probeMode.getDevicePtr();
	CudaSmartPtr psi = new Cuda2DArray<complex_t>(probeMode.getX(), probeMode.getY());
	/*CudaSmartPtr diff = new Cuda2DArray<real_t>(probeMode.getX(), probeMode.getY());
	char fname[1024];*/
	for(unsigned int i=0; i<m_patterns->getNum(); ++i)
	{
		//printf("Simulating diff patt %d (%f,%f)\n", i, scanPositions[i].x, scanPositions[i].y);
		float2 scanPos = make_float2(scanPositions[i].x+offset.x, scanPositions[i].y+offset.y);
		if(scanPos.x<0 || scanPos.y<0)
		{
			fprintf(stderr, "Diffractions::simulate() Runtime Error! Object array too small.\n Try increasing object array size.\n");
			exit(1);
		}

		h_simulatePSI(objectArray->getDevicePtr<complex_t>(), probeWavefront, psi->getDevicePtr<complex_t>(),
					scanPos, psi->getX(), psi->getY(), psi->getAlignedY(),
					objectArray->getX(), objectArray->getY(), objectArray->getAlignedY());

		//PhaserUtil::getInstance()->applyProjectionApprox(objectArray, probeMode, psi, scanPositions[i]);

		COMPLEX2COMPLEX_FFT(FFTPlanner::getInstance()->getC2CPlan(psi.get()), psi->getDevicePtr<complex_t>(),
							psi->getDevicePtr<complex_t>(), CUFFT_FORWARD);
		cutilCheckMsg("Diffractions::simulate() FFT execution failed!\n");


		h_realComplexAbs(psi->getDevicePtr<complex_t>(), m_patterns->getAt(i).getDevicePtr(), psi->getX(), psi->getY(), psi->getAlignedY());
		psi->set(0);

		/*sprintf(fname, "data/sim%06d.bin", i);
		diff->setFromDevice(m_patterns->getAt(i).getDevicePtr(), psi->getX(), psi->getY());
		diff->save<real_t>(fname, true);*/
	}
}

int Diffractions::load(const char* filePattern, const vector<unsigned int>& indeces, unsigned int fStart, const PreprocessingParams* params)
{
	clearPatterns();

	unsigned int diffractionsNum = indeces.size();
	char filePath[1024];
	unsigned int fBins = params->diffsPerFile;

	CudaSmartPtr d_mask;
	if( (unsigned long long)(diffractionsNum*params->symmetric_array_size*params->symmetric_array_size) * sizeof(real_t) >=
		GPUQuery::getInstance()->getGPUAvailableMemory())
		return -1;
	m_patterns= new Cuda3DArray<real_t>(diffractionsNum, make_uint2(params->symmetric_array_size, params->symmetric_array_size));

	for(unsigned int i=0; i<diffractionsNum;++i)
	{
		unsigned int fIndex = indeces[i]/fBins;
		unsigned int dIndex = indeces[i]%fBins;
		sprintf(filePath, filePattern, fIndex+fStart);
		if(!IO::getInstance()->addIOTask(filePath, dIndex, m_patterns->getAt(i)))
			return -2;
	}
	if(params->beamstopMask)
	{
		string fp(filePattern);
		fp = fp.substr(0,fp.find_last_of('/')) + "/beamstopMask.h5";
		m_beamstopMask = new Cuda2DArray<real_t>(params->symmetric_array_size, params->symmetric_array_size);
		if(!IO::getInstance()->addIOTask(fp.c_str(), 0, Cuda3DElement<real_t>(m_beamstopMask->getDevicePtr<real_t>(),
																			m_beamstopMask->getX(),
																			m_beamstopMask->getY())))
			return -3;
	}
	IO::getInstance()->loadData();

	//4- Apply thresholding and square root to all the diffraction patterns at once
	h_preprocessDiffractions(m_patterns->getPtr()->getDevicePtr<real_t>(), params->flags & SQUARE_ROOT,
							params->threshold_raw_data, m_patterns->getNum(), m_patterns->getDimensions().x,
							m_patterns->getDimensions().y, m_patterns->getPtr()->getAlignedY());

	return 0;
}

void Diffractions::fillSquaredSums()
{
	for(unsigned int i=0; i<m_patterns->getNum(); ++i)
	{
		m_squaredSums.push_back(h_realSum(m_patterns->getAt(i).getDevicePtr(), 	0,  m_patterns->getDimensions().x,
																				0, m_patterns->getDimensions().y,
																				m_patterns->getPtr()->getAlignedY()));
	}
}

const real_t* Diffractions::getBeamstopMask()
{return m_beamstopMask.isValid()?m_beamstopMask->getDevicePtr<real_t>():0;}

void Diffractions::dumpSTXM(const ICuda2DArray* scans, const char* filename) const
{
	CudaSmartPtr stxm = new Cuda2DArray<real_t>(scans->getX(),scans->getY());
	h_calculateSTXM(m_patterns->getPtr()->getDevicePtr<real_t>(), stxm->getDevicePtr<real_t>(),
					scans->getX(), scans->getY(), scans->getAlignedY(),
					m_patterns->getDimensions().x, m_patterns->getDimensions().y, m_patterns->getPtr()->getAlignedY());
}
