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

#include "Probe.h"
#include "Cuda2DArray.hpp"
#include "Probe.cuh"
#include "Diffractions.h"
#include "CudaSmartPtr.h"

#include <cmath>

using namespace std;

Probe::Probe(unsigned int size, unsigned int modes) :	m_modes(0),
														m_intensities(0),
														m_maxIntensity(1),
														m_randomGenerator(0),
														m_modesInitialized(false)
														
{
	m_modes = new Cuda3DArray<complex_t>(modes, make_uint2(size,size));
	m_intensities = new Cuda3DArray<real_t>(modes, make_uint2(size,size));
	m_modes->setUseAll(false);
	m_intensities->setUseAll(false);
}

Probe::~Probe()
{
	clear();
	if(m_randomGenerator)
		curandDestroyGenerator(m_randomGenerator);
}

void Probe::clear()
{
	if(m_modes) delete m_modes;
	if(m_intensities) delete m_intensities;
	m_modes = 0;
	m_intensities = 0;
}

CudaSmartPtr Probe::generateRandKernel(unsigned int x, unsigned int y)
{
	//Gaussian Smoothing of a random kernel
	CudaSmartPtr randArray(new Cuda2DArray<real_t>(x, y));

	if(!m_randomGenerator)
	{
		curandCreateGenerator(&m_randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(m_randomGenerator, time(NULL));
	}

	for(unsigned int x=0; x<randArray->getX(); ++x)
#ifdef USE_SINGLE_PRECISION
		curandGenerateUniform(m_randomGenerator, randArray->getDevicePtr<real_t>()+(x*randArray->getAlignedY()), randArray->getY());
#else
		curandGenerateUniformDouble(m_randomGenerator, randArray->getDevicePtr<real_t>()+(x*randArray->getAlignedY()), randArray->getY());
#endif

	PhaserUtil::getInstance()->gaussSmooth(randArray->getDevicePtr<real_t>(), 10, randArray->getX(), randArray->getY());
	h_normalize(randArray->getDevicePtr<real_t>(), randArray->getX(), randArray->getY(), randArray->getAlignedY());

	return randArray;
}

void Probe::orthogonalize()
{
	//The Gram�Schmidt process for orthogonalizing the probe modes
	Cuda3DArray<complex_t> modesCopy(*m_modes);
	CudaSmartPtr temp = new Cuda2DArray<complex_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);
    for(unsigned int i=1; i<m_modes->getNum(); ++i)
		for(int j=i-1; j>=0; --j)
		{
			h_projectUtoV(m_modes->getAt(j).getDevicePtr(), modesCopy.getAt(i).getDevicePtr(), temp->getDevicePtr<complex_t>(),
							m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
			h_subtract(m_modes->getAt(i).getDevicePtr(), temp->getDevicePtr<complex_t>(), m_modes->getAt(i).getDevicePtr(),
							m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
		}
}

void Probe::updateIntensities(bool useSum)
{
	h_realComplexAbs(m_modes->getPtr()->getDevicePtr<complex_t>(), m_intensities->getPtr()->getDevicePtr<real_t>(),
				m_modes->getPtr()->getX(), m_modes->getPtr()->getY(), m_modes->getPtr()->getAlignedY(), true);
	updateMaxIntensity(useSum);
}

void Probe::normalize(CudaSmartPtr d_tmpComplex)
{
	CudaSmartPtr probesWavefront = (m_modes->getNum()==1||(!m_modes->checkUseAll()))  ? m_modes->getPtr() :
																d_tmpComplex.isValid()? d_tmpComplex : 
																new Cuda2DArray<complex_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);

	if(m_modes->checkUseAll() &&  m_modes->getNum() > 1)
		h_probeModalSum(m_modes->getPtr()->getDevicePtr<complex_t>(), probesWavefront->getDevicePtr<complex_t>(), m_modes->getNum(),
						probesWavefront->getX(), probesWavefront->getY(), probesWavefront->getAlignedY());
	h_realComplexAbs(probesWavefront->getDevicePtr<complex_t>(), m_intensities->getPtr()->getDevicePtr<real_t>(),
						probesWavefront->getX(), probesWavefront->getY(), probesWavefront->getAlignedY(), false);
	m_maxIntensity = PhaserUtil::getInstance()->getModalDoubleMax(m_intensities);
	h_normalize(m_modes->getPtr()->getDevicePtr<complex_t>(), m_modes->getPtr()->getX(),
						m_modes->getPtr()->getY(), m_modes->getPtr()->getAlignedY(), 1.0/m_maxIntensity);
	updateIntensities();
}

void Probe::initProbeModes()
{
	if(m_modes->getNum() > 1)
	{
		h_initModesFromBase(m_modes->getPtr()->getDevicePtr<complex_t>(), m_modes->getNum(), 0.05, m_intensities->getPtr()->getDevicePtr<real_t>(),
							m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY()); 
		orthogonalize();
		m_modesInitialized = true;
	}
	updateMaxIntensity();
}

void Probe::simulate(real_t beamSize, real_t dx_s, bool addNoise)
{
	CudaSmartPtr d_func = generateRandKernel(m_modes->getDimensions().x, m_modes->getDimensions().y);
	/*h_simulateProbe(m_modes->getAt(0).getDevicePtr(), d_func->getDevicePtr<real_t>(), m_modes->getDimensions().x/(2.0*2.35),
					beamSize, dx_s, m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());*/
	h_initProbe(m_modes->getAt(0).getDevicePtr(), d_func->getDevicePtr<real_t>(), m_modes->getDimensions().x/(2*2.35),
						beamSize, dx_s, m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY(), true);
	updateIntensities();
}

bool Probe::init(const Cuda3DArray<real_t>* diffractions, real_t beamSize, real_t dx_s, const char* filename)
{
	if(filename)
	{
		CudaSmartPtr d_probeGuess = new Cuda2DArray<complex_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);
		if(!d_probeGuess->load<complex_t>(filename))
		{
			fprintf(stderr,"Probe guess file (%s) not found.\n", filename);
			return false;
		}
		m_modes->getAt(0).setFromDevice(d_probeGuess->getDevicePtr<complex_t>(), d_probeGuess->getX(), d_probeGuess->getY());
		updateIntensities();
	}
	else
	{
		Cuda3DElement<real_t> avdata = m_intensities->getAt(0);

		h_initProbe(m_modes->getAt(0).getDevicePtr(), 0, m_modes->getDimensions().x/(2*2.35), beamSize, dx_s,
					m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY(), false);

		h_normalizeDiffractionIntensities(diffractions->getPtr()->getDevicePtr<real_t>(), avdata.getDevicePtr(),
											diffractions->getNum(), avdata.getX(), avdata.getY(), avdata.getAlignedY());

		PhaserUtil::getInstance()->applyModulusConstraint(m_modes, m_modes, avdata);
		normalize();
	}
	return true;
}

void Probe::toRGBA(float4* out, const char* name, float tf, float ts)
{
	string probeName(name);
	int modeStrPos = probeName.find('P')+1;
	int modeIndex = atoi(probeName.substr(modeStrPos, probeName.length()-modeStrPos+1).c_str());
	real_t maxModeIntensity = (m_modes->checkUseAll()&&m_modes->getNum()>1)?h_realMax(m_intensities->getAt(modeIndex).getDevicePtr(), 
																m_intensities->getAt(modeIndex).getX(), m_intensities->getAt(modeIndex).getY(), 
																m_intensities->getAt(modeIndex).getAlignedY()) : m_maxIntensity;
	h_realToRGBA(m_intensities->getAt(modeIndex).getDevicePtr(), out, m_intensities->getAt(modeIndex).getX(), 
		m_intensities->getAt(modeIndex).getY(), m_intensities->getAt(modeIndex).getAlignedY(), 1.0/maxModeIntensity, tf, ts);

	//m_renderableUpdated = false;
}

void Probe::toGray(float* out, const char* name, bool outAligned)
{
	string probeName(name);
	int modeStrPos = probeName.find('P')+1;
	int modeIndex = atoi(probeName.substr(modeStrPos, probeName.length()-modeStrPos+1).c_str());
	real_t maxModeIntensity = (m_modes->checkUseAll()&&m_modes->getNum()>1)?h_realMax(m_intensities->getAt(modeIndex).getDevicePtr(), 
																m_intensities->getAt(modeIndex).getX(), m_intensities->getAt(modeIndex).getY(), 
																m_intensities->getAt(modeIndex).getAlignedY()) : m_maxIntensity;
	h_realToGray(m_intensities->getAt(modeIndex).getDevicePtr(), out, m_intensities->getAt(modeIndex).getX(),  m_intensities->getAt(modeIndex).getY(),
																m_intensities->getAt(modeIndex).getAlignedY(), 1.0/maxModeIntensity, outAligned);

	//m_renderableUpdated = false;
}

void Probe::updateProbeEstimate(const ICuda2DArray* object, const Cuda3DArray<complex_t>* psi, 
								const Cuda3DArray<complex_t>* psi_old, unsigned int qx, unsigned int qy,
								real_t objectMaxIntensity)
{
	h_updateProbe(m_modes->getPtr()->getDevicePtr<complex_t>(), object->getDevicePtr<complex_t>(), psi->getPtr()->getDevicePtr<complex_t>(),
					psi_old->getPtr()->getDevicePtr<complex_t>(), m_intensities->getPtr()->getDevicePtr<real_t>(), qx, qy, 1.0/objectMaxIntensity,
					m_modes->checkUseAll()?m_modes->getNum():1, psi->getDimensions().x, psi->getDimensions().y, psi->getPtr()->getAlignedY(),
					object->getX(), object->getY(), object->getAlignedY());
	updateMaxIntensity();
}


void Probe::updateMaxIntensity(bool useSum)
{
	m_maxIntensity = useSum?PhaserUtil::getInstance()->getModalDoubleSum(m_intensities) :
							PhaserUtil::getInstance()->getModalDoubleMax(m_intensities);
//	m_renderableUpdated = true;
}

void Probe::beginModalReconstruction() 
{
	m_modes->setUseAll(true);
	m_intensities->setUseAll(true);
	if(!m_modesInitialized)
		initProbeModes();
}

void Probe::endModalReconstruction() 
{
	m_modes->setUseAll(false);
	m_intensities->setUseAll(false);
	m_modesInitialized = false;
}

unsigned int Probe::getWidth() const {return m_modes->getDimensions().y;}
unsigned int Probe::getHeight()const {return m_modes->getDimensions().x;}

void Probe::fillResources()
{
//	for(unsigned int m=0; m<m_modes->getNum(); ++m)
//	{
//		char probeName[255];
//		sprintf(probeName, "|P%d|", m);
//		m_myResources.push_back(Resource(probeName, RAINBOW, this));
//	}
}
