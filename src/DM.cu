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

#include "DM.h"
#include "Sample.h"
#include "Probe.h"
#include "Diffractions.h"

#define EPS 1e-3

DM::DM(): m_psi(0), m_psiOld(0), m_argsDevicePtr(0), m_forwardFFTPlan(0), m_inverseFFTPlan(0)
{}

DM::~DM()
{endPhasing();}

void DM::checkCache(real_t objMax, real_t probeMax,
		bool phaseConstraint,bool updateProbe, bool updateProbeModes, bool RMS)
{
	bool passedFlags[3] = {phaseConstraint, updateProbe, updateProbeModes};
	for(size_t i=0; i<m_cachedFlags.size();++i)
		if(m_cachedFlags[i]!=passedFlags[i])
		{
			m_cachedFlags[i]=passedFlags[i];
			m_flags[i] = m_cachedFlags[i];
		}
	real_t passedFactors[2] = {1.0/objMax, 1.0/probeMax};
	for(size_t i=0; i<m_cachedFactors.size();++i)
	{
		if(fabs(m_cachedFactors[i]-passedFactors[i])>EPS)
		{
			m_cachedFactors[i]=passedFactors[i];
			m_factors[i] = m_cachedFactors[i];
		}
	}
}

void DM::initMem(IPtychoScanMesh* scanMesh, uint2 probeSize)
{

}

real_t DM::iteration(Diffractions* diffs, Probe* probe,
						Sample* object, IPtychoScanMesh* scanMesh, std::vector< std::vector<real_t> >& fourierErrors, const std::vector<float2>& scanPositions,
						bool phaseConstraint, bool updateProbe, bool updateProbeModes, unsigned int iter, bool RMS)
{
	CudaSmartPtr objectArray = object->getObjectArray();
	Cuda3DArray<complex_t>* probeModes = probe->getModes();
	const Cuda3DArray<real_t>* diffPatterns = diffs->getPatterns();

	if (m_scanPositions.empty())
	{
		m_scanPositions = scanPositions;
		m_cachedFlags.push_back(phaseConstraint);m_cachedFlags.push_back(updateProbe);m_cachedFlags.push_back(updateProbeModes);
		m_flags = m_cachedFlags;
		m_cachedFactors.push_back(1.0/object->getMaxIntensity());m_cachedFactors.push_back(1.0/probe->getMaxIntensity());
		m_factors = m_cachedFactors;
		m_psi = new Cuda3DArray<complex_t>(diffPatterns->getNum(), diffPatterns->getDimensions());
		m_psi->setUseAll(true);
		m_psiOld = new Cuda3DArray<complex_t>(diffPatterns->getNum(), diffPatterns->getDimensions());
		m_psiOld->setUseAll(true);

		DM_info h_args(objectArray->getDevicePtr<complex_t>(), probeModes->getPtr()->getDevicePtr<complex_t>(), m_psiOld->getPtr()->getDevicePtr<complex_t>(),
						diffPatterns->getPtr()->getDevicePtr<real_t>(), thrust::raw_pointer_cast(m_scanPositions.data()),
						thrust::raw_pointer_cast(m_flags.data()), thrust::raw_pointer_cast(m_factors.data()),
						diffPatterns->getNum(), probeModes->getNum(),
						probeModes->getDimensions().x, probeModes->getDimensions().y, probeModes->getPtr()->getAlignedY(),
						objectArray->getX(), objectArray->getY(), objectArray->getAlignedY());

		cudaMalloc((void **)&m_argsDevicePtr, sizeof(DM_info));
		cutilCheckMsg("DM::iteration::cudaMalloc() failed!");
		cudaMemcpy(m_argsDevicePtr, &h_args, sizeof(DM_info), cudaMemcpyHostToDevice);
		cutilCheckMsg("DM::iteration::cudaMemcpy() failed!");

		//TODO: Add probe modes
		Cuda3DElement<complex_t> firstMode = probeModes->getAt(0);
		m_forwardFFTPlan = FFTPlanner::getInstance()->getC2CPlan(&firstMode, diffPatterns->getNum(), false);
		m_inverseFFTPlan = FFTPlanner::getInstance()->getC2CPlan(&firstMode, diffPatterns->getNum(), false);
	}

	checkCache(object->getMaxIntensity(), probe->getMaxIntensity(), phaseConstraint, updateProbe, updateProbeModes, RMS);
	h_forwardPropagate(m_forwardFFTPlan, m_argsDevicePtr, m_psi->getPtr()->getDevicePtr<complex_t>(), phaseConstraint);
	h_backPropagate(m_inverseFFTPlan, m_argsDevicePtr, m_psi->getPtr()->getDevicePtr<complex_t>());
	//h_constrainObject(objectArray->getDevicePtr<complex_t>(), phaseConstraint, objectArray->getX(), objectArray->getY(), objectArray->getAlignedY());
	object->updateIntensities(true);
	probe->updateIntensities(true);

	return 0;
}

void DM::endPhasing()
{
	m_scanPositions.clear();
	m_flags.clear();
	m_cachedFlags.clear();
	m_factors.clear();
	m_cachedFactors.clear();
	if(m_psi) delete m_psi;
	if(m_psiOld) delete m_psiOld;
	if(m_argsDevicePtr)
	{
		cudaFree(m_argsDevicePtr);
		cutilCheckMsg("cudaFree() failed!");
	}
	if(m_forwardFFTPlan) cufftDestroy(m_forwardFFTPlan);
	if(m_inverseFFTPlan) cufftDestroy(m_inverseFFTPlan);
}


