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

#include "ePIE.h"
#include "Cuda2DArray.hpp"
#include "utilities.h"
#include "Sample.h"
#include "Probe.h"
#include "Diffractions.h"
#include "CXMath.h"
#include <algorithm>

using namespace std;

ePIE::ePIE() :	m_psi(0), m_psi_old(0)
{}

ePIE::~ePIE()
{
	if (m_psi)		delete m_psi;
	if (m_psi_old)	delete m_psi_old;
}

real_t ePIE::iteration(Diffractions* diffs, Probe* probe,
						Sample* object, IPtychoScanMesh* scanMesh, std::vector< std::vector<real_t> >& fourierErrors, const std::vector<float2>& scanPositions,
						bool phaseConstraint, bool updateProbe, bool updateProbeModes, unsigned int iter, bool RMS)
{
	uint2 probeSize = probe->getModes()->getDimensions();
	if (!m_psi)
	{
		m_psi = new Cuda3DArray<complex_t>(probe->getModes()->getNum(), probeSize);
		m_psi->setUseAll(false);
	}
	if(!m_psi_old)
	{
		m_psi_old = new Cuda3DArray<complex_t>(probe->getModes()->getNum(), probeSize);
		m_psi_old->setUseAll(false);
	}

	vector<unsigned int> positions(scanPositions.size());
	generate(positions.begin(), positions.end(), Counter(0));

	if (updateProbeModes)
	{
		probe->beginModalReconstruction();
		m_psi_old->setUseAll(true);
		m_psi->setUseAll(true);
	}

	real_t error = 0;

	for (unsigned int s=0; s<scanPositions.size(); ++s)
	{
		int randomIndex = rand() % positions.size();
		unsigned int i = positions[randomIndex];
		positions.erase(positions.begin() + randomIndex);

		//int4 offsets = PhaserUtil::getInstance()->applyProjectionApprox(object->getObjectArray(), probe->getModes(), m_psi_old, scanPositions[i]);
		if(scanPositions[i].x<0 || scanPositions[i].y<0)
		{
			fprintf(stderr, "ePIE::iteration(%f,%f) Runtime Error! Object array too small.\n Try increasing object array size.\n",
							scanPositions[i].x, scanPositions[i].y);
			exit(1);
		}

		object->extractROI(m_psi_old->getAt(0), scanPositions[i].x, scanPositions[i].y);

		CXMath::multiply<complex_t>(m_psi_old, probe->getModes(), m_psi_old, true);
		PhaserUtil::getInstance()->applyModulusConstraint(m_psi_old, m_psi, diffs->getPatterns()->getAt(i), diffs->getBeamstopMask());

		//TODO: Figure out how to do object and probe updates with sub-pixel accuracy
		int offsetX = roundf(scanPositions[i].x);
		int offsetY = roundf(scanPositions[i].y);
		//Update Object
		object->updateObjectEstimate(probe->getModes(), m_psi, m_psi_old, offsetX, offsetY, probe->getMaxIntensity(), phaseConstraint);

		//Update Probe
		if (updateProbe)
			probe->updateProbeEstimate(object->getObjectArray().get(), m_psi, m_psi_old,
										offsetX, offsetY, object->getMaxIntensity());

		if(RMS)
			error += PhaserUtil::getInstance()->calculateER(m_psi_old, diffs->getPatterns()->getAt(i))/diffs->getSquaredSum(i);
	}

	return error/scanPositions.size();
}

void ePIE::endPhasing()
{
	if (m_psi)	m_psi->setUseAll(false);
	if (m_psi_old) m_psi_old->setUseAll(false);
}
