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

#include "IPtychoScanMesh.h"
#include "Cuda2DArray.hpp"
#include "ScanMesh.cuh"

IPtychoScanMesh::IPtychoScanMesh(CudaSmartPtr devPtr, real_t sX, real_t sY, unsigned int r): 	m_scanPositions(devPtr),
																								m_stepSizeX(sX),
																								m_stepSizeY(sY),
																								m_jitterRadius(r)
{
	if(m_jitterRadius>0)
	{
		//Allocation of the CUDA random states
		m_randStates = new Cuda2DArray<curandState>(m_scanPositions->getX(), m_scanPositions->getY());
		h_initRandomStates(m_randStates->getX(), m_randStates->getAlignedY(), m_randStates->getDevicePtr<curandState>());
	}
	m_meshDimensions.x = 0;
	m_meshDimensions.y = 0;
	m_meshOffsets.x = 0;
	m_meshOffsets.y = 0;
	m_maxima.x = 0;
	m_maxima.y = 0;
	m_minima.x = 0;
	m_minima.y = 0;
}

void IPtychoScanMesh::clear()
{
	m_indeces.clear();
	m_positions.clear();
}

unsigned int IPtychoScanMesh::getTotalScanPositionsNum() const
{return m_scanPositions->getNum();}

void IPtychoScanMesh::calculateMeshDimensions(unsigned int probeSize)
{
	m_maxima = h_maxFloat2(m_scanPositions->getDevicePtr<float2>(), m_scanPositions->getX(),
							m_scanPositions->getY(), m_scanPositions->getAlignedY());
	m_minima = h_minFloat2(m_scanPositions->getDevicePtr<float2>(), m_scanPositions->getX(),
							m_scanPositions->getY(), m_scanPositions->getAlignedY());

	m_meshDimensions.x = ceil( (m_maxima.x-m_minima.x) + probeSize);
	m_meshDimensions.y = ceil( (m_maxima.y-m_minima.y) + probeSize);
}


void IPtychoScanMesh::addScanPosition(float2 scanPos, unsigned int index, bool flip)
{
	float2 tempPos = scanPos;
	scanPos.x=(flip?tempPos.y-m_minima.y : tempPos.x-m_minima.x);
	scanPos.y=(flip?tempPos.x-m_minima.x : tempPos.y-m_minima.y);
	m_positions.push_back(scanPos);
	m_indeces.push_back(index);
}
