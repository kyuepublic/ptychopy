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

#include "CartesianScanMesh.h"
#include "Parameters.h"
#include "Cuda2DArray.hpp"
#include "ScanMesh.cuh"

using namespace std;

CartesianScanMesh::CartesianScanMesh(unsigned int xDim, unsigned int yDim, real_t sZx, real_t sZy, unsigned int jitterRadius) :
																IPtychoScanMesh(0, sZx, sZy, jitterRadius)
{
	m_gridDimensions.x = xDim;
	m_gridDimensions.y = yDim;
}

void CartesianScanMesh::generateMesh(const int* bounds)
{
	if(m_positions.empty())
	{
		const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
		const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();

		unsigned int xMin = bounds? bounds[0] : 0;
		unsigned int xMax = bounds? bounds[1] : m_gridDimensions.x;
		unsigned int yMin = bounds? bounds[2] : 0;
		unsigned int yMax = bounds? bounds[3] : m_gridDimensions.y;

		if(!m_scanPositions.isValid())
			m_scanPositions = new Cuda2DArray<float2>(xMax-xMin,yMax-yMin);

		h_generateCartesianMesh(m_scanPositions->getDevicePtr<float2>(), m_scanPositions->getX(), m_scanPositions->getY(),
								m_scanPositions->getAlignedY(), m_stepSizeX, m_stepSizeY, eParams->dx_s, m_jitterRadius,
								rParams->mirrorX, rParams->mirrorY, (m_randStates.isValid())?m_randStates->getDevicePtr<curandState>():0,
								eParams->drift);

		calculateMeshDimensions(rParams->desiredShape);
		m_meshOffsets.x = floor(xMin*(m_stepSizeX/eParams->dx_s));
		m_meshOffsets.y = floor(yMin*(m_stepSizeY/eParams->dx_s));

		const float2* h_scanPositions = m_scanPositions->getHostPtr<float2>();
		for(unsigned int i=xMin; i<xMax; ++i)
			for(unsigned int j=yMin; j<yMax; ++j)
			{
				unsigned int scanIndex = ((i-xMin)*m_scanPositions->getY())+(j-yMin);
				unsigned int fileIndex = (i*m_gridDimensions.y)+j;
				addScanPosition(h_scanPositions[scanIndex], fileIndex, rParams->flipScanAxis);
			}

		delete[] h_scanPositions;
	}
}


/*const float2* CartesianScanMesh::getScanPosition(unsigned int i)	 const
{
	return m_scanPositions->getDevicePtr<float2>()  + ((i/m_gridDimensions.x)*m_scanPositions->getAlignedY())
													+ i%m_gridDimensions.y;
}

const vector<unsigned int>& CartesianScanMesh::list(int* bounds, unsigned int stride)
{
	m_indeces.clear();
	unsigned int xMin = bounds? bounds[0] : 0;
	unsigned int xMax = bounds? bounds[1] : m_gridDimensions.x;
	unsigned int yMin = bounds? bounds[2] : 0;
	unsigned int yMax = bounds? bounds[3] : m_gridDimensions.y;
	stride = (stride>0) ? stride : m_gridDimensions.y;



	return m_indeces;
}*/
