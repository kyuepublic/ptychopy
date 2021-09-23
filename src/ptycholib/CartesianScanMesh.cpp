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

//#define PTS 100000
//#define K 11

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

void CartesianScanMesh::generateMeshML(const int* bounds)
{
	if(m_positions.empty())
	{
		const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
		const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
		const PreprocessingParams* pParams = CXParams::getInstance()->getPreprocessingParams();


		unsigned int xMin = bounds? bounds[0] : 0;
		unsigned int xMax = bounds? bounds[1] : m_gridDimensions.x;
		unsigned int yMin = bounds? bounds[2] : 0;
		unsigned int yMax = bounds? bounds[3] : m_gridDimensions.y;

		if(!m_scanPositions.isValid())
			m_scanPositions = new Cuda2DArray<float2>(xMax-xMin,yMax-yMin);


		// Change from eParams->dx_s to eParams->dx
		h_generateCartesianMesh(m_scanPositions->getDevicePtr<float2>(), m_scanPositions->getX(), m_scanPositions->getY(),
								m_scanPositions->getAlignedY(), m_stepSizeX, m_stepSizeY, eParams->dx, m_jitterRadius,
								rParams->mirrorX, rParams->mirrorY, (m_randStates.isValid())?m_randStates->getDevicePtr<curandState>():0,
								eParams->drift);


		calculateMeshDimensions(rParams->desiredShape);

		float extra = 0.2;
		float2 Np_o= make_float2(1e5f, 1e5f);
		int Npos=m_scanPositions->getX()*m_scanPositions->getY();
		float2 Np_p = make_float2(rParams->desiredShape, rParams->desiredShape);
		float2 ceilvar=make_float2(ceil(Np_o.x*1.0/2-Np_p.x*1.0/2), ceil(Np_o.y*1.0/2-Np_p.y*1.0/2));

		float2 resprobex;
		resprobex.x=(m_maxima.x+m_minima.x)/2;
		resprobex.y=(m_maxima.y+m_minima.y)/2;
		unsigned int oROIminx=Np_o.x;
		unsigned int oROIminy=Np_o.y;
		unsigned int oROImaxx=0;
		unsigned int oROImaxy=0;
		unsigned int tmpminx=0;
		unsigned int tmpminy=0;
		unsigned int tmpmaxx=0;
		unsigned int tmpmaxy=0;
		unsigned int tmpx=0;
		unsigned int tmpy=0;

		m_meshOffsets.x = floor(xMin*(m_stepSizeX/eParams->dx_s));
		m_meshOffsets.y = floor(yMin*(m_stepSizeY/eParams->dx_s));

		const float2* h_scanPositions = m_scanPositions->getHostPtr<float2>();

		for(unsigned int i=xMin; i<xMax; ++i)
			for(unsigned int j=yMin; j<yMax; ++j)
			{
				unsigned int scanIndex = ((i-xMin)*m_scanPositions->getY())+(j-yMin);
				unsigned int fileIndex = (i*m_gridDimensions.y)+j;
				addScanPositionMLs(h_scanPositions[scanIndex], fileIndex, rParams->flipScanAxis);

				tmpx=round(h_scanPositions[scanIndex].y-resprobex.y+ceilvar.x);
				tmpminx=tmpx-1;
				tmpmaxx=tmpx+Np_p.x-2;

				if(oROIminx>tmpminx)
					oROIminx=tmpminx;
				if(oROImaxx<tmpmaxx)
					oROImaxx=tmpmaxx;

				tmpy=round(h_scanPositions[scanIndex].x-resprobex.x+ceilvar.y);
				tmpminy=tmpy-1;
				tmpmaxy=tmpy+Np_p.y-2;

				if(oROIminy>tmpminy)
					oROIminy=tmpminy;
				if(oROImaxy<tmpmaxy)
					oROImaxy=tmpmaxy;
			}

		m_Np_o_new=make_uint2(ceil((oROImaxx-oROIminx)*(1+extra)),ceil((oROImaxy-oROIminy)*(1+extra)));

		delete[] h_scanPositions;

	}
}



