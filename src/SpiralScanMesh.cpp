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
//Any publication using the package should cite fo
//Yue K, Deng J, Jiang Y, Nashed Y, Vine D, Vogt S.
//Ptychopy: GPU framework for ptychographic data analysis.
//X-Ray Nanoimaging: Instruments and Methods V 2021 .
//International Society for Optics and Photonics.
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
#include "SpiralScanMesh.h"
#include "Parameters.h"
#include "Cuda2DArray.hpp"
#include "ScanMesh.cuh"

SpiralScanMesh::SpiralScanMesh(unsigned int num, real_t sZx, real_t sZy, unsigned int jitterRadius):
				IPtychoScanMesh(new Cuda2DArray<float2>( gh_iDivUp(num, GPUQuery::getInstance()->getGPUMaxThreads()),
														 GPUQuery::getInstance()->getGPUMaxThreads()),
								sZx, sZy, jitterRadius), m_pointsNum(num)
{}

SpiralScanMesh::~SpiralScanMesh()
{}

void SpiralScanMesh::generateMesh(const int* bounds)
{
	if(m_positions.empty())
	{
		const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
		const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();

		unsigned int yMin = bounds? bounds[2] : 0;
		unsigned int yMax = bounds? bounds[3] : m_pointsNum;

		h_generateSpiralMesh(	m_scanPositions->getDevicePtr<float2>(), m_scanPositions->getX(), m_scanPositions->getY(),
								m_scanPositions->getAlignedY(), m_stepSizeX, m_stepSizeY, eParams->dx_s, m_jitterRadius,
								(m_randStates.isValid())?m_randStates->getDevicePtr<curandState>():0, eParams->drift);

		calculateMeshDimensions(rParams->desiredShape);
		//TODO: calculate mesh offsets for DIY initialization
		m_meshOffsets.x = 0;
		m_meshOffsets.y = 0;

		const float2* h_scanPositions 	= m_scanPositions->getHostPtr<float2>();
		for(unsigned int j=yMin; j<yMax; ++j)
			addScanPosition(h_scanPositions[j], j, rParams->flipScanAxis);

		delete[] h_scanPositions;
	}
}

void SpiralScanMesh::generateMeshML(const int* bounds)
{

}

//void SpiralScanMesh::get_close_indices()
//{
////	const PreprocessingParams* pParams = CXParams::getInstance()->getPreprocessingParams();
////	unsigned int grouping = pParams->projectionsNum/100;
////	unsigned int Ngroups=ceil((m_gridDimensions.x*m_gridDimensions.y)/grouping);
//
//}
//
//void SpiralScanMesh::get_nonoverlapping_indices()
//{
//
//
//}
