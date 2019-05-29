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

#include "ListScanMesh.h"
#include "Parameters.h"
#include "Cuda2DArray.hpp"
#include "ScanMesh.cuh"

using namespace std;

ListScanMesh::ListScanMesh(const char* fname,unsigned int length, real_t sZ, unsigned int jitterRadius) :
			IPtychoScanMesh(new Cuda2DArray<float2>( gh_iDivUp(length, GPUQuery::getInstance()->getGPUMaxThreads()),
													GPUQuery::getInstance()->getGPUMaxThreads()),
							sZ, sZ, jitterRadius),
			m_length(length)
{loadList(fname);}

void ListScanMesh::loadList(const char* fname)
{
	std::ifstream infile(fname);
	if(!infile.is_open())
	{
		fprintf(stderr, "Failed to load raster positions file (%s)\n", fname);
		exit(1);
	}

	float2* h_array =  m_scanPositions->getHostPtr<float2>();
	char comma;
	for(unsigned int i=0; i<m_length; ++i)
	{
		infile >> h_array[i].x;
		infile >> comma;
		if(comma != ',')
		{
			fprintf(stderr, "Invalid raster positions file format [%s]!\nUse\tx1,y1\n\tx2,y2\n\t...\n\txn,yn\n", fname);
			exit(1);
		}
		infile >> h_array[i].y;
	}
	infile.close();
	m_scanPositions->setFromHost<float2>(h_array, m_scanPositions->getX(), m_scanPositions->getY());
	delete [] h_array;
}

void ListScanMesh::generateMesh(const int* bounds)
{
	if(m_positions.empty())
	{
		const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
		const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();

		unsigned int yMin = bounds? bounds[2] : 0;
		unsigned int yMax = bounds? bounds[3] : m_length;

		h_generateListMesh(	m_scanPositions->getDevicePtr<float2>(), m_scanPositions->getX(), m_scanPositions->getY(),
							m_scanPositions->getAlignedY(), m_stepSizeX, eParams->dx_s,  m_jitterRadius,
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
