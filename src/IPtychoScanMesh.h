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

#ifndef IPTYCHOSCANMESH_H_
#define IPTYCHOSCANMESH_H_

#include "CudaSmartPtr.h"
#include "datatypes.h"
#include <vector>
#include <vector_types.h>

class IPtychoScanMesh
{
protected:
	CudaSmartPtr m_scanPositions;
	CudaSmartPtr m_randStates;
	real_t m_stepSizeX;
	real_t m_stepSizeY;
	unsigned int m_jitterRadius;
	uint2 m_meshDimensions;
	uint2 m_meshOffsets;
	float2 m_maxima;
	float2 m_minima;

	std::vector<unsigned int> m_indeces;
	std::vector<float2> m_positions;

	virtual void calculateMeshDimensions(unsigned int);
	virtual void addScanPosition(float2, unsigned int, bool flip=false);

public:
	IPtychoScanMesh(CudaSmartPtr devPtr, real_t sX, real_t sY, unsigned int=0);
	virtual ~IPtychoScanMesh(){}

	virtual void generateMesh(const int* bounds=0) = 0;

	virtual uint2 getMeshDimensions() 						const {return m_meshDimensions;}
	virtual uint2 getMeshOffsets()	 						const {return m_meshOffsets;}
	virtual float2 getMaxima()	 							const {return m_maxima;}
	virtual float2 getMinima()	 							const {return m_minima;}
	virtual unsigned int getScanPositionsNum()				const {return m_positions.size();}
	virtual unsigned int getTotalScanPositionsNum()			const;
	virtual const std::vector<float2>& getScanPositions() 	const {return m_positions;}
	virtual const std::vector<unsigned int>& list() 		const {return m_indeces;}
	//virtual void setScanPositions(const CudaSmartPtr& positions)  {m_scanPositions = positions;}

	void clear();
};

#endif /* IPTYCHOSCANMESH_H_ */
