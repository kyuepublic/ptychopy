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
#ifndef MPIPHASER_H_
#define MPIPHASER_H_

#include "IPhaser.h"
#include "Singleton.h"
#include "CudaSmartPtr.h"
#include "Cuda2DArray.hpp"
#include "Parameters.h"

#ifdef HAVE_MPI
#include <diy/mpi.hpp>
#include <diy/master.hpp>
#include <diy/decomposition.hpp>
#include <iostream>

typedef diy::DiscreteBounds Bounds;
typedef diy::RegularLink<Bounds> RLink;
typedef diy::RegularDecomposer<Bounds> Decomposer;

struct BufferInfo
{
	int maxSize;
	uint2 dims;
	uint2 offset;
	Decomposer* decomposer; //nasty hack for DIY2
	Sample* sample;

	BufferInfo(	uint2 size=make_uint2(0,0),
				uint2 o=make_uint2(0,0),
				Decomposer* d=0, Sample* s=0) : dims(size), offset(o),
												decomposer(d), sample(s)
	{}
};

struct PartialReconstruction
{
	BufferInfo bufferParams;
	std::vector<complex_t> data_h;
	Cuda3DArray<complex_t>* neighborParts_d;
	std::vector<uint2> neighborOffsets_h;
	CudaSmartPtr neighborOffsets_d;
	std::vector<uint2> neighborDims_h;
	CudaSmartPtr neighborDims_d;
	unsigned int neighborNum;

	PartialReconstruction(int n=0): bufferParams(), neighborNum(n), neighborParts_d(0)
	{

	}

	~PartialReconstruction();

	void initBuffer(const diy::Master::ProxyWithLink& cp, void* buffInfo)
	{
		bufferParams = *static_cast<BufferInfo*>(buffInfo);
		data_h.resize(bufferParams.dims.x*bufferParams.dims.y);
	}

	void updateBuffer(const diy::Master::ProxyWithLink& cp, void*);
	void sendBuffer(const diy::Master::ProxyWithLink& cp, void*);
	void getNeighborParts(const diy::Master::ProxyWithLink& cp, void*);
};

struct BlockCreator
{
	diy::Master&  master;

	BlockCreator(diy::Master& m): master(m)
	{

	}
    void  operator()(int gid,                // block global id
                     const Bounds& core,     // block bounds without any ghost added
                     const Bounds& bounds,   // block bounds including any ghost region added
                     const Bounds& domain,   // global data bounds
                     const RLink& link)     // neighborhood
	const
	{
		PartialReconstruction* b = new PartialReconstruction((unsigned int)link.size());
		RLink* l = new RLink(link);
		diy::Master& m = const_cast<diy::Master&>(master);
		m.add(gid, b, l); // add block to the master (mandatory)
	}
};
//

class MPIPhaser: public IPhaser
{
protected:
	diy::mpi::environment  m_mpiEnvironment;
	diy::mpi::communicator m_mpiCommunicator;
	diy::Master m_diyMaster;
	Decomposer* m_diyDecomposer;

	int* m_scanBounds;
	int m_myrank;
	int m_procs;
	uint2 m_recCorner;
	uint2 m_recSize;
	CudaSmartPtr m_myPart;

	Timer m_mergeTimer;

	virtual const int* getBounds();
	virtual real_t phaseStep(IPhasingMethod* m, unsigned int i);
	virtual void postPhase();

	virtual void updatePartialReconstruction();
	virtual void mergePartialReconstructions();
	virtual void sharePartialReconstruction();

public:
	MPIPhaser();
	virtual ~MPIPhaser();

	virtual bool init();
	virtual void writeResultsToDisk(int r=0);
	int getProcs() const {return m_procs;}
	int getRank() const {return m_myrank;}
};
#endif /* HAVE_MPI */

#endif /* MPIPHASER_H_ */
