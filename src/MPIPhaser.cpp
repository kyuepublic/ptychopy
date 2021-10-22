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
//Any publication using the package should cite for
//Yue K, Deng J, Jiang Y, Nashed Y, Vine D, Vogt S.
//Ptychopy: GPU framework for ptychographic data analysis.
//X-Ray Nanoimaging: Instruments and Methods V 2021.
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
#include "MPIPhaser.h"

#ifdef HAVE_MPI
//#define MPI_TIMING
#include <diy/decomposition.hpp>
#include <diy/reduce.hpp>
#include <diy/partners/merge.hpp>

#include "IPtychoScanMesh.h"
#include "Sample.h"
#include "Diffractions.h"
#include "Probe.h"
#include "GLResourceManager.h"
#include "utilities.h"
#include "CudaSmartPtr.hpp"

#ifdef USE_SINGLE_PRECISION
#define MPI_COMPLEX_T MPI_COMPLEX
#else
#define MPI_COMPLEX_T MPI_DOUBLE_COMPLEX
#endif

using namespace std;

PartialReconstruction::~PartialReconstruction()
{
	if(neighborParts_d) delete neighborParts_d;
}

void PartialReconstruction::updateBuffer(const diy::Master::ProxyWithLink& cp, void* devPtr)
{
	CudaSmartPtr ptr_d = *static_cast<CudaSmartPtr*>(devPtr);
	ptr_d->getHostPtr<complex_t>(&(data_h[0]));
}

void PartialReconstruction::sendBuffer(const diy::Master::ProxyWithLink& cp, void*)
{
	diy::Link*    l = cp.link();
	for (size_t i=0; i<l->size(); ++i)
		if(cp.gid() != l->target(i).gid)
		{
			if(neighborParts_d==0) //also exchange offsets if it's the first share
			{
				cp.enqueue(l->target(i), bufferParams.offset);
				cp.enqueue(l->target(i), bufferParams.dims);
			}
			cp.enqueue(l->target(i), data_h);
		}
}

void PartialReconstruction::getNeighborParts(const diy::Master::ProxyWithLink& cp, void*)
{
	bool firstShare = false;
	if(neighborParts_d==0)
	{
		int maxDims[4] = {	bufferParams.dims.x,
							bufferParams.dims.y,
							0, 0};
		MPI_Allreduce( maxDims, maxDims+2, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

		neighborParts_d 	= new Cuda3DArray<complex_t>(neighborNum, make_uint2(maxDims[2], maxDims[3]) );
		neighborOffsets_d 	= new Cuda2DArray<uint2>(1, neighborNum);
		neighborDims_d 		= new Cuda2DArray<uint2>(1, neighborNum);
		neighborOffsets_h.clear();
		neighborDims_h.clear();
		firstShare = true;
	}

	vector<int> in;
	vector<complex_t> neighborData_h;

	cp.incoming(in);
	for (size_t i=0; i<in.size(); ++i)
		if(cp.gid() != in[i])
		{
			if(firstShare)
			{
				uint2 nOffset, nDims;
				cp.dequeue(in[i], nOffset);
				cp.dequeue(in[i], nDims);
				neighborOffsets_h.push_back(nOffset);
				neighborDims_h.push_back(nDims);
			}
			cp.dequeue(in[i], neighborData_h);
			Cuda3DElement<complex_t> neighborData_d = neighborParts_d->getAt(i);
			neighborData_d.setFromHost(&(neighborData_h[0]), neighborDims_h[i].x, neighborDims_h[i].y);
		}

	if(firstShare)
	{
		neighborOffsets_d->setFromHost(&(neighborOffsets_h[0]), 1, neighborOffsets_h.size());
		neighborDims_d->setFromHost(&(neighborDims_h[0]), 1, neighborDims_h.size());
	}

	bufferParams.sample->addNeighborSubsamples(neighborParts_d, neighborOffsets_d, neighborDims_d, bufferParams.offset);
}

void stitchPartial(void* b_, const diy::ReduceProxy& rp, const diy::RegularMergePartners& partners)
{
	PartialReconstruction* block = static_cast<PartialReconstruction*>(b_);

	if(rp.in_link().size()>0) //dequeu and merge
	{
		vector<PartialReconstruction> mergeBlocks;
		uint2 maxDims = make_uint2(0,0);

		for (int i=0; i<rp.in_link().size(); ++i)
		{
			int nbr_gid = rp.in_link().target(i).gid;

			if (nbr_gid != rp.gid())
			{
				PartialReconstruction rcvdBlock;

				rp.dequeue(nbr_gid, rcvdBlock.bufferParams.offset);
				rp.dequeue(nbr_gid, rcvdBlock.bufferParams.dims);
				rp.dequeue(nbr_gid, rcvdBlock.data_h);

				rcvdBlock.bufferParams.offset.x = rcvdBlock.bufferParams.offset.x-block->bufferParams.offset.x;
				rcvdBlock.bufferParams.offset.y = rcvdBlock.bufferParams.offset.y-block->bufferParams.offset.y;
				if(rcvdBlock.bufferParams.offset.x+rcvdBlock.bufferParams.dims.x>maxDims.x)
					maxDims.x = rcvdBlock.bufferParams.offset.x+rcvdBlock.bufferParams.dims.x;
				if(rcvdBlock.bufferParams.offset.y+rcvdBlock.bufferParams.dims.y>maxDims.y)
					maxDims.y = rcvdBlock.bufferParams.offset.y+rcvdBlock.bufferParams.dims.y;

				mergeBlocks.push_back(rcvdBlock);
			}
		}

		vector<complex_t> recon(maxDims.x*maxDims.y);
		for(int i=0; i<=mergeBlocks.size(); ++i)
		{
			PartialReconstruction* mergeBlock = (i-1<0)? block : &(mergeBlocks[i-1]);
			uint2 offset = (i-1<0)? make_uint2(0,0) : mergeBlock->bufferParams.offset;

			for(size_t r=0; r<mergeBlock->bufferParams.dims.x; ++r)
			{
				size_t trgtIndex = (offset.x+r)*maxDims.y + offset.y;
				size_t srcIndex = r*mergeBlock->bufferParams.dims.y;
				std::copy(mergeBlock->data_h.begin()+srcIndex, mergeBlock->data_h.begin()+srcIndex+mergeBlock->bufferParams.dims.y,recon.begin()+trgtIndex);
			}
		}
		block->data_h = recon;
		block->bufferParams.dims = maxDims;
	}
	if(rp.out_link().size() > 0)    //enqueue
	{
		if (rp.out_link().target(0).gid != rp.gid())
		{
			rp.enqueue(rp.out_link().target(0), block->bufferParams.offset);
			rp.enqueue(rp.out_link().target(0), block->bufferParams.dims);
			rp.enqueue(rp.out_link().target(0), block->data_h);
		}
	}
	else //done, output
	{
		CudaSmartPtr recon_d = new Cuda2DArray<complex_t>(block->bufferParams.dims.x, block->bufferParams.dims.y);
		recon_d->setFromHost(&(block->data_h[0]), block->bufferParams.dims.x, block->bufferParams.dims.y);
		block->bufferParams.sample->setObjectArray(recon_d);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
MPIPhaser::MPIPhaser() : 	IPhaser(), m_mpiEnvironment(), m_mpiCommunicator(),
							m_diyMaster(m_mpiCommunicator, 1, -1), m_diyDecomposer(0),
							m_scanBounds(0), m_myrank(0), m_procs(1)
{
	m_procs = m_mpiCommunicator.size();
	m_myrank= m_mpiCommunicator.rank();;
	unsigned int GPUNum = GPUQuery::getInstance()->getDeviceCount();
	cudaSetDevice(m_mpiCommunicator.rank()%GPUNum);
}

MPIPhaser::~MPIPhaser()
{
	if(m_scanBounds) 	delete[] m_scanBounds;
	if(m_diyDecomposer) delete m_diyDecomposer;
}

const int* MPIPhaser::getBounds()
{
	if(m_scanBounds==0 && m_procs>1)
	{
		const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
		const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();

		BlockCreator creator(m_diyMaster);
		diy::RoundRobinAssigner assigner(m_procs, m_procs);
		Bounds domain;
		domain.min[0] = domain.min[1] = 0;
		domain.max[0] = eParams->scanDims.x ; domain.max[1] = eParams->scanDims.y;
		Decomposer::BoolVector share_face(2,false);
		Decomposer::BoolVector wrap(2,false);
		Decomposer::CoordinateVector ghosts;
		ghosts.push_back(rParams->halo); ghosts.push_back(rParams->halo);
		m_diyDecomposer = new Decomposer(2, domain, m_procs, share_face, wrap, ghosts);
		m_diyDecomposer->decompose(m_myrank, assigner, creator);

		Bounds bounds;
		m_scanBounds = new int[4];
		m_diyDecomposer->fill_bounds(bounds, m_myrank, true);
		m_scanBounds[0]=bounds.min[0]; m_scanBounds[1]=bounds.max[0];
		m_scanBounds[2]=bounds.min[1]; m_scanBounds[3]=bounds.max[1];
	}
	return m_scanBounds;
}

bool MPIPhaser::init()
{
	if(!IPhaser::init()) return false;
	if(m_procs>1)
	{
		m_recCorner = m_scanMesh->getMeshOffsets();
		m_recSize 	= m_sample->getObjectArrayShape();

		m_myPart = new Cuda2DArray<complex_t>(m_recSize.x, m_recSize.y);
		BufferInfo info(m_recSize, m_recCorner, m_diyDecomposer, m_sample);
		m_diyMaster.foreach(&PartialReconstruction::initBuffer, &info);

		if(m_renderer)
		{
			Decomposer::DivisionsVector coords;
			int2 windowPos;
			m_diyDecomposer->gid_to_coords(m_myrank, coords);
			m_renderer->getWindowSize(&windowPos.x, &windowPos.y);
			m_renderer->setWindowPos(coords[1]*windowPos.x+2, coords[0]*windowPos.y+20);
		}
	}
	return true;
}


real_t MPIPhaser::phaseStep(IPhasingMethod* m, unsigned int i)
{
	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();

	real_t error = IPhaser::phaseStep(m, i);

	if(rParams->shareFrequency>0 && i%rParams->shareFrequency==0 && m_procs>1)
		sharePartialReconstruction();
	return error;
}

void MPIPhaser::postPhase()
{
	m_probe->endModalReconstruction();
	m_scanMesh->clear();

	m_mergeTimer.start();
	if(m_procs>1)
	{
		//Free GPU memory reserved for the patterns
		m_diffractions->clearPatterns();
		mergePartialReconstructions();
	}
	m_mergeTimer.stop();

	m_mpiCommunicator.barrier();
	if(m_procs==1 || (m_procs>1 && m_myrank==0))
	{
		fprintf(stderr,"[%d]\tmerge=%f\t", m_procs, m_mergeTimer.getElapsedTimeInSec());
		IPhaser::postPhase();
	}
}

void MPIPhaser::writeResultsToDisk(int r)
{
	if(m_myrank==0)
		IPhaser::writeResultsToDisk(r);
}

void MPIPhaser::updatePartialReconstruction()
{
	m_sample->extractROI(m_myPart, 0, 0);
	m_diyMaster.foreach(&PartialReconstruction::updateBuffer, &m_myPart);
}

void MPIPhaser::mergePartialReconstructions()
{
	updatePartialReconstruction();

	diy::RegularMergePartners  partners(*m_diyDecomposer, m_procs, true);
	//diy::RegularMergePartners  partners(Decomposer(1, m_diyDecomposer->domain, m_procs), m_procs, true);
	diy::RoundRobinAssigner assigner(m_procs, m_procs);
	diy::reduce(m_diyMaster, assigner, partners, &stitchPartial);
}

void MPIPhaser::sharePartialReconstruction()
{
	updatePartialReconstruction();
	m_diyMaster.foreach(&PartialReconstruction::sendBuffer);
	m_diyMaster.exchange();
	m_diyMaster.foreach(&PartialReconstruction::getNeighborParts);
}

#endif
