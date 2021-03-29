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

#include "IPhaser.h"
#include "ePIE.h"
#include "DM.h"
#include "MLS.h"
#include "CartesianScanMesh.h"
#include "ListScanMesh.h"
#include "SpiralScanMesh.h"
#include "Sample.h"
#include "Probe.h"
#include "Diffractions.h"
#include "GLResourceManager.h"
#include "RenderServer.h"
#include "Cuda2DArray.hpp"
#include <string>
#include <stdio.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

using namespace std;

IPhaser::IPhaser() : m_diffractions(0), m_probe(0), m_sample(0), m_scanMesh(0), m_renderer(0)
{}

IPhaser::~IPhaser()
{
	if(m_diffractions) delete m_diffractions;
	if(m_probe) delete m_probe;
	if(m_sample) delete m_sample;
	if(m_scanMesh) delete m_scanMesh;
	if(m_renderer) delete m_renderer;

	for(size_t mIndex=0; mIndex<m_phasingMethods.size(); ++mIndex)
		delete m_phasingMethods[mIndex].method;
	m_phasingMethods.clear();
}

bool IPhaser::initSample(uint2 meshDimensions)
{
	if(m_sample==0)
	{
		const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();

		if(rParams->method.compare("MLs")==0)
		{
			meshDimensions=m_scanMesh->getNPONEW();
		}
		else
		{
			meshDimensions=m_scanMesh->getMeshDimensions();
		}

		m_sample = new Sample(meshDimensions.x, meshDimensions.y);
		if(rParams->simulated)
		{
			char magFileName[255];
			char phaseFileName[255];
			sprintf(magFileName, "data/lena%d.pgm", rParams->desiredShape);
			sprintf(phaseFileName, "data/baboon%d.pgm", rParams->desiredShape);
			if(!m_sample->loadFromFile(magFileName, phaseFileName, 0.9f))
				return false;
		}
	}
	return true;
}

bool IPhaser::initProbe()
{
	if(m_probe==0)
	{
		const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
		const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
		m_probe = new Probe(rParams->desiredShape, rParams->probeModes);
		if(rParams->simulated)
			if (rParams->probeGuess.empty())
				m_probe->simulate(eParams->beamSize, eParams->dx_s);
			else
				m_probe->init(0, eParams->beamSize, eParams->dx_s, rParams->probeGuess.c_str());

		if(rParams->method.compare("MLs")==0)
		{
			m_probe->initVarModes();
		}

	}
	return true;
}

bool IPhaser::initScanMesh()
{
	if(m_scanMesh==0)
	{

		const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
		const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
		if(!rParams->positionsFilename.empty())
			m_scanMesh = new ListScanMesh(rParams->positionsFilename.c_str(), eParams->scanDims.x*eParams->scanDims.y,
											eParams->stepSize.x, rParams->jitterRadius);
		else if(eParams->spiralScan)
			m_scanMesh = new SpiralScanMesh(eParams->scanDims.x*eParams->scanDims.y, eParams->stepSize.x,
											eParams->stepSize.y, rParams->jitterRadius);
		else
			m_scanMesh = new CartesianScanMesh(eParams->scanDims.x, eParams->scanDims.y, eParams->stepSize.x,
											eParams->stepSize.y, rParams->jitterRadius);

		if(rParams->method.compare("MLs")==0)
		{
//			std::cout<<rParams->method<<rParams->method.compare("MLs")<<std::endl;
			m_scanMesh->generateMeshML(this->getBounds());
	//		m_scanMesh->get_close_indices(); //MLC
			m_scanMesh->get_nonoverlapping_indices(); // MLs
			m_scanMesh->initoROIvec();
			m_scanMesh->precalculate_ROI();
		}
		else
		{
			m_scanMesh->generateMesh(this->getBounds());
		}



	}
	return true;
}

bool IPhaser::initDiffractions()
{
	if(m_diffractions==0)
	{
		const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
		const PreprocessingParams* pParams = CXParams::getInstance()->getPreprocessingParams();
		const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();

		m_diffractions = new Diffractions;
		if(rParams->simulated)
			m_diffractions->simulate(m_scanMesh->getScanPositions(), m_scanMesh->getMeshOffsets(),
									m_probe->getModes()->getAt(0), m_sample->getObjectArray());
		else
		{
			string fpStr(pParams->dataFilePattern);
			if(fpStr.empty())
			{
				fprintf(stderr,"Could not find data path parameter!\nDid you forget to provide `-fp` argument?\n");
				return false;
			}
			fpStr.replace(fpStr.find('#'), 1, "%");
			switch(m_diffractions->load(fpStr.c_str(), m_scanMesh->list(), pParams->fileStartIndex, pParams))
			{
			case 0: break;
			case -1: fprintf(stderr,"Not enough GPU memory to store all diffraction pattern data.\n"); return false;
			case -2: fprintf(stderr,"Failed to initialize IO thread.\n"); return false;
			case -3: fprintf(stderr,"Failed to initialize IO thread for beamstop mask file.\n"); return false;
			default: break;
			}
		}
		//m_diffractions->getPatterns()->getPtr()->save<real_t>("data/diffs000.bin", true);
	}
	return true;
}

bool IPhaser::init()
{
	srand(time(NULL));

	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();

	if(!this->initScanMesh())								return false;
	#ifdef DEBUG_MSGS
//		fprintf(stderr,"ScanMesh(%d,%d) initialized...\n", m_scanMesh->getMeshDimensions().x, m_scanMesh->getMeshDimensions().y);
		fprintf(stderr,"ScanMesh(%d,%d) initialized...\n", m_scanMesh->getNPONEW().x, m_scanMesh->getNPONEW().y);
	#endif
	if(!this->initSample(m_scanMesh->getNPONEW()))	return false;
	#ifdef DEBUG_MSGS
		fprintf(stderr,"Sample initialized...\n");
	#endif
	if(!this->initProbe())	 								return false;
	#ifdef DEBUG_MSGS
		fprintf(stderr,"Probe initialized...\n");
	#endif
	m_ioTimer.start();
	if(!this->initDiffractions())							return false;
	m_ioTimer.stop();
	#ifdef DEBUG_MSGS
		fprintf(stderr,"Diffraction patterns loaded (%fs)...\n", m_ioTimer.getElapsedTimeInSec());
	#endif

//	if(CXParams::getInstance()->renderResults())
//	{
//		if(rParams->wsServerUrl.empty())
//			m_renderer = new GLResourceManager();
//		else
//			m_renderer = new RenderServer(rParams->wsServerUrl.c_str());
//		m_renderer->init();
//		m_renderer->addResources(m_sample);
//		m_renderer->addResources(m_probe);
//	}

	return true;
}

void IPhaser::addPhasingMethod(const char* mName, unsigned int i)
{
	if(string(mName) == "ePIE")
		m_phasingMethods.push_back(CXPhasing(mName, new ePIE, i));
	else if(string(mName) == "DM")
		m_phasingMethods.push_back(CXPhasing(mName, new DM, i));
	else if(string(mName) == "MLs")
		m_phasingMethods.push_back(CXPhasing(mName, new MLS, i));
	else
		fprintf(stderr,"Erorr! Reconstruction algorithm [%s] not implemented.\n", mName);
}

void IPhaser::prePhase()
{
	//Clear objects for a new phasing round
	const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
	double dx_recon = eParams->dx_s / rParams->desiredShape * rParams->desiredShape;

	if(rParams->method.compare("MLs")==0)
	{
		// initMLH generate the new probe value
		if(!m_probe->initMLH(rParams->desiredShape, eParams->lambda, dx_recon, eParams->beamSize, rParams->nProbes, (rParams->probeGuess.empty())?0:rParams->probeGuess.c_str()))
		{
			fprintf(stderr,"WARNING! Failed to initialize probe\n");
			return;
		}
		m_sample->setObjectArrayShape(m_scanMesh->getNPONEW());
		m_sample->clearObjectArray();
	}
	else
	{
		if(!m_probe->init(m_diffractions->getPatterns(), eParams->beamSize, eParams->dx_s, (rParams->probeGuess.empty())?0:rParams->probeGuess.c_str()))
		{
			fprintf(stderr,"WARNING! Failed to initialize probe\n");
			return;
		}
		m_sample->setObjectArrayShape(m_scanMesh->getMeshDimensions());
		m_sample->clearObjectArray();
	}

//	m_scanMesh->get_close_indices();

	// non mls
//	m_sample->setObjectArrayShape(m_scanMesh->getMeshDimensions());
	//MLS
//	m_sample->setObjectArrayShape(m_scanMesh->getNPONEW());
//	m_sample->clearObjectArray();

	////////////////////// test with matlab
	if(!rParams->objectGuess.empty())
		m_sample->loadGuess(rParams->objectGuess.c_str());
	else if(rParams->method.compare("MLs")==0)
		m_sample->initObject();
    //////////////////////////

	//	if(rParams->calculateRMS)
	//		m_diffractions->fillSquaredSums();

	if(rParams->method.compare("MLs")!=0)
	{
		if(rParams->calculateRMS)
			m_diffractions->fillSquaredSums();

		m_scanMesh->generateMesh(this->getBounds());
		m_errors.resize(rParams->iterations, 0);

	}
	else
	{
		m_diffractions->fillSquaredSums();
		m_diffractions->squaredRoot();
		m_probe->initEvo(m_scanMesh->getScanPositionsNum(), rParams->variable_probe_modes, m_scanMesh->m_oROI1, m_scanMesh->m_oROI2,
				m_scanMesh->m_Np_o_new, m_sample->getObjectArray(), m_diffractions->getPatterns(), m_diffractions->getTotalSquaredSum());
		m_errors.resize(rParams->iterations, 0);
		m_fourierErrors.resize(rParams->iterations);
		size_t betasize= m_scanMesh->getScanPositionsNum();

		complex_t initVal=make_complex_t(1, 0);
		if(rParams->beta_LSQ>0)
		{
			m_sample->beta_objectvec.resize(betasize, initVal);
			m_probe->beta_probevec.resize(betasize, initVal);
		}
		else
		{

		}
	}

//	m_scanMesh->generateMesh(this->getBounds());
}

//void IPhaser::phaseLoop()
//{
//	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
//
//	for(size_t mIndex=0; mIndex<m_phasingMethods.size(); ++mIndex)
//		for(unsigned int i=1; i<=m_phasingMethods[mIndex].iterations; ++i)
//		{
//			m_errors[i-1] += phaseStep(m_phasingMethods[mIndex].method, i);
////			if(i%rParams->updateVis==0 && m_renderer)
////				m_renderer->renderResources();
//			#ifdef DEBUG_MSGS
//			fprintf(stderr,"%s Iteration [%05d] --- RMS= %e\n", m_phasingMethods[mIndex].name.c_str(), i, m_errors[i-1]);
//			#endif
//			if(i%rParams->save == 0)
//				writeResultsToDisk(i/rParams->save);
//			if(rParams->time>=0 && m_phasingTimer.getElapsedTimeInSec()>=rParams->time)
//				break;
//		}
//}
//
//real_t IPhaser::phaseStep(IPhasingMethod* m, unsigned int i)
//{
//	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
//	return m->iteration(m_diffractions, m_probe, m_sample, m_scanMesh->getScanPositions(), i<=rParams->phaseConstraint,
//						i>=rParams->updateProbe, i>=rParams->updateProbeModes, rParams->calculateRMS);
//}

void IPhaser::phaseLoop()
{
	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();

	for(size_t mIndex=0; mIndex<m_phasingMethods.size(); ++mIndex)
		for(unsigned int i=1; i<=m_phasingMethods[mIndex].iterations; ++i)
		{
			m_stepTimer.start();

			m_errors[i-1] += phaseStep(m_phasingMethods[mIndex].method, i);

			m_stepTimer.stop();

			fprintf(stderr,"%s Iteration [%05d] --- step time= %fs\n", m_phasingMethods[mIndex].name.c_str(), i, m_stepTimer.getElapsedTimeInSec());

//			if(i%rParams->updateVis==0 && m_renderer)
//				m_renderer->renderResources();
			#ifdef DEBUG_MSGS
			fprintf(stderr,"%s Iteration [%05d] --- RMS= %e\n", m_phasingMethods[mIndex].name.c_str(), i, m_errors[i-1]);
			#endif
			if(i%rParams->save == 0)
				writeResultsToDisk(i/rParams->save);
			if(rParams->time>=0 && m_phasingTimer.getElapsedTimeInSec()>=rParams->time)
				break;
		}
}

real_t IPhaser::phaseStep(IPhasingMethod* m, unsigned int i)
{
	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
	return m->iteration(m_diffractions, m_probe, m_sample, m_scanMesh, m_fourierErrors, m_scanMesh->getScanPositions(), i<=rParams->phaseConstraint,
						i>=rParams->updateProbe, i>=rParams->updateProbeModes, i, rParams->calculateRMS);


}

void IPhaser::postPhase()
{
	m_probe->endModalReconstruction();
	m_scanMesh->clear();
	fprintf(stderr,"io=%f\tphase=%f\tDone!!\n", m_ioTimer.getElapsedTimeInSec(), m_phasingTimer.getElapsedTimeInSec());
}

void IPhaser::writeResultsToDisk(int r)
{
	#ifdef DEBUG_MSGS
	printf("Writing final reconstruction (%u,%u) to disk...\n", m_sample->getObjectArray()->getX(), m_sample->getObjectArray()->getY());
	#endif
	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
	char filename[255];
	sprintf(filename, "data/%s_probes_%d.%s", rParams->reconstructionID.c_str(), r, rParams->binaryOutput?"bin":"csv");
	m_probe->getModes()->getPtr()->save<complex_t>(filename, rParams->binaryOutput);
	sprintf(filename, "data/%s_object_%d.%s", rParams->reconstructionID.c_str(), r, rParams->binaryOutput?"bin":"csv");
	m_sample->getObjectArray()->save<complex_t>(filename, rParams->binaryOutput);
}

void IPhaser::phase()
{
	if(!m_probe || !m_sample || !m_diffractions || !m_scanMesh)
	{
		#ifdef DEBUG_MSGS
		fprintf(stderr,"[API Usage Error] Phaser::init() must be called before Phaser::phase()!\n");
		#endif
		return;
	}

	m_phasingTimer.start();
	this->prePhase();
	this->phaseLoop();
	m_phasingTimer.stop();
	this->postPhase();
}

/////////////////////////////////////////////////////////////////////////// Start ptychopy wrapper functions

Sample* IPhaser::getSample()
{
	return m_sample;
}

Probe* IPhaser::getProbe()
{
	return m_probe;
}

void IPhaser::phaseinit()
{
	if(!m_probe || !m_sample || !m_diffractions || !m_scanMesh)
	{
		#ifdef DEBUG_MSGS
		fprintf(stderr,"[API Usage Error] Phaser::init() must be called before Phaser::phase()!\n");
		#endif
		return;
	}

	m_phasingTimer.start();
	this->prePhase();

}

void IPhaser::phasepost()
{
	m_phasingTimer.stop();
	this->postPhase();
}

void IPhaser::phasestepvis(unsigned int i)
{

	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();


	for(size_t mIndex=0; mIndex<m_phasingMethods.size(); ++mIndex)
	{
		m_errors[i-1] += phaseStep(m_phasingMethods[mIndex].method, i);

		#ifdef DEBUG_MSGS
		fprintf(stderr,"%s Iteration [%05d] --- RMS= %e\n", m_phasingMethods[mIndex].name.c_str(), i, m_errors[i-1]);
		#endif
		if(rParams->time>=0 && m_phasingTimer.getElapsedTimeInSec()>=rParams->time)
			break;
	}

}


void IPhaser::phaseVisStep()
{

	if(!m_probe || !m_sample || !m_diffractions || !m_scanMesh)
	{
		#ifdef DEBUG_MSGS
		fprintf(stderr,"[API Usage Error] Phaser::init() must be called before Phaser::phase()!\n");
		#endif
		return;
	}

	m_phasingTimer.start();
	this->prePhase();
	this->phaseLoopVisStep();
	m_phasingTimer.stop();
	this->postPhase();


}

void IPhaser::phaseLoopVisStep()
{
	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();

	for(size_t mIndex=0; mIndex<m_phasingMethods.size(); ++mIndex)
		for(unsigned int i=1; i<=m_phasingMethods[mIndex].iterations; ++i)
		{
			m_errors[i-1] += phaseStep(m_phasingMethods[mIndex].method, i);

			#ifdef DEBUG_MSGS
			fprintf(stderr,"%s Iteration [%05d] --- RMS= %e\n", m_phasingMethods[mIndex].name.c_str(), i, m_errors[i-1]);
			#endif

			printf("Writing step reconstruction object array (%u,%u) to disk...\n", m_sample->getObjectArray()->getX(), m_sample->getObjectArray()->getY());
			const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
			char filename[255];
			//sprintf(filename, "data/%s_probes_%d.%s", rParams->reconstructionID.c_str(), r, rParams->binaryOutput?"bin":"csv");
			//m_probe->getModes()->getPtr()->save<complex_t>(filename, rParams->binaryOutput);
			sprintf(filename, "data/%s_object_%d.%s", rParams->reconstructionID.c_str(), i, rParams->binaryOutput?"bin":"csv");
			m_sample->getObjectArray()->save<complex_t>(filename, rParams->binaryOutput);

			if(rParams->time>=0 && m_phasingTimer.getElapsedTimeInSec()>=rParams->time)
				break;
		}
}