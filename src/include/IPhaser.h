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

#ifndef IPHASER_H_
#define IPHASER_H_

#define DEBUG_MSGS

#include "IPhasingMethod.h"
#include <string>
#include "Timer.h"

class Diffractions;
class Probe;
class Sample;
class IPtychoScanMesh;
class IRenderer;

struct CXPhasing
{
	std::string name;
	IPhasingMethod* method;
	unsigned int iterations;
	CXPhasing(const char* n="", IPhasingMethod* m=0, unsigned int i=0): name(n), method(m), iterations(i)
	{}
};

class IPhaser
{
protected:
	Diffractions* m_diffractions;
	Probe* m_probe;
	Sample* m_sample;
	IPtychoScanMesh* m_scanMesh;
	IRenderer* m_renderer;
	std::vector<CXPhasing> m_phasingMethods;
	std::vector<real_t> m_errors;

	std::vector< std::vector<real_t> > m_fourierErrors;

	//timers
	Timer m_ioTimer;
	Timer m_phasingTimer;

	Timer m_stepTimer;

	virtual const int* getBounds() {return 0;}

	virtual bool initScanMesh();
	virtual bool initSample(uint2 meshDimensions);
	virtual bool initProbe();
	virtual bool initDiffractions();

	virtual void prePhase();
	virtual void phaseLoop();
    virtual real_t phaseStep(IPhasingMethod* m, unsigned int i);
	virtual void postPhase();

public:
	IPhaser();
	virtual ~IPhaser();

	virtual bool init();
	virtual void addPhasingMethod(const char*, unsigned int);
	virtual void phase();
	virtual void writeResultsToDisk(int r=0);
	virtual const std::vector<real_t>& getPhaseErrors() const {return m_errors;}

	/////////Start ptychopy function
    virtual void phaseLoopVisStep();
	virtual void phaseinit();
	virtual void phasepost();
	virtual void phasestepvis(unsigned int i);
	virtual void phaseVisStep();
	Sample* getSample();
	Probe* getProbe();
};

#endif /* IPHASER_H_ */
