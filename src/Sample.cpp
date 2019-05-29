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

#include "Sample.h"
#include "CudaSmartPtr.hpp"
#include "Sample.cuh"
#include "CudaFFTPlan.h"

using namespace std;

Sample::Sample() : m_maxObjectIntensity(0)
{}

Sample::Sample(unsigned int arrayShapeX, unsigned int arrayShapeY) : m_maxObjectIntensity(0)
{setObjectArrayShape(make_uint2(arrayShapeX, arrayShapeY));}

Sample::~Sample()
{h_unbindObjArrayTex();}

void Sample::setObjectArrayShape(uint2 xy)
{
	m_objectArrayShape = xy;
	if(m_objectArray.isValid())
	{
		h_unbindObjArrayTex();
		m_objectArray->reshapeDeviceMemory(m_objectArrayShape);
		m_objectIntensities->reshapeDeviceMemory(m_objectArrayShape);
	}
	else
	{
		m_objectArray = new Cuda2DArray<complex_t>(m_objectArrayShape.x, m_objectArrayShape.y);
		m_objectIntensities = new Cuda2DArray<real_t>(m_objectArrayShape.x, m_objectArrayShape.y);
	}
	h_bindObjArrayToTex(m_objectArray->getDevicePtr<complex_t>(), m_objectArray->getX(),
						m_objectArray->getY(), m_objectArray->getAlignedY());
}

bool Sample::loadFromFile(const char* filename, const char* phaseFile, real_t normalize)
{
	unsigned char* h_data = 0; unsigned char* h_phaseData = 0;
	unsigned int imageWidth = 0, imageHeight = 0;
	if(_LOAD_PGMU(filename, &h_data, &imageWidth, &imageHeight))
	{
		CudaSmartPtr mag, arg;
		real_t* h_dataf = new real_t[imageWidth*imageHeight];

		unsigned char maxValue = 0;
		for(unsigned int i=0; i<imageWidth*imageHeight; ++i)
			if(h_data[i]>maxValue) maxValue = h_data[i];

		for(unsigned int i=0; i<imageWidth*imageHeight; ++i)
			h_dataf[i] = ((real_t)h_data[i]/maxValue)*normalize+(1.0-normalize);

		mag = new Cuda2DArray<real_t>(imageHeight,imageWidth);
		mag->setFromHost(h_dataf,imageHeight,imageWidth);

		if(phaseFile!=0)
		{
			if(_LOAD_PGMU(phaseFile, &h_phaseData, &imageWidth, &imageHeight))
			{
				if(mag->getX()==imageHeight && mag->getY()==imageWidth)
				{
					maxValue = 0;
					for(unsigned int i=0; i<imageWidth*imageHeight; ++i)
						if(h_phaseData[i]>maxValue) maxValue = h_phaseData[i];

					for(unsigned int i=0; i<imageWidth*imageHeight; ++i)
						h_dataf[i] = ((real_t)h_phaseData[i]/maxValue)*1.0471975512;

					arg = new Cuda2DArray<real_t>(imageHeight,imageWidth);
					arg->setFromHost(h_dataf,imageHeight,imageWidth);
				}

				free(h_phaseData);
			}
		}

		delete[] h_dataf;
		free(h_data);

		h_applyHammingToSample(m_objectArray->getDevicePtr<complex_t>(), mag->getDevicePtr<real_t>(), arg->getDevicePtr<real_t>(),
								mag->getX(), mag->getY(), mag->getAlignedY(), m_objectArray->getX(), m_objectArray->getY(),
								m_objectArray->getAlignedY());
	}
	else
	{
		fprintf(stderr, "Failed to load sample from file: %s!\n", filename);
		return false;
	}

	return true;
}

void Sample::loadGuess(const char* filename)
{
	CudaSmartPtr guess;
	if(!guess.initFromFile<complex_t>(filename))
	{
		fprintf(stderr, "Failed to load object guess from file: %s!\n", filename);
		return;
	}
	guess->load<complex_t>(filename);
	h_setObjectArray(m_objectArray->getDevicePtr<complex_t>(), guess->getDevicePtr<complex_t>(), guess->getX(),
						guess->getY(), guess->getAlignedY(), m_objectArray->getX(), m_objectArray->getY(),
						m_objectArray->getAlignedY());
}

void Sample::clearObjectArray()
{
	m_objectArray->set(0);
	m_maxObjectIntensity = 0;
}

bool isPhaseName(const char* name)
{
	string rName(name);
	return (rName.find_first_of('|') != 0 );
}

const CudaSmartPtr& Sample::updateRenderable(const char* name)
{
	if( isPhaseName(name) ) //Sample Phase
	{
		h_getObjectPhase(m_objectArray->getDevicePtr<complex_t>(), m_objectPhases->getDevicePtr<real_t>(),
						m_objectPhases->getX(), m_objectPhases->getY(), m_objectPhases->getAlignedY());
		return m_objectPhases;
	}
	else 				//Sample magnitude
		return m_objectIntensities;
}

void Sample::toRGBA(float4* out, const char* name, float tf, float ts)
{
//	h_realToRGBA(updateRenderable(name)->getDevicePtr<real_t>(), out, m_objectIntensities->getX(), m_objectIntensities->getY(),
//					m_objectIntensities->getAlignedY(), (isPhaseName(name)?1.0:1.0/m_maxObjectIntensity), tf, ts);
}

void Sample::toGray(float* out, const char* name, bool outAligned)
{
//	h_realToGray(updateRenderable(name)->getDevicePtr<real_t>(), out, m_objectIntensities->getX(), m_objectIntensities->getY(),
//					m_objectIntensities->getAlignedY(), (isPhaseName(name)?1.0:1.0/m_maxObjectIntensity), outAligned);
}

void Sample::extractROI(Cuda3DElement<complex_t> roi, float qx, float qy) const
{
	h_extractObjectArray(m_objectArray->getDevicePtr<complex_t>(), roi.getDevicePtr(),
				qx, qy, roi.getX(), roi.getY(), roi.getAlignedY(),
				m_objectArray->getX(), m_objectArray->getY(), m_objectArray->getAlignedY());
}

void Sample::extractROI(CudaSmartPtr roi, float qx, float qy) const
{
	h_extractObjectArray(m_objectArray->getDevicePtr<complex_t>(), roi->getDevicePtr<complex_t>(),
				qx, qy, roi->getX(), roi->getY(), roi->getAlignedY(),
				m_objectArray->getX(), m_objectArray->getY(), m_objectArray->getAlignedY());
}

void Sample::updateObjectEstimate(	const Cuda3DArray<complex_t>* probeModes, const Cuda3DArray<complex_t>* psi, 
									const Cuda3DArray<complex_t>* psi_old, unsigned int qx, unsigned int qy,
									real_t probeMaxIntensity, bool phaseConstraint)
{
	h_updateObjectArray(m_objectArray->getDevicePtr<complex_t>(), probeModes->getPtr()->getDevicePtr<complex_t>(), psi->getPtr()->getDevicePtr<complex_t>(),
					psi_old->getPtr()->getDevicePtr<complex_t>(), m_objectIntensities->getDevicePtr<real_t>(), qx, qy, 1.0/probeMaxIntensity, probeModes->checkUseAll()?probeModes->getNum():1,
					psi->getDimensions().x, psi->getDimensions().y, psi->getPtr()->getAlignedY(), m_objectArray->getX(), m_objectArray->getY(), m_objectArray->getAlignedY(), phaseConstraint);

	updateMaxIntensity();
}

void Sample::updateIntensities(bool useSum)
{
	h_realComplexAbs(m_objectArray->getDevicePtr<complex_t>(), m_objectIntensities->getDevicePtr<real_t>(),
					m_objectArray->getX(), m_objectArray->getY(), m_objectArray->getAlignedY(), true);
	updateMaxIntensity(useSum);
}

void Sample::updateMaxIntensity(bool useSum)
{
	m_maxObjectIntensity = useSum ? h_realSum(m_objectIntensities->getDevicePtr<real_t>(), m_objectIntensities->getX(), m_objectIntensities->getY(), m_objectIntensities->getAlignedY()):
									h_realMax(m_objectIntensities->getDevicePtr<real_t>(), m_objectIntensities->getX(), m_objectIntensities->getY(), m_objectIntensities->getAlignedY());
//	m_renderableUpdated = true;
}

void Sample::addNeighborSubsamples(const Cuda3DArray<complex_t>* ns, CudaSmartPtr no, CudaSmartPtr nd, uint2 myOffset)
{
	Cuda3DElement<complex_t> n = ns->getAt(0);
	h_addSubsamples(m_objectArray->getDevicePtr<complex_t>(), ns->getPtr()->getDevicePtr<complex_t>(),
					no->getDevicePtr<uint2>(), nd->getDevicePtr<uint2>(), n.getX(), n.getAlignedY(),
					myOffset, ns->getNum(), m_objectArray->getX(), m_objectArray->getY(), m_objectArray->getAlignedY());
}

const CudaSmartPtr& Sample::stitchSubsamples(const ICuda2DArray* s1, const ICuda2DArray* s2, unsigned char dir, bool simulated)
{
	CudaSmartPtr tempC = new Cuda2DArray<complex_t>(s1->getX(), s1->getY());

	m_objectArray->reshapeDeviceMemory(s1->getDimensions());
	m_objectIntensities->reshapeDeviceMemory(s1->getDimensions());

	//Find registration point based on phase correlation
	//Smooth the subsamples first using hamming windows and generate gradient images
	if(simulated)
	{
		h_getMagnitudeGradient(s1->getDevicePtr<complex_t>(), m_objectArray->getDevicePtr<complex_t>(), s1->getX(), s1->getY(), s1->getAlignedY(), true);
		h_getMagnitudeGradient(s2->getDevicePtr<complex_t>(), tempC->getDevicePtr<complex_t>(), tempC->getX(), tempC->getY(), tempC->getAlignedY(), true);
	}
	else
	{
		h_getPhaseGradient(s1->getDevicePtr<complex_t>(), m_objectArray->getDevicePtr<complex_t>(), s1->getX(), s1->getY(), s1->getAlignedY(), true);
		h_getPhaseGradient(s2->getDevicePtr<complex_t>(), tempC->getDevicePtr<complex_t>(), tempC->getX(), tempC->getY(), tempC->getAlignedY(), true);
	}

	//F1 = fft2(I1)
	COMPLEX2COMPLEX_FFT(FFTPlanner::getInstance()->getC2CPlan(m_objectArray.get()), m_objectArray->getDevicePtr<complex_t>(),
						m_objectArray->getDevicePtr<complex_t>(), CUFFT_FORWARD);
	cutilCheckMsg("Sample::stitchSubsamples() FFT execution failed!\n");
	//F2 = fft2(I2)
	COMPLEX2COMPLEX_FFT(FFTPlanner::getInstance()->getC2CPlan(tempC.get()), tempC->getDevicePtr<complex_t>(),
						tempC->getDevicePtr<complex_t>(), CUFFT_FORWARD);
	cutilCheckMsg("Sample::stitchSubsample() FFT execution failed!\n");

	//Calculate cross power spectrum between F1 and F2
	h_crossPowerSpectrum(m_objectArray->getDevicePtr<complex_t>(), tempC->getDevicePtr<complex_t>(),
						tempC->getDevicePtr<complex_t>(), tempC->getX(), tempC->getY(), tempC->getAlignedY());

	//Obtain the normalized cross-correlation ncc = ifft2(cps)
	COMPLEX2COMPLEX_FFT(FFTPlanner::getInstance()->getC2CPlan(tempC.get()), tempC->getDevicePtr<complex_t>(),
						tempC->getDevicePtr<complex_t>(), CUFFT_INVERSE);
	cutilCheckMsg("Sample::stitchSubsample() FFT execution failed!\n");
	h_realComplexAbs(tempC->getDevicePtr<complex_t>(), m_objectIntensities->getDevicePtr<real_t>(),
						tempC->getX(), tempC->getY(), tempC->getAlignedY(), false);

	//Now adjust, register, and stitch the subsample
	int2 registrationPoint = h_realArgMax2D(m_objectIntensities->getDevicePtr<real_t>(), m_objectIntensities->getX(), m_objectIntensities->getY(), m_objectIntensities->getAlignedY(), dir);
	h_realComplexModulate(s1->getDevicePtr<complex_t>(), s2->getDevicePtr<complex_t>(), registrationPoint, s1->getX(), s1->getY(), s1->getAlignedY(), dir);
	registrationPoint.x = registrationPoint.x<0?0:registrationPoint.x;
	registrationPoint.y = registrationPoint.y<0?0:registrationPoint.y;

	m_objectArray->reshapeDeviceMemory(make_uint2(s1->getX()+registrationPoint.x, s1->getY()+registrationPoint.y));
	int halfXOverlap = registrationPoint.x ? gh_iDivUp(s1->getX()-registrationPoint.x, 2) : 0;
	int halfYOverlap = registrationPoint.y ? gh_iDivUp(s1->getY()-registrationPoint.y, 2) : 0;
	
	h_mergeObjectArray(m_objectArray->getDevicePtr<complex_t>(), s1->getDevicePtr<complex_t>(), s2->getDevicePtr<complex_t>(),
						s1->getAlignedY(), s2->getAlignedY(), halfXOverlap, halfYOverlap, m_objectArray->getX(), m_objectArray->getY(), m_objectArray->getAlignedY(),
						registrationPoint.x+halfXOverlap, registrationPoint.y+halfYOverlap);

	return m_objectArray;
}

void Sample::fillResources()
{
//	m_myResources.push_back(Resource("|O|", GRAYS, this));
//	m_myResources.push_back(Resource("arg(O)", RAINBOW, this));
//	m_objectPhases = new Cuda2DArray<real_t>(m_objectArrayShape.x, m_objectArrayShape.y);
}
