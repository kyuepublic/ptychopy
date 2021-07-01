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
#include "utilities.h"
#include "CudaFFTPlan.h"
#include "CudaSmartPtr.h"
#include "Cuda2DArray.hpp"
#include "CXMath.h"
#include "Parameters.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

GPUProperties::GPUProperties()
{
	cudaGetDeviceCount(&m_gpuCount);
	cutilCheckMsg("Device# 0 property querying failed!");
	int deviceID = 0;
#ifdef HAVE_MPI
	int initialized;
	MPI_Initialized(&initialized);
	if(initialized)
		MPI_Comm_rank(MPI_COMM_WORLD, &deviceID);
	if(deviceID>=m_gpuCount)
		deviceID = 0;
#endif
	cudaDeviceProp gpuProperties;
	cudaGetDeviceProperties(&gpuProperties, deviceID);
	char msg[100];
	sprintf(msg, "Device# %d property querying failed!", deviceID);
	cutilCheckMsg(msg);
	m_gpuWarpSize = gpuProperties.warpSize;
	m_gpuMaxThread = gpuProperties.maxThreadsPerBlock;
}

size_t GPUProperties::getGPUAvailableMemory() const
{
	size_t freeMem, totalMem;
	cudaMemGetInfo ( &freeMem, &totalMem);
	cutilCheckMsg("Failed to get GPU available memory\n");
	return freeMem;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int4 calculateOffsets(float2 scanPos)
{
	int4 offsets=make_int4(roundf(scanPos.x), roundf(scanPos.y), 0,0);
	if(scanPos.x<0)
	{
		offsets.z = -roundf(scanPos.x);
		offsets.x = 0;
	}
	if(scanPos.y<0)
	{
		offsets.w = -roundf(scanPos.y);
		offsets.y = 0;
	}
	return offsets;
}

int4 CXUtil::applyProjectionApprox(const CudaSmartPtr& object, const Cuda3DArray<complex_t>* probeModes,
		Cuda3DArray<complex_t>* psi, float2 scanPos)
{
	int4 offsets =  calculateOffsets(scanPos);
	CXMath::multiply<complex_t>(object.get(), probeModes, psi, offsets.x, offsets.y, offsets.z, offsets.w);
	return offsets;
}

int4 CXUtil::applyProjectionApprox(const CudaSmartPtr& object, Cuda3DElement<complex_t> probeMode,
		CudaSmartPtr psi, float2 scanPos)
{
	int4 offsets =  calculateOffsets(scanPos);
	CXMath::multiply<complex_t>(object.get(), probeMode, psi.get(), offsets.x, offsets.y, offsets.z, offsets.w);
	return offsets;
}

void CXUtil::gaussSmooth(real_t* d_arr, unsigned int smoothWindow, unsigned int x, unsigned int y)
{
	CudaSmartPtr gaussKernel(new Cuda2DArray<real_t>(x, y));

	h_simulateGaussian(gaussKernel->getDevicePtr<real_t>(), smoothWindow/2.3548,
						gaussKernel->getX(), gaussKernel->getY(),gaussKernel->getAlignedY());

	CudaSmartPtr fftGauss(new Cuda2DArray<complex_t>(x, y));
	CudaSmartPtr fftArr(new Cuda2DArray<complex_t>(x, y));

	REAL2COMPLEX_FFT(FFTPlanner::getInstance()->getR2CPlan(gaussKernel.get()), gaussKernel->getDevicePtr<real_t>(),
				fftGauss->getDevicePtr<complex_t>());
	cutilCheckMsg("CXUtil::gaussSmooth() FFT execution failed!\n");
	REAL2COMPLEX_FFT(FFTPlanner::getInstance()->getR2CPlan(gaussKernel.get()), d_arr,
				fftArr->getDevicePtr<complex_t>());
	cutilCheckMsg("CXUtil::gaussSmooth() FFT execution failed!\n");

	h_multiply(fftArr->getDevicePtr<complex_t>(), fftGauss->getDevicePtr<complex_t>(), fftArr->getDevicePtr<complex_t>(),
							fftArr->getX(), fftArr->getY(), fftArr->getAlignedY(), true);

	COMPLEX2REAL_FFT(FFTPlanner::getInstance()->getC2RPlan(gaussKernel.get()), fftArr->getDevicePtr<complex_t>(), d_arr);
	cutilCheckMsg("CXUtil::gaussSmooth() FFT execution failed!\n");
}

void CXUtil::applyModulusConstraint(const Cuda3DArray<complex_t>* d_psi, Cuda3DArray<complex_t>* d_output, 
									Cuda3DElement<real_t> d_det_mod, const real_t* d_beamstopMask)
{
	COMPLEX2COMPLEX_FFT(FFTPlanner::getInstance()->getC2CPlan(&d_det_mod, d_psi->checkUseAll()?d_psi->getNum():1),
				d_psi->getPtr()->getDevicePtr<complex_t>(), d_output->getPtr()->getDevicePtr<complex_t>(), CUFFT_FORWARD);
	cutilCheckMsg("CXUtil::applyModulusConstraint() FFT execution failed!\n");

	const ExperimentParams* eParams = CXParams::getInstance()->getExperimentParams();
	real_t saturationValue = (real_t)eParams->pixelSaturation;

	h_adjustFFT(d_output->getPtr()->getDevicePtr<complex_t>(), d_output->getPtr()->getDevicePtr<complex_t>(), d_det_mod.getDevicePtr(), d_beamstopMask,
				saturationValue, d_psi->checkUseAll()?d_psi->getNum():1, d_det_mod.getX(), d_det_mod.getY(), d_det_mod.getAlignedY());

	COMPLEX2COMPLEX_FFT(FFTPlanner::getInstance()->getC2CPlan(&d_det_mod, d_psi->checkUseAll()?d_psi->getNum():1),
				d_output->getPtr()->getDevicePtr<complex_t>(), d_output->getPtr()->getDevicePtr<complex_t>(), CUFFT_INVERSE);
	cutilCheckMsg("CXUtil::applyModulusConstraint() FFT execution failed!\n");
}

void CXUtil::ff2Mat(Cuda3DArray<complex_t>* d_psi, Cuda3DArray<complex_t>* d_output,
		Cuda3DElement<real_t> d_det_mod, const real_t* d_beamstopMask)
{
	COMPLEX2COMPLEX_FFT(FFTPlanner::getInstance()->getC2CPlan(&d_det_mod, d_psi->checkUseAll()?d_psi->getNum():1),
				d_psi->getPtr()->getDevicePtr<complex_t>(), d_output->getPtr()->getDevicePtr<complex_t>(), CUFFT_FORWARD);
	cutilCheckMsg("CXUtil::applyModulusConstraint() FFT execution failed!\n");
}


void CXUtil::iff2Mat(Cuda3DArray<complex_t>* d_psi, Cuda3DArray<complex_t>* d_output,
		Cuda3DElement<real_t> d_det_mod, const real_t* d_beamstopMask)
{
	if(COMPLEX2COMPLEX_FFT(FFTPlanner::getInstance()->getC2CPlan(&d_det_mod, d_psi->checkUseAll()?d_psi->getNum():1),
				d_psi->getPtr()->getDevicePtr<complex_t>(), d_output->getPtr()->getDevicePtr<complex_t>(), CUFFT_INVERSE)!=CUFFT_SUCCESS)
		fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
	cutilCheckMsg("CXUtil::iff2Mat() FFT execution failed!\n");

//	if (cudaDeviceSynchronize() != cudaSuccess){
//		fprintf(stderr, "Cuda error: Failed to synchronize\n");
//	}
}

void CXUtil::applyFFT(const Cuda3DArray<complex_t>* d_psi, Cuda3DArray<complex_t>* d_output,
					  Cuda3DElement<real_t> d_det_mod, const real_t* d_beamstopMask)
{
	COMPLEX2COMPLEX_FFT(FFTPlanner::getInstance()->getC2CPlan(&d_det_mod, 1),
				d_psi->getPtr()->getDevicePtr<complex_t>(), d_output->getPtr()->getDevicePtr<complex_t>(), CUFFT_FORWARD);
	cutilCheckMsg("CXUtil::applyModulusConstraint() FFT execution failed!\n");

}


real_t CXUtil::calculateER(Cuda3DArray<complex_t>* d_psi, Cuda3DElement<real_t> d_det_mod)
{
	COMPLEX2COMPLEX_FFT(FFTPlanner::getInstance()->getC2CPlan(&d_det_mod, d_psi->checkUseAll()?d_psi->getNum():1),
				d_psi->getPtr()->getDevicePtr<complex_t>(), d_psi->getPtr()->getDevicePtr<complex_t>(), CUFFT_FORWARD);
	cutilCheckMsg("CXUtil::calculateER() FFT execution failed!\n");
	return h_calculateER(d_psi->getPtr()->getDevicePtr<complex_t>(), d_det_mod.getDevicePtr(), d_psi->checkUseAll()?d_psi->getNum():1,
				d_det_mod.getX(), d_det_mod.getY(), d_det_mod.getAlignedY());
}

real_t CXUtil::calculateER(const CudaSmartPtr& GT, const CudaSmartPtr& object,
						unsigned int qx, unsigned int qy, unsigned int x, unsigned int y)
{
	return h_calculateER(GT->getDevicePtr<complex_t>(), object->getDevicePtr<complex_t>(), x, y, qx, qy,
						GT->getX(), GT->getY(), GT->getAlignedY(), object->getX(), object->getY(), object->getAlignedY());
}

real_t CXUtil::getModalDoubleMax(const Cuda3DArray<real_t>* d_arr)
{
	if( (!d_arr->checkUseAll()) || d_arr->getNum()==1)
		return h_realMax(d_arr->getPtr()->getDevicePtr<real_t>(), d_arr->getPtr()->getX(), d_arr->getPtr()->getY(), d_arr->getPtr()->getAlignedY());
	else
	{
		if(!m_workingMemory.isValid())
			m_workingMemory = new Cuda2DArray<real_t>(d_arr->getDimensions().x, d_arr->getDimensions().y);
		h_realModalSum(d_arr->getPtr()->getDevicePtr<real_t>(), m_workingMemory->getDevicePtr<real_t>(), d_arr->getNum(),
						m_workingMemory->getX(), m_workingMemory->getY(), m_workingMemory->getAlignedY());
		return h_realMax(m_workingMemory->getDevicePtr<real_t>(), m_workingMemory->getX(), m_workingMemory->getY(), m_workingMemory->getAlignedY());
	}
}

real_t CXUtil::getModalDoubleSum(const Cuda3DArray<real_t>* d_arr)
{
	if( (!d_arr->checkUseAll()) || d_arr->getNum()==1)
		return h_realSum(d_arr->getPtr()->getDevicePtr<real_t>(), d_arr->getPtr()->getX(), d_arr->getPtr()->getY(), d_arr->getPtr()->getAlignedY());
	else
	{
		if(!m_workingMemory.isValid())
			m_workingMemory = new Cuda2DArray<real_t>(d_arr->getDimensions().x, d_arr->getDimensions().y);
		h_realModalSum(d_arr->getPtr()->getDevicePtr<real_t>(), m_workingMemory->getDevicePtr<real_t>(), d_arr->getNum(),
						m_workingMemory->getX(), m_workingMemory->getY(), m_workingMemory->getAlignedY());
		return h_realSum(m_workingMemory->getDevicePtr<real_t>(), m_workingMemory->getX(), m_workingMemory->getY(), m_workingMemory->getAlignedY());
	}
}

//void CXUtil::median(std::vector<double>& vec, double &x)
//{
//
//	size_t size = vec.size();
//	sort(vec.begin(), vec.end());
//	if (size % 2 == 0)
//	{
//		x=1.0*(scoresx[size / 2 - 1] + scoresx[size / 2]) / 2;
//	}
//	else
//	{
//		x=scoresx[size / 2];
//	}
//
//}

