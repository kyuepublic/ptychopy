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

#include "DiffractionLoader.h"
#include "Diffractions.cuh"
#include "FileManager.h"
#include <fstream>

using namespace std;

DiffractionLoader::DiffractionLoader(int gpuID, const char* fname, Cuda3DElement<real_t> pattern, unsigned int dpi, const PreprocessingParams& m_params) :
										m_gpuID(gpuID), m_pattern(pattern), m_fileName(fname), m_dpIndex(dpi), m_params(m_params)
{}

DiffractionLoader::~DiffractionLoader()
{}

void DiffractionLoader::run()
{
	unsigned int rows=m_params.rawFileSize.x,cols=m_params.rawFileSize.y;
	real_t* h_data=IO::getInstance()->readBuffer(m_fileName, m_dpIndex, rows, cols);
	if(h_data==0)
	{
		fprintf(stderr, "Failed to read diffraction pattern from file (%s)\n", m_fileName.c_str());
		return;
	}

	/*Preprocessing*/
	cudaSetDevice(m_gpuID);
	cudaStream_t cudaStream;
	cudaStreamCreate(&cudaStream);
	cutilCheckMsg("DiffractionLoader::run() CUDA stream creation failed!");

	CudaSmartPtr tempPattern;
	if( (m_params.rotate_90_times>0) || (m_params.flags&FFT_SHIFT))
		tempPattern = new Cuda2DArray<real_t>(m_params.symmetric_array_size,m_params.symmetric_array_size);

	if(rows==m_params.symmetric_array_size && cols==m_params.symmetric_array_size)
	{
		m_pattern.setFromHost(h_data, rows, cols, &cudaStream);
		/*CudaSmartPtr d_dataRead = new Cuda2DArray<real_t>(rows, cols);
		d_dataRead->setFromHost<real_t>(h_data, rows, cols, &cudaStream);
		d_dataRead->save<real_t>("data/readDiff.csv");
		exit(1);*/
	}
	else
	{
		CudaSmartPtr d_dataRead = new Cuda2DArray<real_t>(rows, cols);
		//1- Symmetrize diffraction to a square matrix
		d_dataRead->setFromHost<real_t>(h_data, rows, cols, &cudaStream);

		int2 dataOffset, outOffset, dataSize;
		dataSize.x = m_params.symmetric_array_center.x - (m_params.symmetric_array_size/2);
		dataSize.y = m_params.symmetric_array_center.y - (m_params.symmetric_array_size/2);
		outOffset.x = (dataSize.x<0)? abs(dataSize.x): 0;
		dataOffset.x =(dataSize.x<0)? 0: dataSize.x;
		outOffset.y = (dataSize.y<0)? abs(dataSize.y): 0;
		dataOffset.y =(dataSize.y<0)? 0: dataSize.y;

		dataSize.x = m_params.symmetric_array_center.x + (m_params.symmetric_array_size/2);
		dataSize.y = m_params.symmetric_array_center.y + (m_params.symmetric_array_size/2);
		dataSize.x = (dataSize.x<rows)? m_params.symmetric_array_size: rows-dataOffset.x;
		dataSize.y = (dataSize.y<cols)? m_params.symmetric_array_size: cols-dataOffset.y;

		h_symmetrizeDiffraction(d_dataRead->getDevicePtr<real_t>(), m_pattern.getDevicePtr(),
								dataOffset.x, dataOffset.y, d_dataRead->getAlignedY(),
								outOffset.x, outOffset.y, m_pattern.getAlignedY(),
								dataSize.x, dataSize.y, &cudaStream);
	}

	//2- Rotate/Transpose the matrix n times
	if(m_params.rotate_90_times > 0)
	{
		tempPattern->setFromDevice(m_pattern.getDevicePtr(), tempPattern->getX(), tempPattern->getY(), &cudaStream);
		h_realRotate90(tempPattern->getDevicePtr<real_t>(), m_pattern.getDevicePtr(), tempPattern->getX(), tempPattern->getY(),
				tempPattern->getAlignedY(), m_params.rotate_90_times, &cudaStream);
	}
	//3- Store the matrix in FFT shifted form (0 frequency in the middle)
	if(m_params.flags & FFT_SHIFT)
		h_shiftFFT(m_pattern.getDevicePtr(), tempPattern->getDevicePtr<real_t>(), tempPattern->getX(),
				tempPattern->getY(), tempPattern->getAlignedY(), &cudaStream);

	cudaStreamDestroy(cudaStream);
	cutilCheckMsg("DiffractionLoader::run() CUDA stream destruction failed!");

	IO::getInstance()->bufferDone(m_fileName);
}
