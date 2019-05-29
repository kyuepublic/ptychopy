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

#include "Parameters.h"
#include "utilities.h"

using namespace std;

Parameters::Parameters(): m_eParams(), m_pParams(0), m_rParams()
{}

Parameters::~Parameters()
{}

void Parameters::updatePixelSaturation()
{
	/*m_eParams.pixelSaturation = (1<<m_eParams.bit_depth)-1;*/
	if (m_eParams.bit_depth==32)
	{
		m_eParams.pixelSaturation =4294967295;
	}
	else if (m_eParams.bit_depth==16)
	{
		m_eParams.pixelSaturation =65535;
	}
	else
	{
		m_eParams.pixelSaturation =4294967295;
	}

//	printf("the pixel value is %u \n", m_eParams.pixelSaturation);
	m_eParams.pixelSaturation -= m_pParams.threshold_raw_data;
	if(m_pParams.flags & SQUARE_ROOT)
		m_eParams.pixelSaturation = sqrt(m_eParams.pixelSaturation)-1;
//		printf("inside if the pixel value is %u \n", m_eParams.pixelSaturation);
//	printf("FINAL if the pixel value is %u \n", m_eParams.pixelSaturation);
}


void Parameters::parseFromCommandLine(int argc, char *argv[])
{
	int value;
	char* strValue;
	string temp;

	if (_CHECK_CMDLINE (argc, (const char**)argv, "sqrtData"))
	{
		m_pParams.flags |= SQUARE_ROOT;
		updatePixelSaturation();
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "fftShiftData"))
		m_pParams.flags |= FFT_SHIFT;
	if (_CHECK_CMDLINE (argc, (const char**)argv, "blind"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "blind", &value);
		m_rParams.blind = value>0;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "simulate"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "simulate", &value);
		m_rParams.simulated = value>0;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "spiralScan"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "spiralScan", &value);
		m_eParams.spiralScan = value>0;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "bitDepth"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "bitDepth", &value);
		if(value>=1)
		{
			m_eParams.bit_depth = value;
			updatePixelSaturation();
		}
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "flipScanAxis"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "flipScanAxis", &value);
		m_rParams.flipScanAxis = value>0;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "mirror1stScanAxis"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "mirror1stScanAxis", &value);
		m_rParams.mirrorX = value>0;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "mirror2ndScanAxis"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "mirror2ndScanAxis", &value);
		m_rParams.mirrorY = value>0;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "RMS"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "RMS", &value);
		m_rParams.calculateRMS = value>0;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "binaryOutput"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "binaryOutput", &value);
		m_rParams.binaryOutput = value>0;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "threads"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "threads", &value);
		if(value>=1)
			m_pParams.io_threads = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "i"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "i", &value);
		if(value>=0)
			m_rParams.iterations = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "T"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "T", &strValue);
		m_rParams.time = atof(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "size"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "size", &value);
		if(value>=32)
			m_rParams.desiredShape = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "probeModes"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "probeModes", &value);
		if(value>=1)
			m_rParams.probeModes = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "fs"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "fs", &value);
		if(value>=0)
			m_pParams.fileStartIndex = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "dpf"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "dpf", &value);
		if(value>=1)
			m_pParams.diffsPerFile = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "angles"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "angles", &value);
		if(value>=0)
			m_pParams.projectionsNum = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "threshold"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "threshold", &value);
		if(value>=0)
		{
			m_pParams.threshold_raw_data = value;
			updatePixelSaturation();
		}
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "jitterRadius"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "jitterRadius", &value);
		if(value>=0)
			m_rParams.jitterRadius = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "rotate90"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "rotate90", &value);
		if(value>=0)
			m_pParams.rotate_90_times = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "overlap"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "overlap", &value);
		if(value>=0)
			m_rParams.halo = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "shareFrequency"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "shareFrequency", &value);
		if(value>0)
			m_rParams.shareFrequency = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "updateProbe"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "updateProbe", &value);
		if(value>=0)
			m_rParams.updateProbe = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "updateModes"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "updateModes", &value);
		if(value>=0)
			m_rParams.updateProbeModes = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "updateVis"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "updateVis", &value);
		if(value>=0)
			m_rParams.updateVis = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "phaseConstraint"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "phaseConstraint", &value);
		if(value>=0)
			m_rParams.phaseConstraint = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "fp"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "fp", &strValue);
		m_pParams.dataFilePattern = string(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "hdf5path"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "hdf5path", &strValue);
		m_pParams.hdf5DatasetName = string(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "scanDims"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "scanDims", &strValue);
		temp = string(strValue);
		size_t commaPos = temp.find(',');
		if (commaPos==std::string::npos)
		{
			m_eParams.scanDims.x=atoi(temp.c_str());
			m_eParams.scanDims.y=1;
		}
		else
		{
			m_eParams.scanDims.x=atoi(temp.substr(0,commaPos).c_str());
			m_eParams.scanDims.y=atoi(temp.substr(commaPos+1, temp.length()-commaPos).c_str());
		}
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "drift"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "drift", &strValue);
		temp = string(strValue);
		size_t commaPos = temp.find(',');
		m_eParams.drift.x=atof(temp.substr(0,commaPos).c_str());
		m_eParams.drift.y=atof(temp.substr(commaPos+1, temp.length()-commaPos).c_str());
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "qxy"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "qxy", &strValue);
		temp = string(strValue);
		size_t commaPos = temp.find(',');
		m_pParams.symmetric_array_center.x=atoi(temp.substr(0,commaPos).c_str());
		m_pParams.symmetric_array_center.y=atoi(temp.substr(commaPos+1, temp.length()-commaPos).c_str());
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "nxy"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "nxy", &strValue);
		temp = string(strValue);
		size_t commaPos = temp.find(',');
		m_pParams.rawFileSize.x=atoi(temp.substr(0,commaPos).c_str());
		m_pParams.rawFileSize.y=atoi(temp.substr(commaPos+1, temp.length()-commaPos).c_str());
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "step"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "step", &strValue);
		temp = string(strValue);
		size_t commaPos = temp.find(',');
		if (commaPos==std::string::npos)
			m_eParams.stepSize.y=m_eParams.stepSize.x=atoi(temp.c_str());
		else
		{
			m_eParams.stepSize.x=atof(temp.substr(0,commaPos).c_str());
			m_eParams.stepSize.y=atof(temp.substr(commaPos+1, temp.length()-commaPos).c_str());
		}
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "beamSize"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "beamSize", &strValue);
		m_eParams.beamSize = atof(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "lambda"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "lambda", &strValue);
		m_eParams.lambda = atof(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "dx_d"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "dx_d", &strValue);
		m_eParams.dx_d = atof(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "z"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "z", &strValue);
		m_eParams.z_d = atof(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "jobID"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "jobID", &strValue);
		m_rParams.reconstructionID = string(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "algorithm"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "algorithm", &strValue);
		m_rParams.algorithm = string(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "lf"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "lf", &strValue);
		m_rParams.positionsFilename = string(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "probeGuess"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "probeGuess", &strValue);
		m_rParams.probeGuess = string(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "objectGuess"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "objectGuess", &strValue);
		m_rParams.objectGuess = string(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "wsServerUrl"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "wsServerUrl", &strValue);
		m_rParams.wsServerUrl = string(strValue);
	}


	m_pParams.symmetric_array_size = m_rParams.desiredShape;
	m_eParams.dx_s = samplePlanePixelSize();
}
