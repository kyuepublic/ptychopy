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
{

}

Parameters::~Parameters()
{

}

void Parameters::updatePixelSaturation()
{
//	printf("the bitdepth is %d \n", m_eParams.bit_depth);
	m_eParams.pixelSaturation = pow(2,m_eParams.bit_depth)-1;
//	printf("the pixel value is lu %lu \n", m_eParams.pixelSaturation);
	m_eParams.pixelSaturation -= m_pParams.threshold_raw_data;
	if(m_pParams.flags & SQUARE_ROOT)
		m_eParams.pixelSaturation = sqrt(m_eParams.pixelSaturation);
//	printf("The saturation value is set to %u \n", m_eParams.pixelSaturation);
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
	if (_CHECK_CMDLINE (argc, (const char**)argv, "beamstopMask"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "beamstopMask", &value);
		m_pParams.beamstopMask = value>0;
	}
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
	if (_CHECK_CMDLINE (argc, (const char**)argv, "save"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "save", &value);
		if(value>0)
			m_rParams.save = value;
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
	if (_CHECK_CMDLINE (argc, (const char**)argv, "PPS"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "PPS", &value);
		if(value>=1)
			m_rParams.probe_pos_search = value;
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "probeModes"))
	{
		cutGetCmdLineArgumenti(argc, (const char**) argv, "probeModes", &value);
		if(value>=1)
		{
			m_rParams.probeModes = value;
			m_rParams.nProbes = value;
		}

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
//		sscanf(strValue, "%lf", &m_eParams.lambda);
		m_eParams.lambda = atof(strValue);
	}
	if (_CHECK_CMDLINE (argc, (const char**)argv, "dx_d"))
	{
		cutGetCmdLineArgumentstr(argc, (const char**) argv, "dx_d", &strValue);
		m_eParams.dx_d = atof(strValue);
	}
//	if (_CHECK_CMDLINE (argc, (const char**)argv, "dx"))
//	{
//		cutGetCmdLineArgumentstr(argc, (const char**) argv, "dx", &strValue);
//		m_eParams.dx = atof(strValue);
//	}
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
//		m_rParams.method = string(strValue);
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
	m_eParams.dx=m_eParams.dx_s;

}

void Parameters::parseFromCPython(char *jobID, char *algorithm, char *fp, int fs, char *hdf5path, int dpf, double beamSize, char *probeGuess, char *objectGuess, \
                int size, int qx, int qy, int nx, int ny, int scanDimsx, int scanDimsy, int spiralScan, int flipScanAxis, int mirror1stScanAxis, \
                int mirror2ndScanAxis, double stepx, double stepy, int probeModes, double lambda, double dx_d, double z, int iter, int T, int jitterRadius, \
                double delta_p,  int threshold, int rotate90, int sqrtData, int fftShiftData, int binaryOutput, int simulate, \
                int phaseConstraint, int updateProbe, int updateModes, int beamstopMask, char *lf, int PPS, double ***diffarr, complex_t *samplearr, \
                complex_t *probearr)
{
	int value;
	char* strValue;
	string temp;

	if(diffarr!=NULL)
	{
		m_rParams.diffarr=diffarr;
//		printf("Inside ptycho param 3D: %f 1D: .\n", m_rParams.diffarr[0][0][2]);
	}

	if(samplearr!=NULL)
	{
		m_rParams.samplearr=samplearr;
//        for (int i=0;i<scanDimsx;i++)
//            for (int j=0;j<scanDimsx;j++)
//            {
//            	if(j+i*scanDimsx < 20)
//            		printf("2D complex: %f + %fi\n", samplearr[j+i*scanDimsx].x, samplearr[j+i*scanDimsx].y);
//            }
	}

	if(probearr!=NULL)
	{
		m_rParams.probearr=probearr;
	}

	if (sqrtData==1)
	{
		m_pParams.flags |= SQUARE_ROOT;
		updatePixelSaturation();
	}
	if (fftShiftData==1)
	{
	    m_pParams.flags |= FFT_SHIFT;
	}
	if(beamstopMask==1)
	{
	    m_pParams.beamstopMask = true;
	}
	if(simulate==1)
	{
	    m_rParams.simulated = true;
	}
    if(spiralScan==1)
    {
        m_eParams.spiralScan = true;
    }
//	if (_CHECK_CMDLINE (argc, (const char**)argv, "bitDepth"))
//	{
//		cutGetCmdLineArgumenti(argc, (const char**) argv, "bitDepth", &value);
//		if(value>=1)
//		{
//			m_eParams.bit_depth = value;
//			updatePixelSaturation();
//		}
//	}
    if(flipScanAxis==1)
    {
        m_rParams.flipScanAxis = true;
    }
    if(mirror1stScanAxis==1)
    {
        m_rParams.mirrorX = true;
    }
    if(mirror2ndScanAxis==1)
    {
        m_rParams.mirrorY = true;
    }
//	if (_CHECK_CMDLINE (argc, (const char**)argv, "RMS"))
//	{
//		cutGetCmdLineArgumenti(argc, (const char**) argv, "RMS", &value);
//		m_rParams.calculateRMS = value>0;
//	}
    if(binaryOutput==1)
    {
        m_rParams.binaryOutput = true;
    }
//	if (_CHECK_CMDLINE (argc, (const char**)argv, "threads"))
//	{
//		cutGetCmdLineArgumenti(argc, (const char**) argv, "threads", &value);
//		if(value>=1)
//			m_pParams.io_threads = value;
//	}
    if(iter >= 0)
    {
        m_rParams.iterations = iter;
    }
//	if (_CHECK_CMDLINE (argc, (const char**)argv, "save"))
//	{
//		cutGetCmdLineArgumenti(argc, (const char**) argv, "save", &value);
//		if(value>0)
//			m_rParams.save = value;
//	}
    if(T>0)
    {
        m_rParams.time = T;
    }
//	if (_CHECK_CMDLINE (argc, (const char**)argv, "T"))
//	{
//		cutGetCmdLineArgumentstr(argc, (const char**) argv, "T", &strValue);
//		m_rParams.time = atof(strValue);
//	}
    if(size>16)
    {
        m_rParams.desiredShape = size;
    }
    else
    {
        printf("the size must be large than 16 \n");
    }
    if(probeModes>=1)
    {
        m_rParams.probeModes = probeModes;
        m_rParams.nProbes = probeModes;
    }
    if(fs >= 0)
    {
        m_pParams.fileStartIndex = fs;
    }
    if(dpf > 1)
    {
        m_pParams.diffsPerFile = dpf;
    }
//	if (_CHECK_CMDLINE (argc, (const char**)argv, "angles"))
//	{
//		cutGetCmdLineArgumenti(argc, (const char**) argv, "angles", &value);
//		if(value>=0)
//			m_pParams.projectionsNum = value;
//	}
    if (threshold > 0)
    {
        m_pParams.threshold_raw_data = threshold;
        updatePixelSaturation();
    }
    if (jitterRadius > 0)
    {
        m_rParams.jitterRadius = jitterRadius;
    }
    if (rotate90 >= 0)
    {
        m_pParams.rotate_90_times = rotate90;
    }
//	if (_CHECK_CMDLINE (argc, (const char**)argv, "overlap"))
//	{
//		cutGetCmdLineArgumenti(argc, (const char**) argv, "overlap", &value);
//		if(value>=0)
//			m_rParams.halo = value;
//	}
//	if (_CHECK_CMDLINE (argc, (const char**)argv, "shareFrequency"))
//	{
//		cutGetCmdLineArgumenti(argc, (const char**) argv, "shareFrequency", &value);
//		if(value>0)
//			m_rParams.shareFrequency = value;
//	}
    if (updateProbe >= 0)
    {
        m_rParams.updateProbe = updateProbe;
    }
    if (updateModes >= 0)
    {
        m_rParams.updateProbeModes = updateModes;
    }
    if (phaseConstraint >= 0)
    {
        m_rParams.phaseConstraint = phaseConstraint;
    }
    if((fp != NULL) && (fp[0] != '\0'))
    {
        m_pParams.dataFilePattern = string(fp);
    }
    if((hdf5path != NULL) && (hdf5path[0] != '\0'))
    {
        m_pParams.hdf5DatasetName = string(hdf5path);
    }
    if(scanDimsx > 0 && scanDimsy >0)
    {
        m_eParams.scanDims.x = scanDimsx;
        m_eParams.scanDims.y = scanDimsy;
    }
//	if (_CHECK_CMDLINE (argc, (const char**)argv, "drift"))
//	{
//		cutGetCmdLineArgumentstr(argc, (const char**) argv, "drift", &strValue);
//		temp = string(strValue);
//		size_t commaPos = temp.find(',');
//		m_eParams.drift.x=atof(temp.substr(0,commaPos).c_str());
//		m_eParams.drift.y=atof(temp.substr(commaPos+1, temp.length()-commaPos).c_str());
//	}
    if(qx > 0 && qy >0)
    {
        m_pParams.symmetric_array_center.x = qx;
        m_pParams.symmetric_array_center.y = qy;
    }
    if(nx > 0 && ny >0)
    {
        m_pParams.rawFileSize.x = nx;
        m_pParams.rawFileSize.y = ny;
    }
    if(stepx > 0 && stepy >0)
    {
        m_eParams.stepSize.x = stepx;
        m_eParams.stepSize.y = stepy;
    }
    if (beamSize > 0)
    {
        m_eParams.beamSize = beamSize;
    }
    if (lambda > 0)
    {
        m_eParams.lambda = lambda;
    }
    if (dx_d > 0)
    {
        m_eParams.dx_d = dx_d;
    }

//	if (_CHECK_CMDLINE (argc, (const char**)argv, "dx"))
//	{
//		cutGetCmdLineArgumentstr(argc, (const char**) argv, "dx", &strValue);
//		m_eParams.dx = atof(strValue);
//	}
	if (PPS >=1)
	{
		m_rParams.probe_pos_search = PPS;
	}
    if (z > 0)
    {
        m_eParams.z_d = z;
    }
    if((jobID != NULL) && (jobID[0] != '\0'))
    {
        m_rParams.reconstructionID = string(jobID);
    }
    if((algorithm != NULL) && (algorithm[0] != '\0'))
    {
        m_rParams.algorithm = string(algorithm);
//        m_rParams.method=m_rParams.algorithm;
    }
    if((lf != NULL) && (lf[0] != '\0'))
    {
        m_rParams.positionsFilename = string(lf);
    }
    if((probeGuess != NULL) && (probeGuess[0] != '\0'))
    {
        m_rParams.probeGuess = string(probeGuess);
    }
    if((objectGuess != NULL) && (objectGuess[0] != '\0'))
    {
        m_rParams.objectGuess = string(objectGuess);
    }

//	if (_CHECK_CMDLINE (argc, (const char**)argv, "wsServerUrl"))
//	{
//		cutGetCmdLineArgumentstr(argc, (const char**) argv, "wsServerUrl", &strValue);
//		m_rParams.wsServerUrl = string(strValue);
//	}
//
//
	m_pParams.symmetric_array_size = m_rParams.desiredShape;
	m_eParams.dx_s = samplePlanePixelSize();
	m_eParams.dx=m_eParams.dx_s;
}
