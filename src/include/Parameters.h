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
#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <vector_types.h>
#include <string>
#include "Singleton.h"
#include "datatypes.h"

typedef enum
{
	NONE		= 0x00,
	SQUARE_ROOT	= 0x01,
	BIN			= 0x02,
	FFT_SHIFT	= 0x04,
}PREPROCESSING_FLAGS;

struct PreprocessingParams
{
	int flags;
	uint2 symmetric_array_center;
	uint2 rawFileSize;
	std::string dataFilePattern;
	std::string hdf5DatasetName;
	unsigned int symmetric_array_size;
	unsigned int fileStartIndex;
	unsigned int diffsPerFile;
	unsigned int projectionsNum;
	unsigned int rotate_90_times;
	unsigned int bin_factor;
	unsigned int io_threads;
	real_t threshold_raw_data;
	real_t pixel_saturation_level;
	bool beamstopMask;

	PreprocessingParams(int f=NONE) :
			flags(f), symmetric_array_size(256), fileStartIndex(0), diffsPerFile(1), projectionsNum(0), rotate_90_times(0),
			bin_factor(1), io_threads(1), threshold_raw_data(0.0), pixel_saturation_level(-1.0), beamstopMask(false)
	{
		symmetric_array_center.x = 128;
		symmetric_array_center.y = 128;
		rawFileSize.x = 256;
		rawFileSize.y = 256;
		hdf5DatasetName = "/entry/data/data"; //for backward compatability
	}
};

struct ExperimentParams
{
	double lambda;
	double beamSize, dx_d, dx, dx_s, z_d;
	uint2 scanDims;
	float2 stepSize;
	float2 drift;
	bool spiralScan;
	int bit_depth;
//	size_t pixelSaturation;
	unsigned long long pixelSaturation;

	ExperimentParams() :beamSize(1e-7), lambda(2.48e-10), dx_d(172e-6), dx(1.4089e-8), dx_s(5.631549e-09), z_d(1.0), spiralScan(false),
						bit_depth(32), pixelSaturation( (1<<bit_depth)-1 )
	{
		scanDims.x = 12; scanDims.y=12;
		stepSize.x = 40e-9; stepSize.y = 40e-9;
		drift.x = 0; drift.y = 0;
	}
};

struct ReconstructionParams
{
	std::string reconstructionID;
	std::string algorithm;
	unsigned int iterations;
	unsigned int save;
	double time;
	unsigned int desiredShape;
	unsigned int probeModes;
	unsigned int jitterRadius;
	unsigned int halo;
	unsigned int shareFrequency;
	unsigned int updateProbe;
	unsigned int updateProbeModes;
	unsigned int updateVis;
	unsigned int phaseConstraint;

	unsigned int nProbes;
	int Niter;
	int variable_probe_modes;
	int variable_intensity;
	int variable_probe;
	double beta_LSQ;
	int beta_object;
	int beta_probe;
	int probe_pos_search;
	int probe_reconstruct;
	int apply_subpix_shift;
	int apply_multimodal_update;
	int object_reconstruct;
	unsigned int Nobjects;
	double delta_p;
	std::string method;
	int Nrec;

	bool blind;
	bool simulated;
	bool flipScanAxis;
	bool mirrorX;
	bool mirrorY;
	bool calculateRMS;
	bool binaryOutput;
	std::string positionsFilename;
	std::string probeGuess;
	std::string objectGuess;
	std::string wsServerUrl;

	ReconstructionParams() :
		algorithm("ePIE"), iterations(100), save(999999), time(-1.0), desiredShape(256), probeModes(5), jitterRadius(0), halo(0), shareFrequency(10),
		updateProbe(10), updateProbeModes(20), updateVis(10), phaseConstraint(1),
		blind(true), simulated(false), flipScanAxis(false), mirrorX(false), mirrorY(false), calculateRMS(false), binaryOutput(false), nProbes(5),
		Niter(100), variable_probe_modes(1), beta_LSQ(0.9),beta_object(1), beta_probe(1), probe_pos_search(5), variable_intensity(1),
		variable_probe(1), apply_subpix_shift(1), apply_multimodal_update(0), object_reconstruct(1), probe_reconstruct(1), Nobjects(1), delta_p(0.1),
		method("MLs"), Nrec(1)
	{}
};

class Parameters
{
private:
	ExperimentParams m_eParams;
	PreprocessingParams m_pParams;
	ReconstructionParams m_rParams;

	real_t samplePlanePixelSize() const
	{
		//m_eParams.lambda =  (PLANCK_CONSTANT*SPEED_OF_LIGHT) / (m_eParams.energy*1e3*ELEMENTARY_CHARGE);
		return (m_eParams.lambda*m_eParams.z_d)/(m_rParams.desiredShape*m_eParams.dx_d);
	}
public:
	Parameters();
	virtual ~Parameters();

	void updatePixelSaturation();
	void setDrift(float2 d) {m_eParams.drift = d;}
	void parseFromCommandLine(int argc, char *argv[]);
	const ExperimentParams* 	getExperimentParams() 	const {return &m_eParams;}
	const PreprocessingParams* 	getPreprocessingParams()const {return &m_pParams;}
	const ReconstructionParams* getReconstructionParams()const {return &m_rParams;}
	bool renderResults() const {return !m_rParams.blind;}

	/////ptychpy function

	void parseFromCPython(char *jobID, char *fp, int fs, char *hdf5path, int dpf, double beamSize, char *probeGuess, char *objectGuess, \
                int size, int qx, int qy, int nx, int ny, int scanDimsx, int scanDimsy, int spiralScan, int flipScanAxis, int mirror1stScanAxis, \
                int mirror2ndScanAxis, double stepx, double stepy, int probeModes, double lambda, double dx_d, double z, int iter, int T, int jitterRadius, \
                double delta_p,  int threshold, int rotate90, int sqrtData, int fftShiftData, int binaryOutput, int simulate, \
                int phaseConstraint, int updateProbe, int updateModes, int beamstopMask, char *lf);
};

typedef Singleton<Parameters> CXParams;

#endif /* PARAMETERS_H_ */
