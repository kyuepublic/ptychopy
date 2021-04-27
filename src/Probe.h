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
#ifndef PROBE_H_
#define PROBE_H_

#include "CudaSmartPtr.h"
#include "IRenderable.h"
#include "datatypes.h"
#include <vector_types.h>
#include "Timer.h"

template<typename T> class Cuda3DArray;
template<typename T> class Cuda3DElement;

struct curandGenerator_st;

class Probe : public IRenderable
{
public:

	CudaSmartPtr p_object;
	real_t MAX_ILLUM;
	std::vector <complex_t> beta_probevec;

	CudaSmartPtr tempArrR;

private:
	// 3d array for the 2d probes
	Cuda3DArray<complex_t>* m_modes;
	Cuda3DArray<real_t>* m_intensities;

	// variable modes for the first mode
	CudaSmartPtr m_extramodes;

	std::vector < std::vector <double> > m_probe_evolution;

	real_t m_maxIntensity;
    curandGenerator_st* m_randomGenerator;
    bool m_modesInitialized;
    real_t m_MAX_ILLUM;

	Timer m_testTimer;

	void initProbeModes();

public:
	Probe(unsigned int size, unsigned int modes=1);
	~Probe();

	void clear();
	bool init(const Cuda3DArray<real_t>* diffractions, real_t beamSize, real_t dx_s, const char* filename=0);
	bool initVarModes();

	bool initMLH(unsigned int desiredShape, double lambda, double dx_recon, double beamSize, unsigned int nProbes, const char* filename=0);

	bool initEvo(int Npos, int variable_probe_modes, std::vector <uint2> oROI1, std::vector <uint2> oROI2, uint2 Np_o,
			CudaSmartPtr objectArray, const Cuda3DArray<real_t>* diffractions, std::vector<real_t> diffSquareRoot);

	CudaSmartPtr generateRandKernel(unsigned int x, unsigned int y);
	CudaSmartPtr generatepureRandKernel(unsigned int x, unsigned int y);

	void remove_extra_degree();

	void simulate(real_t beamSize, real_t dx_s, bool addNoise=false);
	void orthogonalize();

	void ortho_modes();

	void normalize(CudaSmartPtr d_tmpComplex=CudaSmartPtr());
	real_t getMaxIntensity()				const {return m_maxIntensity;}
	Cuda3DArray<complex_t>* getModes()		const {return m_modes;}
	Cuda3DArray<real_t>* getIntensities()	const {return m_intensities;}
	void updateProbeEstimate(	const ICuda2DArray* object, const Cuda3DArray<complex_t>* psi, 
								const Cuda3DArray<complex_t>* psi_old, unsigned int qx, unsigned int qy,
								real_t objectMaxIntensity);
	void beginModalReconstruction();
	void endModalReconstruction();
	void updateIntensities(bool useSum=false);
	void updateMaxIntensity(bool useSum=false);

	void calc_object_norm(uint2 objectx, uint2 objecty, CudaSmartPtr objectArray);

	void updated_cached_illumination(std::vector <uint2> oROI1, std::vector <uint2> oROI2);

	void get_projections(CudaSmartPtr objectArray, Cuda3DArray<complex_t>* obj_proj, std::vector<int> ind_read, std::vector <uint2> oROI1, std::vector <uint2> oROI2);

	void get_projections(CudaSmartPtr objectArray, Cuda3DArray<real_t>* obj_proj, std::vector<int> ind_read, std::vector <uint2> oROI1, std::vector <uint2> oROI2);

	void get_projections_cpu(CudaSmartPtr objectArray, Cuda3DArray<complex_t>* obj_proj, std::vector<int> ind_read, std::vector < std::vector <int> > oROI_vec1,
			std::vector < std::vector <int> > oROI_vec2);

	void get_projections_cpu(CudaSmartPtr objectArray, Cuda3DArray<real_t>* obj_proj, std::vector<int> ind_read, std::vector < std::vector <int> > oROI_vec1,
			std::vector < std::vector <int> > oROI_vec2);

	void gradient_position_solver(Cuda3DArray<complex_t>* xi, Cuda3DArray<complex_t>* obj_proj, Cuda3DArray<complex_t>* varProbe, std::vector<int>& g_ind_vec,
			std::vector<float2>& positions_o, std::vector<float2>& probe_positions);

	void set_projections(CudaSmartPtr objectArray, Cuda3DArray<complex_t>* obj_proj, std::vector<int> ind_read, std::vector <uint2> oROI1, std::vector <uint2> oROI2);

//	void set_projections(CudaSmartPtr objectArray, CudaSmartPtr* obj_proj, std::vector<int> ind_read, std::vector <uint2> oROI1, std::vector <uint2> oROI2);

//	void get_illumination_probe(std::vector<int>& g_ind_vec, std::vector<float2>& sub_px_shift, Cuda3DArray<complex_t>* varProbe, Cuda3DElement<complex_t> psiElement,
//			Cuda3DArray<complex_t>* obj_proj);

	void get_illumination_probe(std::vector<int>& g_ind_vec, std::vector<float2>& sub_px_shift, Cuda3DArray<complex_t>* varProbe, std::vector < Cuda3DArray<complex_t>* >& psivec,
			Cuda3DArray<complex_t>* obj_proj, Cuda3DArray<real_t>* apsi);

	void shift_probe(std::vector<int>& g_ind_vec, std::vector<float2>& sub_px_shift, Cuda3DArray<complex_t>* varProbe);

	void update_probe(unsigned int ll, CudaSmartPtr probe_update_m, std::vector<int>& g_ind_vec);

	void update_variable_probe(CudaSmartPtr probe_update_m, Cuda3DArray<complex_t>* probe_update, Cuda3DArray<complex_t>* obj_proj, Cuda3DArray<complex_t>* psi,
			std::vector<int>& g_ind_vec, std::vector <uint2>& oROI1, std::vector <uint2>& oROI2, std::vector < std::vector <int> >& oROI_vec1,
			std::vector < std::vector <int> >& oROI_vec2);

	//For rendering purposes
	void fillResources();
	unsigned int getWidth() const;
	unsigned int getHeight()const;
	void toRGBA(float4*,const char*,float,float);
	void toGray(float*,const char*,bool=false);

//	double rand_gen();
//	double normalRandom();
	void printProbe(int column, int row);

};	

#endif /* PROBE_H_ */


