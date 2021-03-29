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
#include "Probe.h"
#include "Cuda2DArray.hpp"
#include "Probe.cuh"
#include "Diffractions.h"
#include "CudaSmartPtr.h"
#include "CXMath.h"

#include <cmath>
#include <algorithm>
#include <math_constants.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

using namespace std;

Probe::Probe(unsigned int size, unsigned int modes) :	m_modes(0),
														m_intensities(0),
														m_maxIntensity(1),
														m_randomGenerator(0),
														m_modesInitialized(false)
														
{
	m_modes = new Cuda3DArray<complex_t>(modes, make_uint2(size,size));
	m_intensities = new Cuda3DArray<real_t>(modes, make_uint2(size,size));

	m_MAX_ILLUM=0;
	m_modes->setUseAll(false);
	m_intensities->setUseAll(false);
}

Probe::~Probe()
{
	clear();
	if(m_randomGenerator)
		curandDestroyGenerator(m_randomGenerator);
}

void Probe::clear()
{
	if(m_modes) delete m_modes;
	if(m_intensities) delete m_intensities;
	m_modes = 0;
	m_intensities = 0;
}

CudaSmartPtr Probe::generateRandKernel(unsigned int x, unsigned int y)
{
	//Gaussian Smoothing of a random kernel
	CudaSmartPtr randArray(new Cuda2DArray<real_t>(x, y));

	if(!m_randomGenerator)
	{
		curandCreateGenerator(&m_randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(m_randomGenerator, time(NULL));
	}

	for(unsigned int x=0; x<randArray->getX(); ++x)
#ifdef USE_SINGLE_PRECISION
		curandGenerateUniform(m_randomGenerator, randArray->getDevicePtr<real_t>()+(x*randArray->getAlignedY()), randArray->getY());
#else
		curandGenerateUniformDouble(m_randomGenerator, randArray->getDevicePtr<real_t>()+(x*randArray->getAlignedY()), randArray->getY());
#endif


	PhaserUtil::getInstance()->gaussSmooth(randArray->getDevicePtr<real_t>(), 10, randArray->getX(), randArray->getY());
	h_normalize(randArray->getDevicePtr<real_t>(), randArray->getX(), randArray->getY(), randArray->getAlignedY());

	return randArray;
}

CudaSmartPtr Probe::generatepureRandKernel(unsigned int x, unsigned int y)
{
	CudaSmartPtr randArray(new Cuda2DArray<real_t>(x, y));

	if(!m_randomGenerator)
	{
		curandCreateGenerator(&m_randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(m_randomGenerator, time(NULL));
	}

	for(unsigned int x=0; x<randArray->getX(); ++x)
#ifdef USE_SINGLE_PRECISION
		curandGenerateUniform(m_randomGenerator, randArray->getDevicePtr<real_t>()+(x*randArray->getAlignedY()), randArray->getY());
#else
		curandGenerateUniformDouble(m_randomGenerator, randArray->getDevicePtr<real_t>()+(x*randArray->getAlignedY()), randArray->getY());
#endif

	return randArray;
}

void Probe::orthogonalize()
{
	//The Gram�Schmidt process for orthogonalizing the probe modes
	Cuda3DArray<complex_t> modesCopy(*m_modes);
	CudaSmartPtr temp = new Cuda2DArray<complex_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);
    for(unsigned int i=1; i<m_modes->getNum(); ++i)
		for(int j=i-1; j>=0; --j)
		{
			h_projectUtoV(m_modes->getAt(j).getDevicePtr(), modesCopy.getAt(i).getDevicePtr(), temp->getDevicePtr<complex_t>(),
							m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
			h_subtract(m_modes->getAt(i).getDevicePtr(), temp->getDevicePtr<complex_t>(), m_modes->getAt(i).getDevicePtr(),
							m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
		}
}

void Probe::ortho_modes()
{

	CudaSmartPtr tempR = new Cuda2DArray<real_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);
	CudaSmartPtr tempC = new Cuda2DArray<complex_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);
//	std::vector <real_t> vec;
	vector<pair<real_t, int> >vec;
    for(unsigned int i=1; i<=m_modes->getNum(); ++i)
    {
    	h_realComplexAbs(m_modes->getAt(i-1).getDevicePtr(), tempR->getDevicePtr<real_t>(),
    			m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY(), true);
    	real_t sumtemp=h_realSum(tempR->getDevicePtr<real_t>(), tempR->getX(), tempR->getY(), tempR->getAlignedY());
    	vec.push_back(make_pair(-sumtemp, i-1));
//    	vec.push_back(sumtemp);
		for(int j=1; j<i; j++)
		{	//           mx_j = mean(x{j}(:,:,:,1),3);
			//	           mx_i = mean(x{i}(:,:,:,1),3);
	    	h_realComplexAbs(m_modes->getAt(j-1).getDevicePtr(), tempR->getDevicePtr<real_t>(),
	    			m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY(), true);
	    	real_t sumRight=h_realSum(tempR->getDevicePtr<real_t>(), tempR->getX(), tempR->getY(), tempR->getAlignedY());

//	    	Cuda3DArray<complex_t> modesCopy(*m_modes);
			h_multiplyConju(m_modes->getAt(i-1).getDevicePtr(), m_modes->getAt(j-1).getDevicePtr(), tempC->getDevicePtr<complex_t>(),
					m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
			complex_t sumLeft=h_complexSum(tempC->getDevicePtr<complex_t>(), 0, tempC->getX(), 0, tempC->getY(), tempC->getAlignedY());

			complex_t proj=div_complex_t(sumLeft, make_complex_t(sumRight, 0));

			h_multiply(m_modes->getAt(j-1).getDevicePtr(), proj, tempC->getDevicePtr<complex_t>(), m_modes->getDimensions().x,
					m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
			h_subtract(m_modes->getAt(i-1).getDevicePtr(), tempC->getDevicePtr<complex_t>(), m_modes->getAt(i-1).getDevicePtr(),
							m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
		}

//				complex_t* pobjectHost=m_modes->getAt(i-1).getHostPtr();
//				int Npx=m_modes->getPtr()->getY();
//				for(int j=165; j<Npx; j++)
//				{
//					int offset=161*Npx;
//					printf("%d %.10e %.10e ", j, pobjectHost[offset+j].x, pobjectHost[offset+j].y);
//
//				}
//				printf("\n");

//				int temp=1;

    }

	sort (vec.begin(),vec.end());
    // Recopy the modesCopy back to the modes with correct order, m_modes has the  final probe
	Cuda3DArray<complex_t> modesCopy(*m_modes);
    for(unsigned int i=1; i<m_modes->getNum(); ++i)
	{
        h_switchprobe(m_modes->getAt(i).getDevicePtr(), modesCopy.getAt(vec[i].second).getDevicePtr(),
        		m_modes->getNum(), m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
	}

}

void Probe::updateIntensities(bool useSum)
{
	h_realComplexAbs(m_modes->getPtr()->getDevicePtr<complex_t>(), m_intensities->getPtr()->getDevicePtr<real_t>(),
				m_modes->getPtr()->getX(), m_modes->getPtr()->getY(), m_modes->getPtr()->getAlignedY(), true);
	updateMaxIntensity(useSum);
}

void Probe::normalize(CudaSmartPtr d_tmpComplex)
{
	CudaSmartPtr probesWavefront = (m_modes->getNum()==1||(!m_modes->checkUseAll()))  ? m_modes->getPtr() :
																d_tmpComplex.isValid()? d_tmpComplex : 
																new Cuda2DArray<complex_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);

	if(m_modes->checkUseAll() &&  m_modes->getNum() > 1)
		h_probeModalSum(m_modes->getPtr()->getDevicePtr<complex_t>(), probesWavefront->getDevicePtr<complex_t>(), m_modes->getNum(),
						probesWavefront->getX(), probesWavefront->getY(), probesWavefront->getAlignedY());
	h_realComplexAbs(probesWavefront->getDevicePtr<complex_t>(), m_intensities->getPtr()->getDevicePtr<real_t>(),
						probesWavefront->getX(), probesWavefront->getY(), probesWavefront->getAlignedY(), false);
	m_maxIntensity = PhaserUtil::getInstance()->getModalDoubleMax(m_intensities);
	h_normalize(m_modes->getPtr()->getDevicePtr<complex_t>(), m_modes->getPtr()->getX(),
						m_modes->getPtr()->getY(), m_modes->getPtr()->getAlignedY(), 1.0/m_maxIntensity);
	updateIntensities();
}

void Probe::initProbeModes()
{
	if(m_modes->getNum() > 1)
	{
		h_initModesFromBase(m_modes->getPtr()->getDevicePtr<complex_t>(), m_modes->getNum(), 0.05, m_intensities->getPtr()->getDevicePtr<real_t>(),
							m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY()); 
		orthogonalize();
		m_modesInitialized = true;
	}
	updateMaxIntensity();
}

void Probe::simulate(real_t beamSize, real_t dx_s, bool addNoise)
{
	CudaSmartPtr d_func = generateRandKernel(m_modes->getDimensions().x, m_modes->getDimensions().y);
	/*h_simulateProbe(m_modes->getAt(0).getDevicePtr(), d_func->getDevicePtr<real_t>(), m_modes->getDimensions().x/(2.0*2.35),
					beamSize, dx_s, m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());*/
	h_initProbe(m_modes->getAt(0).getDevicePtr(), d_func->getDevicePtr<real_t>(), m_modes->getDimensions().x/(2*2.35),
						beamSize, dx_s, m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY(), true);
	updateIntensities();
}

bool Probe::init(const Cuda3DArray<real_t>* diffractions, real_t beamSize, real_t dx_s, const char* filename)
{
	if(filename)
	{
		CudaSmartPtr d_probeGuess = new Cuda2DArray<complex_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);
		if(!d_probeGuess->load<complex_t>(filename))
		{
			fprintf(stderr,"Probe guess file (%s) not found.\n", filename);
			return false;
		}
		m_modes->getAt(0).setFromDevice(d_probeGuess->getDevicePtr<complex_t>(), d_probeGuess->getX(), d_probeGuess->getY());
		updateIntensities();
	}
	else
	{
		Cuda3DElement<real_t> avdata = m_intensities->getAt(0);

		h_initProbe(m_modes->getAt(0).getDevicePtr(), 0, m_modes->getDimensions().x/(2*2.35), beamSize, dx_s,
					m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY(), false);

		h_normalizeDiffractionIntensities(diffractions->getPtr()->getDevicePtr<real_t>(), avdata.getDevicePtr(),
											diffractions->getNum(), avdata.getX(), avdata.getY(), avdata.getAlignedY());

		PhaserUtil::getInstance()->applyModulusConstraint(m_modes, m_modes, avdata);
		normalize();
	}
	return true;
}

bool Probe::initMLH(unsigned int desiredShape, double lambda, double dx_recon, double beamSize, unsigned int nProbes, const char* filename)
{
	double Rn = 90e-6;
	double dRn = 50e-9;
	double fl = 2*Rn*dRn/lambda;
	double D_FZP = 180e-6;
	double D_H = 60e-6;
	double dx_fzp = lambda*fl/(desiredShape*dx_recon);
//	complex_t fzpValue = make_complex_t(0.0, -2*CUDART_PI/(lambda*2*fl));
	double fzpValue = -2/(lambda*2*fl);
	CudaSmartPtr temp = new Cuda2DArray<complex_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);

	// use the file name to load the initial probe guess to the m_modes
	if(filename)
	{
		CudaSmartPtr d_probeGuess = new Cuda2DArray<complex_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);
		if(!d_probeGuess->load<complex_t>(filename))
		{
			fprintf(stderr,"Probe guess file (%s) not found.\n", filename);
			return false;
		}
		m_modes->getAt(0).setFromDevice(d_probeGuess->getDevicePtr<complex_t>(), d_probeGuess->getX(), d_probeGuess->getY());

	}
	else
	{
		h_initProbeMLH(m_modes->getAt(0).getDevicePtr(), dx_fzp, fzpValue, D_FZP, D_H, lambda, fl, beamSize, temp->getDevicePtr<complex_t>(),
					   m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY(), false);
		Cuda3DElement<real_t> avdata = m_intensities->getAt(0);
		PhaserUtil::getInstance()->applyFFT(m_modes, m_modes, avdata);
		h_endProbeMLH(m_modes->getAt(0).getDevicePtr(), temp->getDevicePtr<complex_t>(),
					   m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY(), false);
	}

	// Nprobe init

	CudaSmartPtr tempProbe;
	tempProbe = new Cuda2DArray<complex_t>(m_modes->getDimensions().x,m_modes->getDimensions().y);

	/////////////////////////////// test with matlab code, uncomment h_initVarProbe
    float randna[]={0, 0.3362, 0.0803, 1.1553, 0.5397};
    float randnb[]={0, -0.3993, -1.2279, -1.3133, 0.2743};
	////////////////////////////////////

    for(unsigned int i=1; i<m_modes->getNum(); ++i)
	{
		h_initVarProbe(m_modes->getAt(i).getDevicePtr(), tempProbe->getDevicePtr<complex_t>(), m_modes->getAt(0).getDevicePtr(),
				m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY(), randna[i], randnb[i]);
//		h_initVarProbe(m_modes->getAt(i).getDevicePtr(), tempProbe->getDevicePtr<complex_t>(), m_modes->getAt(0).getDevicePtr(),
//				m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY(), normalRandom(), normalRandom());

	}

	Cuda3DArray<complex_t> modesCopy(*m_modes);
	CudaSmartPtr tempInner = new Cuda2DArray<complex_t>(1,m_modes->getDimensions().y);// Ussed to store the temporary value for sum

//	  std::complex<float> I(0.0, 1.0); // imaginary unit
	Eigen::MatrixXcf A= Eigen::MatrixXcf(m_modes->getNum(), m_modes->getNum()); // declare a real (double) 2x2 matrix
//	std::vector< std::vector<complex_t> > vect(m_modes->getNum(), std::vector<complex_t>(m_modes->getNum()));

    for(unsigned int i=0; i<m_modes->getNum(); ++i)
	{
        for(unsigned int j=0; j<m_modes->getNum(); ++j)
    	{
			complex_t resInner=h_Orth(m_modes->getAt(i).getDevicePtr(), modesCopy.getAt(j).getDevicePtr(), tempInner->getDevicePtr<complex_t>(),
							m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
			 std::complex<float> I(resInner.x, resInner.y);
			 A(i,j) = I;
    	}
	}

	Eigen::ComplexEigenSolver<Eigen::MatrixXcf > s;  // the instance s(A) includes the eigensystem
	s.compute(A);
	vector<pair<float,int> >vec;
	for(unsigned int i=0; i<m_modes->getNum(); ++i)
	{
		vec.push_back(make_pair(-abs(real(s.eigenvalues()(i))), i));
	}

	sort (vec.begin(),vec.end());

	CudaSmartPtr d_v = new Cuda2DArray<complex_t>(m_modes->getNum(), m_modes->getNum());
	complex_t* h_array = d_v->getHostPtr<complex_t>();

    for(unsigned int i=0; i<m_modes->getNum(); ++i)
	{
        for(unsigned int j=0; j<m_modes->getNum(); ++j)
    	{
        	complex_t resv=make_complex_t(real(s.eigenvectors()(i,j)), imag(s.eigenvectors()(i,j)));
        	h_array[i*m_modes->getNum()+j]=resv;
    	}
	}

    d_v->setFromHost<complex_t>(h_array, 1, m_modes->getNum()*m_modes->getNum());

    // Calculate m_modes and put the result into modesCopy
//    Cuda3DArray<complex_t> modesCopy2(*m_modes);

    for(unsigned int i=0; i<m_modes->getNum(); ++i)
	{
        h_orthro(m_modes->getPtr()->getDevicePtr<complex_t>(), modesCopy.getAt(i).getDevicePtr(), d_v->getDevicePtr<complex_t>(), i,
        		m_modes->getNum(), m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
	}

    // Recopy the modesCopy back to the modes with correct order, m_modes has the  final probe
    for(unsigned int i=0; i<m_modes->getNum(); ++i)
	{
        h_switchprobe(m_modes->getAt(vec[i].second).getDevicePtr(), modesCopy.getAt(i).getDevicePtr(),
        		m_modes->getNum(), m_modes->getDimensions().x, m_modes->getDimensions().y, m_modes->getPtr()->getAlignedY());
	}

	return true;

}

bool Probe::initVarModes()
{
	CudaSmartPtr randarr1 = generatepureRandKernel(m_modes->getDimensions().x, m_modes->getDimensions().y);
	CudaSmartPtr randarr2 = generatepureRandKernel(m_modes->getDimensions().x, m_modes->getDimensions().y);
    // Init the extra modes value

    m_extramodes = new Cuda2DArray<complex_t>(m_modes->getDimensions().x, m_modes->getDimensions().y);
    h_initVarModes(m_extramodes->getDevicePtr<complex_t>(), randarr1->getDevicePtr<real_t>(), randarr2->getDevicePtr<real_t>(), m_extramodes->getDimensions().x,
    		m_extramodes->getDimensions().y, m_extramodes->getAlignedY());
}

bool Probe::initEvo(int Npos, int variable_probe_modes, std::vector <uint2> oROI1, std::vector <uint2> oROI2, uint2 Np_o,
		CudaSmartPtr objectArray, const Cuda3DArray<real_t>* diffractions, std::vector<real_t> diffSquareRoot)
{

	m_probe_evolution.resize(variable_probe_modes+1);
	std::vector <double> probe_evolution(Npos, 1);
	m_probe_evolution[0]=probe_evolution;


	double sigma = 1.0;
	double Mi =  0;

	for(int j=1; j<(variable_probe_modes+1); j++)
	{
		for (int i=0; i<Npos; ++i)
			probe_evolution[i]=1e-6*(normalRandom()*sigma+Mi);
		m_probe_evolution[j]=probe_evolution;
	}

	/////////////////////////////////////
	// Load from file to test initVarModes has the real randome one test with matlab
//	char* filename1="/data2/JunjingData/probe21.csv";
//	char* filename2="/data2/JunjingData/probe22.csv";
//	m_extramodes->load2Complex<complex_t>(filename1, filename2);
//	char* filename3="/data2/JunjingData/probeevolution.csv";
//	std::vector <double> vec;
//	PhaserUtil::getInstance()->load<double>(filename3, vec);
//	m_probe_evolution[1]=vec;
	/////////////////////////////////////

	CudaSmartPtr d_result = new Cuda2DArray<real_t>(m_extramodes->getDimensions().x, m_extramodes->getDimensions().y);

	if(variable_probe_modes>0)
	{
		std::vector <double> pnorm;
		real_t result=0;
//		CudaSmartPtr d_result = new Cuda2DArray<real_t>(m_extramodes->getDimensions().x, m_extramodes->getDimensions().y);
		result=h_norm2(m_modes->getAt(0).getDevicePtr(), d_result->getDevicePtr<real_t>(), m_extramodes->getDimensions().x,
	    		m_extramodes->getDimensions().y, m_extramodes->getAlignedY());
		pnorm.push_back(result);

		result=h_norm2(m_extramodes->getDevicePtr<complex_t>(), d_result->getDevicePtr<real_t>(), m_extramodes->getDimensions().x,
	    		m_extramodes->getDimensions().y, m_extramodes->getAlignedY());
		h_normalizeVariProbe(m_extramodes->getDevicePtr<complex_t>(), result, m_extramodes->getDimensions().x,
	    		m_extramodes->getDimensions().y, m_extramodes->getAlignedY());

		for(int j=1; j<(variable_probe_modes+1); j++)
		{
			for (int i=0; i<Npos; ++i)
				m_probe_evolution[j][i]=m_probe_evolution[j][i]*result;
		}

		pnorm.push_back(result);

	}
	//p_object=illum_sum_0 d_result=aprobe2
	p_object = new Cuda2DArray<real_t>(Np_o.x, Np_o.y);
	p_object->set();
	CudaSmartPtr d_tmpObjResult = new Cuda2DArray<real_t>(Np_o.x, Np_o.y);
	CudaSmartPtr p_positions_x = new Cuda2DArray<unsigned int>(1, oROI1.size());
	unsigned int* h_p_positions_x = p_positions_x->getHostPtr<unsigned int>();
	CudaSmartPtr p_positions_y = new Cuda2DArray<unsigned int>(1, oROI2.size());
	unsigned int* h_p_positions_y = p_positions_y->getHostPtr<unsigned int>();
    for(unsigned int i=0; i<oROI1.size(); ++i)
	{
    	h_p_positions_x[i]=oROI1[i].x;
    	h_p_positions_y[i]=oROI2[i].x;
	}
    p_positions_x->setFromHost<unsigned int>(h_p_positions_x, 1, oROI1.size());
    p_positions_y->setFromHost<unsigned int>(h_p_positions_y, 1, oROI2.size());

	h_preCalillum(m_modes->getAt(0).getDevicePtr(), d_result->getDevicePtr<real_t>(), p_object->getDevicePtr<real_t>(), Npos, Np_o,
			p_positions_x->getDevicePtr<unsigned int>(), p_positions_y->getDevicePtr<unsigned int>(), m_extramodes->getDimensions().x,
			m_extramodes->getDimensions().y, m_extramodes->getAlignedY(), p_object->getAlignedY(), objectArray->getDevicePtr<complex_t>(),
			d_tmpObjResult->getDevicePtr<real_t>());

	h_realComplexAbs(m_modes->getPtr()->getDevicePtr<complex_t>(), m_intensities->getPtr()->getDevicePtr<real_t>(),
				m_modes->getPtr()->getX(), m_modes->getPtr()->getY(), m_modes->getPtr()->getAlignedY(), true);
    real_t resultillum=h_realSum(m_intensities->getPtr()->getDevicePtr<real_t>(), m_modes->getPtr()->getX(), m_modes->getPtr()->getY(), m_modes->getPtr()->getAlignedY());
    real_t meanillum=resultillum/(m_extramodes->getDimensions().x*m_extramodes->getDimensions().y);

    std::vector <real_t> corrvec;
    real_t diffDim=m_extramodes->getDimensions().x*m_extramodes->getDimensions().y;
    for(int i=0; i<diffractions->getNum(); ++i)
    {
    	corrvec.push_back(meanillum/(diffSquareRoot[i]/diffDim));
    }
    real_t corr=0;
    PhaserUtil::getInstance()->median<real_t>(corrvec, corr);

    corr=sqrt_real_t(corr*diffDim);
	h_normalize(m_modes->getPtr()->getDevicePtr<complex_t>(), m_modes->getPtr()->getX(),
						m_modes->getPtr()->getY(), m_modes->getPtr()->getAlignedY(), 1.0/corr);
	h_normalize(m_extramodes->getDevicePtr<complex_t>(), m_extramodes->getDimensions().x,
			m_extramodes->getDimensions().y, m_extramodes->getAlignedY(), 1.0/corr);
	h_normalize(p_object->getDevicePtr<real_t>(), corr*corr, Np_o.x, Np_o.y, p_object->getAlignedY());
//	h_multiplyReal(p_object->getDevicePtr<real_t>(), p_object->getDevicePtr<real_t>(), Np_o.x, Np_o.y, p_object->getAlignedY());

	MAX_ILLUM=h_maxFloat(p_object->getDevicePtr<real_t>(), Np_o.x, Np_o.y, p_object->getAlignedY());

//	h_squareRoot(d_result->getDevicePtr<real_t>(), d_result->getDevicePtr<real_t>(),
//			m_extramodes->getDimensions().x, m_extramodes->getDimensions().y, m_extramodes->getAlignedY());

	return true;

}

void Probe::remove_extra_degree()
{
	CudaSmartPtr d_result = new Cuda2DArray<real_t>(m_extramodes->getDimensions().x, m_extramodes->getDimensions().y);
	double vprobe_norm=0;
	vprobe_norm=h_norm2(m_extramodes->getDevicePtr<complex_t>(), d_result->getDevicePtr<real_t>(), m_extramodes->getDimensions().x,
		    		m_extramodes->getDimensions().y, m_extramodes->getAlignedY());
	h_normalizeVariProbe(m_extramodes->getDevicePtr<complex_t>(), vprobe_norm, m_extramodes->getDimensions().x,
    		m_extramodes->getDimensions().y, m_extramodes->getAlignedY());

	//self.probe_evolution(:,2:end) = self.probe_evolution(:,2:end) .* reshape(mean(vprobe_norm,3),1,[]);
	// TODO change to multiple variable probe
	int Npos=m_probe_evolution[0].size();
	for (int i=0; i<Npos; ++i)
		m_probe_evolution[1][i]=m_probe_evolution[1][i]*vprobe_norm;

	float2 meanPositions;
	double sum1x=0, sum1y=0;
	size_t len=m_probe_evolution.size();
	for(int i=0; i<len; i++)
	{
		sum1x += m_probe_evolution[0][i];
		sum1y += m_probe_evolution[1][i];
	}
	meanPositions.x=sum1x/len;
	meanPositions.y=sum1y/len;
	double sum=0;
	for(int i=0; i<len; i++)
	{
		m_probe_evolution[0][i]=(m_probe_evolution[0][i]-meanPositions.x)*0.99 + 1;
		m_probe_evolution[1][i]=m_probe_evolution[1][i]-meanPositions.y;
		sum+=pow(abs(m_probe_evolution[0][i]),2);
	}
//	cprobe_norm = norm2(self.probe_evolution(:,1));
	double cprobe_norm=sqrt_real_t(sum/len);
	for(int i=0; i<len; i++)
	{
		m_probe_evolution[0][i]=m_probe_evolution[0][i]/cprobe_norm;
	}

	h_normalize(m_modes->getAt(0).getDevicePtr(), m_modes->getDimensions().x, m_modes->getDimensions().y,
			m_modes->getPtr()->getAlignedY(), cprobe_norm);
	//TODO
    /*vprobe_norm = Ggather(norm2(self.probe{1}(:,:,:,2:end)));
    self.probe{1}(:,:,:,2:end) = self.probe{1}(:,:,:,2:end) ./ vprobe_norm;
    self.probe_evolution(:,2:end) = self.probe_evolution(:,2:end) .* reshape(mean(vprobe_norm,3),1,[]);
    % average variable mode correction is zero
    self.probe_evolution(ind,:) = bsxfun(@minus, self.probe_evolution(ind,:) , mean(self.probe_evolution(ind,:),1));
    % average intensity correction is zero

    % remove degrees of intensity correction and regularize a bit
    self.probe_evolution(ind,1) = self.probe_evolution(ind,1)*0.99 + 1;
    cprobe_norm = norm2(self.probe_evolution(:,1));
    self.probe_evolution(:,1) = self.probe_evolution(:,1) / cprobe_norm;
    self.probe{1}(:,:,:,1) = self.probe{1}(:,:,:,1) .* cprobe_norm;*/
}

void Probe::calc_object_norm(uint2 objectx, uint2 objecty, CudaSmartPtr objectArray)
{

	CudaSmartPtr d_W = new Cuda2DArray<real_t>(objectx.y-objectx.x+1, objecty.y-objecty.x+1);
	CudaSmartPtr d_temp = new Cuda2DArray<real_t>(objectx.y-objectx.x+1, objecty.y-objecty.x+1);
	//cache.illum_sum_0{ll}(cache.object_ROI{:});
	h_extracSubArrReal(p_object->getDevicePtr<real_t>(), d_W->getDevicePtr<real_t>(),
			objectx.x-1, objecty.x-1, d_W->getX(), d_W->getY(), d_W->getAlignedY(),
				p_object->getX(), p_object->getY(), p_object->getAlignedY());
	//norm2(W)
	real_t norm2Wresult=h_norm2Mat(d_W->getDevicePtr<real_t>(), d_temp->getDevicePtr<real_t>(), d_W->getX(), d_W->getY(), d_W->getAlignedY());

	CudaSmartPtr d_roi = new Cuda2DArray<complex_t>(objectx.y-objectx.x+1, objecty.y-objecty.x+1);
	h_extracSubArrComplex(objectArray->getDevicePtr<complex_t>(), d_roi->getDevicePtr<complex_t>(),
			objectx.x-1, objecty.x-1, d_roi->getX(), d_roi->getY(), d_roi->getAlignedY(),
			objectArray->getX(), objectArray->getY(), objectArray->getAlignedY());

	h_multiply(d_W->getDevicePtr<real_t>(), d_roi->getDevicePtr<complex_t>(), d_roi->getDevicePtr<complex_t>(), d_roi->getX(), d_roi->getY(), d_roi->getAlignedY());

	real_t norm2Roiresult=h_norm2Mat(d_roi->getDevicePtr<complex_t>(), d_temp->getDevicePtr<real_t>(), d_roi->getX(), d_roi->getY(), d_roi->getAlignedY());
	real_t object_norm=norm2Roiresult/norm2Wresult;
	//self.probe{ll} = self.probe{ll} * object_norm ;
	h_normalize(m_modes->getPtr()->getDevicePtr<complex_t>(), m_modes->getPtr()->getX(),
						m_modes->getPtr()->getY(), m_modes->getPtr()->getAlignedY(), object_norm);
	h_normalize(m_extramodes->getDevicePtr<complex_t>(), m_extramodes->getDimensions().x,
			m_extramodes->getDimensions().y, m_extramodes->getAlignedY(), object_norm);
	//self.object{ll} = self.object{ll} / object_norm ;
	h_normalize(objectArray->getDevicePtr<complex_t>(), objectArray->getX(), objectArray->getY(), objectArray->getAlignedY(), 1.0/object_norm);
}

void Probe::updated_cached_illumination(std::vector <uint2> oROI1, std::vector <uint2> oROI2)
{
	CudaSmartPtr aprobe2 = new Cuda2DArray<real_t>(m_extramodes->getDimensions().x, m_extramodes->getDimensions().y);
	h_realComplexAbs(m_modes->getAt(0).getDevicePtr(), aprobe2->getDevicePtr<real_t>(), m_extramodes->getDimensions().x,
			m_extramodes->getDimensions().y, m_extramodes->getAlignedY(), true);

	CudaSmartPtr p_positions_x = new Cuda2DArray<unsigned int>(1, oROI1.size());
	unsigned int* h_p_positions_x = p_positions_x->getHostPtr<unsigned int>();
	CudaSmartPtr p_positions_y = new Cuda2DArray<unsigned int>(1, oROI2.size());
	unsigned int* h_p_positions_y = p_positions_y->getHostPtr<unsigned int>();
    for(unsigned int i=0; i<oROI1.size(); ++i)
	{
    	h_p_positions_x[i]=oROI1[i].x;
    	h_p_positions_y[i]=oROI2[i].x;
	}
    p_positions_x->setFromHost<unsigned int>(h_p_positions_x, 1, oROI1.size());
    p_positions_y->setFromHost<unsigned int>(h_p_positions_y, 1, oROI2.size());

	unsigned int Npos = oROI1.size();
	// set to p_object
	h_set_projections(p_object->getDevicePtr<real_t>(), aprobe2->getDevicePtr<real_t>(), p_positions_x->getDevicePtr<unsigned int>(), p_positions_y->getDevicePtr<unsigned int>(),
			p_object->getX(), p_object->getY(), p_object->getAlignedY(), aprobe2->getX(), aprobe2->getY(), Npos);
	h_normalize(p_object->getDevicePtr<real_t>(), 2.0, p_object->getX(), p_object->getY(), p_object->getAlignedY());

	CudaSmartPtr d_temp = new Cuda2DArray<real_t>(p_object->getX(), p_object->getY());
	real_t illum_norm=h_norm2Mat(p_object->getDevicePtr<real_t>(), d_temp->getDevicePtr<real_t>(), p_object->getX(), p_object->getY(), p_object->getAlignedY());
	m_MAX_ILLUM=h_maxFloat(p_object->getDevicePtr<real_t>(), p_object->getX(), p_object->getY(), p_object->getAlignedY());

}

void Probe::set_projections(CudaSmartPtr objectArray, Cuda3DArray<complex_t>* obj_proj, std::vector<int> ind_read, std::vector <uint2> oROI1, std::vector <uint2> oROI2)
{
	CudaSmartPtr p_positions_x = new Cuda2DArray<unsigned int>(1, ind_read.size());
	unsigned int* h_p_positions_x = p_positions_x->getHostPtr<unsigned int>();
	CudaSmartPtr p_positions_y = new Cuda2DArray<unsigned int>(1, ind_read.size());
	unsigned int* h_p_positions_y = p_positions_y->getHostPtr<unsigned int>();
    for(int i=0; i<ind_read.size(); ++i)
	{
    	h_p_positions_x[i]=oROI1[ind_read[i]].x;
    	h_p_positions_y[i]=oROI2[ind_read[i]].x;
	}
    p_positions_x->setFromHost<unsigned int>(h_p_positions_x, 1, ind_read.size());
    p_positions_y->setFromHost<unsigned int>(h_p_positions_y, 1, ind_read.size());

//	h_set_projections(objectArray->getDevicePtr<complex_t>(), obj_proj->getPtr()->getDevicePtr<complex_t>(),
//			p_positions_x->getDevicePtr<unsigned int>(), p_positions_y->getDevicePtr<unsigned int>(),
//			objectArray->getX(), objectArray->getY(), objectArray->getAlignedY(), obj_proj->getDimensions().x, obj_proj->getDimensions().y,
//			obj_proj->getPtr()->getAlignedY(), ind_read.size(), false);

    // Change for ndims=2 or 3
    bool isFlat=false;
    if(obj_proj->getNum()==1)
    	isFlat=true;

	h_set_projections(objectArray->getDevicePtr<complex_t>(), obj_proj->getPtr()->getDevicePtr<complex_t>(),
			p_positions_x->getDevicePtr<unsigned int>(), p_positions_y->getDevicePtr<unsigned int>(),
			objectArray->getX(), objectArray->getY(), objectArray->getAlignedY(), obj_proj->getDimensions().x, obj_proj->getDimensions().y,
			obj_proj->getPtr()->getAlignedY(), ind_read.size(), isFlat);

}
//void set_projections(CudaSmartPtr objectArray, CudaSmartPtr* obj_proj, std::vector<int> ind_read, std::vector <uint2> oROI1, std::vector <uint2> oROI2)
//{
//	CudaSmartPtr p_positions_x = new Cuda2DArray<unsigned int>(1, ind_read.size());
//	unsigned int* h_p_positions_x = p_positions_x->getHostPtr<unsigned int>();
//	CudaSmartPtr p_positions_y = new Cuda2DArray<unsigned int>(1, ind_read.size());
//	unsigned int* h_p_positions_y = p_positions_y->getHostPtr<unsigned int>();
//    for(int i=0; i<ind_read.size(); ++i)
//	{
//    	h_p_positions_x[i]=oROI1[ind_read[i]].x;
//    	h_p_positions_y[i]=oROI2[ind_read[i]].x;
//	}
//    p_positions_x->setFromHost<unsigned int>(h_p_positions_x, 1, ind_read.size());
//    p_positions_y->setFromHost<unsigned int>(h_p_positions_y, 1, ind_read.size());
//
//	h_set_projections(objectArray->getDevicePtr<complex_t>(), obj_proj->getDevicePtr(),
//			p_positions_x->getDevicePtr<unsigned int>(), p_positions_y->getDevicePtr<unsigned int>(),
//			objectArray->getX(), objectArray->getY(), objectArray->getAlignedY(), obj_proj->getDimensions().x, obj_proj->getDimensions().y,
//			obj_proj->getPtr()->getAlignedY(), ind_read.size(), false);
//}

void Probe::get_projections_cpu(CudaSmartPtr objectArray, Cuda3DArray<complex_t>* obj_proj, std::vector<int> ind_read, std::vector < std::vector <int> > oROI_vec1,
		std::vector < std::vector <int> > oROI_vec2)
{

	complex_t* h_objectArray=objectArray->getHostPtr<complex_t>();
	int onpy=objectArray->getY();
	int index=ind_read[0];
	complex_t* h_obj_proj=obj_proj->getPtr()->getHostPtr<complex_t>();
	int pnpx=obj_proj->getPtr()->getX();
	int pnpy=obj_proj->getPtr()->getY();

	for(int i=0; i<oROI_vec1[index].size(); i++)
	{
		for(int j=0; j<oROI_vec2[index].size(); j++)
		{
			complex_t temp=h_objectArray[(oROI_vec1[index][i]-1)*onpy+oROI_vec2[index][j]-1];
			h_obj_proj[i*pnpy+j]=temp;
		}
	}

	obj_proj->getPtr()->setFromHost<complex_t>(h_obj_proj, pnpx, pnpx);

}

void Probe::get_projections_cpu(CudaSmartPtr objectArray, Cuda3DArray<real_t>* obj_proj, std::vector<int> ind_read, std::vector < std::vector <int> > oROI_vec1,
		std::vector < std::vector <int> > oROI_vec2)
{

	real_t* h_objectArray=objectArray->getHostPtr<real_t>();
	int onpy=objectArray->getY();
	int index=ind_read[0];
	real_t* h_obj_proj=obj_proj->getPtr()->getHostPtr<real_t>();
	int pnpx=obj_proj->getPtr()->getX();
	int pnpy=obj_proj->getPtr()->getY();

	for(int i=0; i<oROI_vec1[index].size(); i++)
	{
		for(int j=0; j<oROI_vec2[index].size(); j++)
		{
			real_t temp=h_objectArray[(oROI_vec1[index][i]-1)*onpy+oROI_vec2[index][j]-1];
			h_obj_proj[i*pnpy+j]=temp;
		}
	}
	obj_proj->getPtr()->setFromHost<real_t>(h_obj_proj, pnpx, pnpx);

}

void Probe::get_projections(CudaSmartPtr objectArray, Cuda3DArray<complex_t>* obj_proj, std::vector<int> ind_read, std::vector <uint2> oROI1, std::vector <uint2> oROI2)
{
	CudaSmartPtr p_positions_x = new Cuda2DArray<unsigned int>(1, ind_read.size());
	unsigned int* h_p_positions_x = p_positions_x->getHostPtr<unsigned int>();
	CudaSmartPtr p_positions_y = new Cuda2DArray<unsigned int>(1, ind_read.size());
	unsigned int* h_p_positions_y = p_positions_y->getHostPtr<unsigned int>();
    for(int i=0; i<ind_read.size(); ++i)
	{
    	h_p_positions_x[i]=oROI1[ind_read[i]].x;
    	h_p_positions_y[i]=oROI2[ind_read[i]].x;
	}
    p_positions_x->setFromHost<unsigned int>(h_p_positions_x, 1, ind_read.size());
    p_positions_y->setFromHost<unsigned int>(h_p_positions_y, 1, ind_read.size());

    // Changed when obj_proj==1

	h_get_projections(objectArray->getDevicePtr<complex_t>(), obj_proj->getPtr()->getDevicePtr<complex_t>(),
			p_positions_x->getDevicePtr<unsigned int>(), p_positions_y->getDevicePtr<unsigned int>(),
			objectArray->getX(), objectArray->getY(), objectArray->getAlignedY(), obj_proj->getDimensions().x, obj_proj->getDimensions().y, obj_proj->getNum(),
			obj_proj->getPtr()->getAlignedY(), ind_read.size());

//	if(obj_proj->getNum()==1)
//	{
//
//
//	}
//	else
//	{
//		h_get_projections(objectArray->getDevicePtr<complex_t>(), obj_proj->getPtr()->getDevicePtr<complex_t>(),
//				p_positions_x->getDevicePtr<unsigned int>(), p_positions_y->getDevicePtr<unsigned int>(),
//				objectArray->getX(), objectArray->getY(), objectArray->getAlignedY(), obj_proj->getDimensions().x, obj_proj->getDimensions().y, obj_proj->getNum(),
//				obj_proj->getPtr()->getAlignedY(), ind_read.size());
//
//	}

}

void Probe::get_projections(CudaSmartPtr objectArray, Cuda3DArray<real_t>* obj_proj, std::vector<int> ind_read, std::vector <uint2> oROI1, std::vector <uint2> oROI2)
{
	CudaSmartPtr p_positions_x = new Cuda2DArray<unsigned int>(1, ind_read.size());
	unsigned int* h_p_positions_x = p_positions_x->getHostPtr<unsigned int>();
	CudaSmartPtr p_positions_y = new Cuda2DArray<unsigned int>(1, ind_read.size());
	unsigned int* h_p_positions_y = p_positions_y->getHostPtr<unsigned int>();
    for(int i=0; i<ind_read.size(); ++i)
	{
    	h_p_positions_x[i]=oROI1[ind_read[i]].x;
    	h_p_positions_y[i]=oROI2[ind_read[i]].x;
	}
    p_positions_x->setFromHost<unsigned int>(h_p_positions_x, 1, ind_read.size());
    p_positions_y->setFromHost<unsigned int>(h_p_positions_y, 1, ind_read.size());


	h_get_projections(objectArray->getDevicePtr<real_t>(), obj_proj->getPtr()->getDevicePtr<real_t>(),
			p_positions_x->getDevicePtr<unsigned int>(), p_positions_y->getDevicePtr<unsigned int>(),
			objectArray->getX(), objectArray->getY(), objectArray->getAlignedY(), obj_proj->getDimensions().x, obj_proj->getDimensions().y, obj_proj->getNum(),
			obj_proj->getPtr()->getAlignedY(), ind_read.size());

}

void Probe::get_illumination_probe(std::vector<int>& g_ind_vec, std::vector<float2>& sub_px_shift, Cuda3DArray<complex_t>* varProbe, std::vector < Cuda3DArray<complex_t>* >& psivec,
		Cuda3DArray<complex_t>* obj_proj, Cuda3DArray<real_t>* apsi)
{
	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
//	CudaSmartPtr d_probe_evolution = new Cuda2DArray<real_t>(m_probe_evolution.size(), g_ind_vec.size());
//	real_t* h_probe_evolutio = d_probe_evolution->getHostPtr<real_t>();

	for(int i=0; i<m_modes->getNum(); i++)
	{
		if((i==0)&&(rParams->variable_probe||rParams->variable_intensity))
		{
			std::vector < std::vector <double> > probe_evolution;
			probe_evolution.resize(m_probe_evolution.size());

			for(int j=0; j<g_ind_vec.size(); j++)
			{
				for(int p=0; p<probe_evolution.size();p++)
					probe_evolution[p].push_back(m_probe_evolution[p][g_ind_vec[j]]);
			}

			for(int j=0; j<g_ind_vec.size(); j++)
			{
				double leftFactor=probe_evolution[0][j];
				double rightFactor=probe_evolution[1][j];

				h_complexSum(m_modes->getAt(0).getDevicePtr(), m_extramodes->getDevicePtr<complex_t>(), varProbe->getAt(j).getDevicePtr(), leftFactor, rightFactor,
						m_extramodes->getDimensions().x, m_extramodes->getDimensions().y, m_extramodes->getAlignedY());
			}
		}
		else
		{
			//TODO the probe just does not change probevec= m_modes
		}
	}
// % Nlayers
	Cuda3DElement<real_t> avdata = m_intensities->getAt(0);
	int Npx=apsi->getDimensions().x;
	int indSize=g_ind_vec.size();
	Cuda3DArray<real_t>* tmpgrid=new Cuda3DArray<real_t>(indSize, make_uint2(Npx,Npx));
	for(int i=0; i<m_modes->getNum(); i++)
	{
		if((i==0)&&rParams->apply_subpix_shift)
		{// varProbe=probe1 other probe just keep the same 26*256*256
			shift_probe(g_ind_vec, sub_px_shift, varProbe);
			CXMath::multiply<complex_t>(obj_proj, varProbe, psivec[i]);
		}
		else
		{
			CXMath::multiply<complex_t>(m_modes->getAt(i), obj_proj, psivec[i]);

		}
		// FFT2 psi
		PhaserUtil::getInstance()->ff2Mat(psivec[i], psivec[i], avdata);
		//
		h_realComplexAbs(psivec[i]->getPtr()->getDevicePtr<complex_t>(), tmpgrid->getPtr()->getDevicePtr<real_t>(),
				psivec[i]->getPtr()->getX(), psivec[i]->getPtr()->getY(), psivec[i]->getPtr()->getAlignedY(), true);
		//get intensity (modulus) on detector including different corrections
		h_realSum(apsi->getPtr()->getDevicePtr<real_t>(), tmpgrid->getPtr()->getDevicePtr<real_t>(), apsi->getPtr()->getDevicePtr<real_t>(), 1.0, 1.0,
				apsi->getPtr()->getX(), apsi->getPtr()->getY(), apsi->getPtr()->getAlignedY());
	}
	h_squareRoot(apsi->getPtr()->getDevicePtr<real_t>(), apsi->getPtr()->getDevicePtr<real_t>(),
			apsi->getPtr()->getX(), apsi->getPtr()->getY(), apsi->getPtr()->getAlignedY());

	delete tmpgrid;

}

void Probe::gradient_position_solver(Cuda3DArray<complex_t>* xi, Cuda3DArray<complex_t>* obj_proj, Cuda3DArray<complex_t>* varProbe, std::vector<int>& g_ind_vec,
		std::vector<float2>& positions_o, std::vector<float2>& probe_positions)
{
	int Npx=obj_proj->getDimensions().x;
	int Npy=obj_proj->getDimensions().y;
	// p_positions_x=X
	CudaSmartPtr p_positions_x = new Cuda2DArray<real_t>(1, Npy);
	real_t* h_p_positions_x = p_positions_x->getHostPtr<real_t>();
	for(int i=0;i<Npy;i++)
		h_p_positions_x[i]=(i*1.0/Npy)-0.5;
    p_positions_x->setFromHost<real_t>(h_p_positions_x, 1, Npy);
	// p_positions_y=Y
	CudaSmartPtr p_positions_y = new Cuda2DArray<real_t>(1, Npx);
	real_t* h_p_positions_y = p_positions_y->getHostPtr<real_t>();
	for(int i=0;i<Npx;i++)
		h_p_positions_y[i]=(i*1.0/Npx)-0.5;
	p_positions_y->setFromHost<real_t>(h_p_positions_y, 1, Npx);

    h_shiftFFTy(p_positions_x->getDevicePtr<real_t>(), p_positions_x->getDevicePtr<real_t>(), p_positions_x->getX(), p_positions_x->getY(), p_positions_x->getAlignedY());
    h_shiftFFTy(p_positions_y->getDevicePtr<real_t>(), p_positions_y->getDevicePtr<real_t>(), p_positions_y->getX(), p_positions_y->getY(), p_positions_y->getAlignedY());
    //img = fft2(img);
    Cuda3DArray<complex_t>* img=new Cuda3DArray<complex_t>(obj_proj->getNum(), obj_proj->getDimensions());
    Cuda3DArray<complex_t>* dX=new Cuda3DArray<complex_t>(obj_proj->getNum(), obj_proj->getDimensions());
    Cuda3DArray<complex_t>* dY=new Cuda3DArray<complex_t>(obj_proj->getNum(), obj_proj->getDimensions());

	Cuda3DElement<real_t> tempDet = m_intensities->getAt(0);
	PhaserUtil::getInstance()->ff2Mat(obj_proj, img, tempDet);

	//[dX, dY] = Gfun(@multiply_gfun, img, X, Y);
	complex_t factor=make_complex_t(0, 2.0*CUDART_PI);
	h_multiply(img->getPtr()->getDevicePtr<complex_t>(), factor, img->getPtr()->getDevicePtr<complex_t>(), img->getPtr()->getX(),
			img->getPtr()->getY(), img->getPtr()->getAlignedY());
	// The problem is the 3d multiply by a 2d TODO change to a loop for iteratio
	h_multiplyRow(img->getPtr()->getDevicePtr<complex_t>(), p_positions_x->getDevicePtr<real_t>(), dX->getPtr()->getDevicePtr<complex_t>(), dX->getPtr()->getX(),
			dX->getPtr()->getY(), dX->getPtr()->getAlignedY());

	for(int i=0; i<obj_proj->getNum(); i++)
	{
		h_multiplyColumn(img->getAt(i).getDevicePtr(), p_positions_y->getDevicePtr<real_t>(), dY->getAt(i).getDevicePtr(), dY->getDimensions().x,
				dY->getDimensions().y, dY->getPtr()->getAlignedY());
	}

	//ifft2mat, cudafft has to normailze by the totall number of elments, here is 256*256 varProbe->getPtr()->getX()=256*26
	PhaserUtil::getInstance()->iff2Mat(dX, dX, tempDet);
	h_normalize(dX->getPtr()->getDevicePtr<complex_t>(), dX->getPtr()->getX(),
			dX->getPtr()->getY(), dX->getPtr()->getAlignedY(), 1.0/(Npx*Npy));
	PhaserUtil::getInstance()->iff2Mat(dY, dY, tempDet);
	h_normalize(dY->getPtr()->getDevicePtr<complex_t>(), dY->getPtr()->getX(),
			dY->getPtr()->getY(), dY->getPtr()->getAlignedY(), 1.0/(Npx*Npy));

	Cuda3DArray<complex_t>* tempArrC=new Cuda3DArray<complex_t>(obj_proj->getNum(), obj_proj->getDimensions());
	Cuda3DArray<real_t>* nom=new Cuda3DArray<real_t>(obj_proj->getNum(), obj_proj->getDimensions());
	Cuda3DArray<real_t>* denom=new Cuda3DArray<real_t>(obj_proj->getNum(), obj_proj->getDimensions());
	double cutoff=std::numeric_limits<double>::max();
	for(int i=0; i<obj_proj->getNum(); i++)
	{
		h_multiply(dX->getAt(i).getDevicePtr(), varProbe->getAt(i).getDevicePtr(), tempArrC->getAt(i).getDevicePtr(),
				tempArrC->getDimensions().x, tempArrC->getDimensions().y, tempArrC->getPtr()->getAlignedY());
		h_realComplexAbs(tempArrC->getAt(i).getDevicePtr(), denom->getAt(i).getDevicePtr(), denom->getDimensions().x,
				denom->getDimensions().y, denom->getPtr()->getAlignedY(), true);
		h_multiplyConju(xi->getAt(i).getDevicePtr(), tempArrC->getAt(i).getDevicePtr(), tempArrC->getAt(i).getDevicePtr(),
				tempArrC->getDimensions().x, tempArrC->getDimensions().y, tempArrC->getPtr()->getAlignedY());
		h_realComplexReal(tempArrC->getAt(i).getDevicePtr(), nom->getAt(i).getDevicePtr(), nom->getDimensions().x, nom->getDimensions().y,
				nom->getPtr()->getAlignedY());

		double sum2nom1=h_realSum(nom->getAt(i).getDevicePtr(), nom->getDimensions().x, nom->getDimensions().y, nom->getPtr()->getAlignedY());
		double sum2denom1=h_realSum(denom->getAt(i).getDevicePtr(), denom->getDimensions().x, denom->getDimensions().y, denom->getPtr()->getAlignedY());
		double dx=sum2nom1/sum2denom1;

		h_multiply(dY->getAt(i).getDevicePtr(), varProbe->getAt(i).getDevicePtr(), tempArrC->getAt(i).getDevicePtr(),
				tempArrC->getDimensions().x, tempArrC->getDimensions().y, tempArrC->getPtr()->getAlignedY());
		h_realComplexAbs(tempArrC->getAt(i).getDevicePtr(), denom->getAt(i).getDevicePtr(), denom->getDimensions().x,
				denom->getDimensions().y, denom->getPtr()->getAlignedY(), true);

		h_multiplyConju(xi->getAt(i).getDevicePtr(), tempArrC->getAt(i).getDevicePtr(), tempArrC->getAt(i).getDevicePtr(),
				tempArrC->getDimensions().x, tempArrC->getDimensions().y, tempArrC->getPtr()->getAlignedY());
		h_realComplexReal(tempArrC->getAt(i).getDevicePtr(), nom->getAt(i).getDevicePtr(), nom->getDimensions().x, nom->getDimensions().y,
				nom->getPtr()->getAlignedY());

		double sum2nom2=h_realSum(nom->getAt(i).getDevicePtr(), nom->getDimensions().x, nom->getDimensions().y, nom->getPtr()->getAlignedY());
		double sum2denom2=h_realSum(denom->getAt(i).getDevicePtr(), denom->getDimensions().x, denom->getDimensions().y, denom->getPtr()->getAlignedY());
		double dy=sum2nom2/sum2denom2;

		int signx=PhaserUtil::getInstance()->sgn(dx);
		int signy=PhaserUtil::getInstance()->sgn(dy);
		double shiftx=std::min(abs(dx),0.2)*signx;
		double shifty=std::min(abs(dy),0.2)*signy;
		int index=g_ind_vec[i];
		//diff = sqrt(sum((mode.probe_positions(ind,:)+shift - position0).^2,2));
		double diff=sqrt_real_t(pow((probe_positions[index].x+shiftx-positions_o[index].x), 2)+pow((probe_positions[index].y+shifty-positions_o[index].y), 2));
		if(diff>cutoff)
		{
			shiftx=0;
			shifty=0;
		}

//		printf("shiftx is %.10e, shifty is %.10e \n", shiftx, shifty);

		probe_positions[index].x=probe_positions[index].x+shiftx;
		probe_positions[index].y=probe_positions[index].y+shifty;
	}

	delete img;
	delete dX;
	delete dY;
	delete tempArrC;
	delete nom;
	delete denom;
}

void Probe::shift_probe(std::vector<int>& g_ind_vec, std::vector<float2>& sub_px_shift, Cuda3DArray<complex_t>* varProbe)
{
	int Npx=m_extramodes->getDimensions().x;
	int indSize=g_ind_vec.size();

	CudaSmartPtr p_x = new Cuda2DArray<real_t>(indSize, Npx);
	real_t* h_p_x = p_x->getHostPtr<real_t>();
	for(int j=0; j<indSize; j++)
	{
		for(int p=0;p<Npx;p++)
			h_p_x[j*Npx+p]=sub_px_shift[g_ind_vec[j]].x;
	}
	p_x->setFromHost<real_t>(h_p_x, indSize, Npx);

	CudaSmartPtr p_y = new Cuda2DArray<real_t>(indSize, Npx);
	real_t* h_p_y = p_y->getHostPtr<real_t>();
	for(int j=0; j<indSize; j++)
	{
		for(int p=0;p<Npx;p++)
			h_p_y[j*Npx+p]=sub_px_shift[g_ind_vec[j]].y;
	}
	p_y->setFromHost<real_t>(h_p_y, indSize, Npx);

	// FFT2 varProbe=img
	Cuda3DElement<real_t> avdata = m_intensities->getAt(0);
	PhaserUtil::getInstance()->ff2Mat(varProbe, varProbe, avdata);

	CudaSmartPtr p_positions_x = new Cuda2DArray<real_t>(indSize, Npx);
	real_t* h_p_positions_x = p_positions_x->getHostPtr<real_t>();
	for(int j=0; j<indSize; j++)
	{
		for(int p=0;p<Npx;p++)
			h_p_positions_x[j*Npx+p]=p*1.0/Npx-0.5;
	}
    p_positions_x->setFromHost<real_t>(h_p_positions_x, indSize, Npx);
    CudaSmartPtr grid = new Cuda2DArray<real_t>(indSize, Npx);
    h_shiftFFTy(p_positions_x->getDevicePtr<real_t>(), grid->getDevicePtr<real_t>(), grid->getX(), grid->getY(), grid->getAlignedY());

    CudaSmartPtr xgrid = new Cuda2DArray<real_t>(indSize, Npx);
    h_multiply(p_x->getDevicePtr<real_t>(), grid->getDevicePtr<real_t>(), xgrid->getDevicePtr<real_t>(), xgrid->getX(), xgrid->getY(), xgrid->getAlignedY());

    CudaSmartPtr ygrid = new Cuda2DArray<real_t>(indSize, Npx);
    h_multiply(p_y->getDevicePtr<real_t>(), grid->getDevicePtr<real_t>(), ygrid->getDevicePtr<real_t>(), ygrid->getX(), ygrid->getY(), ygrid->getAlignedY());

	Cuda3DArray<real_t>* allgrid=new Cuda3DArray<real_t>(indSize, make_uint2(Npx,Npx));
	for(int i=0; i<allgrid->getNum(); i++)
	{
		h_realSingleSum(xgrid->getDevicePtr<real_t>()+(size_t)i*(size_t)xgrid->getAlignedY(), ygrid->getDevicePtr<real_t>()+(size_t)i*(size_t)ygrid->getAlignedY(),
				allgrid->getAt(i).getDevicePtr(), allgrid->getDimensions().x, allgrid->getDimensions().y, allgrid->getPtr()->getAlignedY());
	}

	Cuda3DArray<complex_t>* tmpgrid=new Cuda3DArray<complex_t>(indSize, make_uint2(Npx,Npx));
	for(int i=0; i<allgrid->getNum(); i++)
	{
		h_realComplexExp(allgrid->getAt(i).getDevicePtr(), tmpgrid->getAt(i).getDevicePtr(), allgrid->getDimensions().x, allgrid->getDimensions().y,
				allgrid->getPtr()->getAlignedY(), (-2.0*CUDART_PI));
	}

	CXMath::multiply<complex_t>(varProbe, tmpgrid, varProbe);

	PhaserUtil::getInstance()->iff2Mat(varProbe, varProbe, avdata);
	h_normalize(varProbe->getPtr()->getDevicePtr<complex_t>(), varProbe->getPtr()->getX(),
			varProbe->getPtr()->getY(), varProbe->getPtr()->getAlignedY(), 1.0/(Npx*Npx));

	delete allgrid;
	delete tmpgrid;


}

void Probe::update_probe(unsigned int ll, CudaSmartPtr probe_update_m, std::vector<int>& g_ind_vec)
{

	complex_t objSum=make_complex_t(0, 0);
	int No=0;
	for(int i=0; i<g_ind_vec.size(); i++)
	{
		objSum=add_complex_t(objSum, beta_probevec[g_ind_vec[i]]);
		No++;
	}
	complex_t objMean=div_complex_t(objSum, make_complex_t(No, 0));

	CudaSmartPtr tmpResult=new Cuda2DArray<complex_t>(probe_update_m->getX(), probe_update_m->getY());
	h_multiply(probe_update_m->getDevicePtr<complex_t>(), objMean, tmpResult->getDevicePtr<complex_t>(),
			probe_update_m->getX(), probe_update_m->getY(), probe_update_m->getAlignedY());

	h_complexSum(m_modes->getAt(ll-1).getDevicePtr(), tmpResult->getDevicePtr<complex_t>(), m_modes->getAt(ll-1).getDevicePtr(), 1.0, 1.0,
			tmpResult->getDimensions().x, tmpResult->getDimensions().y, tmpResult->getAlignedY());

	tmpResult->set();

}

void Probe::update_variable_probe(CudaSmartPtr probe_update_m, Cuda3DArray<complex_t>* probe_update, Cuda3DArray<complex_t>* obj_proj, Cuda3DArray<complex_t>* chi,
		std::vector<int>& g_ind_vec, std::vector <uint2>& oROI1, std::vector <uint2>& oROI2, std::vector < std::vector <int> >& oROI_vec1,
		std::vector < std::vector <int> >& oROI_vec2)
{

	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
	int block_size=g_ind_vec.size();
	int Npos=oROI1.size();
	std::vector <double> probe_evol;
	double probe_evolNorm=0;

	Cuda3DArray<real_t>* weight_proj=new Cuda3DArray<real_t>(g_ind_vec.size(), obj_proj->getDimensions());
	Cuda3DArray<complex_t>* resid=new Cuda3DArray<complex_t>(g_ind_vec.size(), obj_proj->getDimensions());
	Cuda3DArray<real_t>* proj=new Cuda3DArray<real_t>(g_ind_vec.size(), obj_proj->getDimensions());
	Cuda3DArray<complex_t>* tempArr=new Cuda3DArray<complex_t>(g_ind_vec.size(), obj_proj->getDimensions());
	CudaSmartPtr var_probe_upd= new Cuda2DArray<complex_t>(obj_proj->getDimensions().x, obj_proj->getDimensions().y);
	CudaSmartPtr tempArrR= new Cuda2DArray<real_t>(obj_proj->getDimensions().x, obj_proj->getDimensions().y);
	Cuda3DArray<real_t>* denum=new Cuda3DArray<real_t>(g_ind_vec.size(), obj_proj->getDimensions());
	Cuda3DArray<real_t>* num=new Cuda3DArray<real_t>(g_ind_vec.size(), obj_proj->getDimensions());

	CudaSmartPtr weights= new Cuda2DArray<real_t>(p_object->getX(), p_object->getY());

	if(rParams->variable_probe)
	{
		double relax_U=block_size*1.0/Npos;
		int relax_V = 1;
		for(int i=0; i<g_ind_vec.size(); i++)
		{
			probe_evol.push_back(m_probe_evolution[1][g_ind_vec[i]]);
			probe_evolNorm+=pow(m_probe_evolution[1][g_ind_vec[i]], 2);
		}

		double tempWeight=h_maxFloat(p_object->getDevicePtr<real_t>(), p_object->getX(), p_object->getY(), p_object->getAlignedY());
		h_multiply(p_object->getDevicePtr<real_t>(), 1.0/tempWeight, weights->getDevicePtr<real_t>(),
				weights->getX(), weights->getY(), weights->getAlignedY());

		if(obj_proj->getNum()==1)
		{
			get_projections_cpu(weights, weight_proj, g_ind_vec, oROI_vec1, oROI_vec2);
		}
		else
		{	// run on GPU
			get_projections(weights, weight_proj, g_ind_vec, oROI1, oROI2);
		}

		//get_SVD_update
		for(int i=0; i<probe_update->getNum(); i++)
		{
			h_subtract(probe_update->getAt(i).getDevicePtr(), probe_update_m->getDevicePtr<complex_t>(), resid->getAt(i).getDevicePtr(),
					probe_update->getDimensions().x, probe_update->getDimensions().y, probe_update->getPtr()->getAlignedY());

			h_multiply(weight_proj->getAt(i).getDevicePtr(), resid->getAt(i).getDevicePtr(), resid->getAt(i).getDevicePtr(),
					resid->getDimensions().x, resid->getDimensions().y, resid->getPtr()->getAlignedY());

			h_multiplyConju(m_extramodes->getDevicePtr<complex_t>(), resid->getAt(i).getDevicePtr(), tempArr->getAt(i).getDevicePtr(),
					resid->getDimensions().x, resid->getDimensions().y, resid->getPtr()->getAlignedY());
			h_realComplexReal(tempArr->getAt(i).getDevicePtr(), proj->getAt(i).getDevicePtr(), tempArr->getDimensions().x, tempArr->getDimensions().y,
					tempArr->getPtr()->getAlignedY());
			h_addFactor(proj->getAt(i).getDevicePtr(), proj->getAt(i).getDevicePtr(), probe_evol[i],
					proj->getDimensions().x, proj->getDimensions().y, proj->getPtr()->getAlignedY());
			h_normalize(proj->getAt(i).getDevicePtr(), probe_evolNorm, proj->getDimensions().x, proj->getDimensions().y, proj->getPtr()->getAlignedY());

			real_t mean2Proj=h_mean2(proj->getAt(i).getDevicePtr(), proj->getDimensions().x, proj->getDimensions().y, proj->getPtr()->getAlignedY());
			h_normalize(resid->getAt(i).getDevicePtr(), resid->getDimensions().x, resid->getDimensions().y, resid->getPtr()->getAlignedY(), mean2Proj);
		}

		h_realModalSum(resid->getPtr()->getDevicePtr<complex_t>(), var_probe_upd->getDevicePtr<complex_t>(), resid->getNum(), resid->getDimensions().x,
				resid->getDimensions().y, resid->getPtr()->getAlignedY());
		h_normalize(var_probe_upd->getDevicePtr<complex_t>(), var_probe_upd->getX(), var_probe_upd->getY(), var_probe_upd->getAlignedY(),
				1.0/(var_probe_upd->getX()*var_probe_upd->getY()));

		double norm2var_probe_upd=h_norm2Mat(var_probe_upd->getDevicePtr<complex_t>(), tempArrR->getDevicePtr<real_t>(),
				var_probe_upd->getX(), var_probe_upd->getY(), var_probe_upd->getAlignedY());
		double temp=relax_U/std::max(1.0, norm2var_probe_upd);
		h_normalize(var_probe_upd->getDevicePtr<complex_t>(), var_probe_upd->getX(), var_probe_upd->getY(), var_probe_upd->getAlignedY(), temp);
		h_complexSum(m_extramodes->getDevicePtr<complex_t>(), var_probe_upd->getDevicePtr<complex_t>(), m_extramodes->getDevicePtr<complex_t>(), 1.0, 1.0,
				var_probe_upd->getX(), var_probe_upd->getY(), var_probe_upd->getAlignedY());

		double norm2var_probe=h_norm2Mat(m_extramodes->getDevicePtr<complex_t>(), tempArrR->getDevicePtr<real_t>(),
				m_extramodes->getX(), m_extramodes->getY(), m_extramodes->getAlignedY());
		h_normalize(m_extramodes->getDevicePtr<complex_t>(), m_extramodes->getX(), m_extramodes->getY(), m_extramodes->getAlignedY(),
				1.0/norm2var_probe);

		double sum2denum=0;
		std::vector <double> denumvec;
		std::vector <double> numvec;
		for(int i=0; i<probe_update->getNum(); i++)
		{
			h_multiply(obj_proj->getAt(i).getDevicePtr(), m_extramodes->getDevicePtr<complex_t>(), tempArr->getAt(i).getDevicePtr(),
					tempArr->getDimensions().x, tempArr->getDimensions().y, tempArr->getPtr()->getAlignedY());
			h_realComplexAbs(tempArr->getAt(i).getDevicePtr(), denum->getAt(i).getDevicePtr(), tempArr->getDimensions().x, tempArr->getDimensions().y, tempArr->getPtr()->getAlignedY(), true);

			h_multiplyConju(chi->getAt(i).getDevicePtr(), tempArr->getAt(i).getDevicePtr(), tempArr->getAt(i).getDevicePtr(),
					chi->getDimensions().x, chi->getDimensions().y, chi->getPtr()->getAlignedY());
			h_realComplexReal(tempArr->getAt(i).getDevicePtr(), num->getAt(i).getDevicePtr(), num->getDimensions().x, num->getDimensions().y,
					num->getPtr()->getAlignedY());
			double mean2denum=h_mean2(denum->getAt(i).getDevicePtr(), denum->getDimensions().x, denum->getDimensions().y, denum->getPtr()->getAlignedY());
			denumvec.push_back(mean2denum);
			sum2denum+=mean2denum;
			double mean2num=h_mean2(num->getAt(i).getDevicePtr(), num->getDimensions().x, num->getDimensions().y, num->getPtr()->getAlignedY());
			numvec.push_back(mean2num);
		}
		double mean3denum=sum2denum/probe_update->getNum();

		for(int i=0; i<probe_update->getNum(); i++)
		{
			double temp=numvec[i]/(denumvec[i]+0.1*mean3denum);
			int index=g_ind_vec[i];
			m_probe_evolution[1][index]=m_probe_evolution[1][index]+relax_V*temp;
		}
	}
	if(rParams->variable_intensity)
	{
		double sum2nom=0;
		double sum2denom=0;
		for(int i=0; i<probe_update->getNum(); i++)
		{
			h_multiply(obj_proj->getAt(i).getDevicePtr(), m_modes->getAt(0).getDevicePtr(), tempArr->getAt(i).getDevicePtr(),
					tempArr->getDimensions().x, tempArr->getDimensions().y, tempArr->getPtr()->getAlignedY());
			h_realComplexAbs(tempArr->getAt(i).getDevicePtr(), denum->getAt(i).getDevicePtr(), tempArr->getDimensions().x,
					tempArr->getDimensions().y, tempArr->getPtr()->getAlignedY(), true);
			sum2denom=h_realSum(denum->getAt(i).getDevicePtr(), denum->getDimensions().x, denum->getDimensions().y, denum->getPtr()->getAlignedY());
			h_multiplyConju(chi->getAt(i).getDevicePtr(), tempArr->getAt(i).getDevicePtr(), tempArr->getAt(i).getDevicePtr(),
					chi->getDimensions().x, chi->getDimensions().y, chi->getPtr()->getAlignedY());
			h_realComplexReal(tempArr->getAt(i).getDevicePtr(), num->getAt(i).getDevicePtr(), num->getDimensions().x, num->getDimensions().y,
					num->getPtr()->getAlignedY());
			sum2nom=h_realSum(num->getAt(i).getDevicePtr(), num->getDimensions().x, num->getDimensions().y, num->getPtr()->getAlignedY());
			double temp=0.1*sum2nom/sum2denom;
			int index=g_ind_vec[i];
			m_probe_evolution[0][index]=m_probe_evolution[0][index]+temp;
		}

	}

	delete weight_proj;
	delete resid;
	delete proj;
	delete tempArr;
	delete denum;
	delete num;
}

void Probe::toRGBA(float4* out, const char* name, float tf, float ts)
{
	string probeName(name);
	int modeStrPos = probeName.find('P')+1;
	int modeIndex = atoi(probeName.substr(modeStrPos, probeName.length()-modeStrPos+1).c_str());
	real_t maxModeIntensity = (m_modes->checkUseAll()&&m_modes->getNum()>1)?h_realMax(m_intensities->getAt(modeIndex).getDevicePtr(), 
																m_intensities->getAt(modeIndex).getX(), m_intensities->getAt(modeIndex).getY(), 
																m_intensities->getAt(modeIndex).getAlignedY()) : m_maxIntensity;
	h_realToRGBA(m_intensities->getAt(modeIndex).getDevicePtr(), out, m_intensities->getAt(modeIndex).getX(), 
		m_intensities->getAt(modeIndex).getY(), m_intensities->getAt(modeIndex).getAlignedY(), 1.0/maxModeIntensity, tf, ts);

	//m_renderableUpdated = false;
}

void Probe::toGray(float* out, const char* name, bool outAligned)
{
	string probeName(name);
	int modeStrPos = probeName.find('P')+1;
	int modeIndex = atoi(probeName.substr(modeStrPos, probeName.length()-modeStrPos+1).c_str());
	real_t maxModeIntensity = (m_modes->checkUseAll()&&m_modes->getNum()>1)?h_realMax(m_intensities->getAt(modeIndex).getDevicePtr(), 
																m_intensities->getAt(modeIndex).getX(), m_intensities->getAt(modeIndex).getY(), 
																m_intensities->getAt(modeIndex).getAlignedY()) : m_maxIntensity;
	h_realToGray(m_intensities->getAt(modeIndex).getDevicePtr(), out, m_intensities->getAt(modeIndex).getX(),  m_intensities->getAt(modeIndex).getY(),
																m_intensities->getAt(modeIndex).getAlignedY(), 1.0/maxModeIntensity, outAligned);

	//m_renderableUpdated = false;
}

void Probe::updateProbeEstimate(const ICuda2DArray* object, const Cuda3DArray<complex_t>* psi, 
								const Cuda3DArray<complex_t>* psi_old, unsigned int qx, unsigned int qy,
								real_t objectMaxIntensity)
{
	h_updateProbe(m_modes->getPtr()->getDevicePtr<complex_t>(), object->getDevicePtr<complex_t>(), psi->getPtr()->getDevicePtr<complex_t>(),
					psi_old->getPtr()->getDevicePtr<complex_t>(), m_intensities->getPtr()->getDevicePtr<real_t>(), qx, qy, 1.0/objectMaxIntensity,
					m_modes->checkUseAll()?m_modes->getNum():1, psi->getDimensions().x, psi->getDimensions().y, psi->getPtr()->getAlignedY(),
					object->getX(), object->getY(), object->getAlignedY());
	updateMaxIntensity();
}

void Probe::updateMaxIntensity(bool useSum)
{
	m_maxIntensity = useSum?PhaserUtil::getInstance()->getModalDoubleSum(m_intensities) :
							PhaserUtil::getInstance()->getModalDoubleMax(m_intensities);
	m_renderableUpdated = true;
}

void Probe::beginModalReconstruction() 
{
	m_modes->setUseAll(true);
	m_intensities->setUseAll(true);
	if(!m_modesInitialized)
		initProbeModes();
}

void Probe::endModalReconstruction() 
{
	m_modes->setUseAll(false);
	m_intensities->setUseAll(false);
	m_modesInitialized = false;
}

unsigned int Probe::getWidth() const {return m_modes->getDimensions().y;}
unsigned int Probe::getHeight()const {return m_modes->getDimensions().x;}

void Probe::fillResources()
{
	for(unsigned int m=0; m<m_modes->getNum(); ++m)
	{
		char probeName[255];
		sprintf(probeName, "|P%d|", m);
		m_myResources.push_back(Resource(probeName, RAINBOW, this));
	}
}
