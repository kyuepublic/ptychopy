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
#ifndef SAMPLE_H_
#define SAMPLE_H_

#include "CudaSmartPtr.h"
#include "IRenderable.h"
#include "datatypes.h"
#include <vector_types.h>

template<typename T> class Cuda3DArray;
template<typename T> class Cuda3DElement;
struct curandGenerator_st;

class Sample : public IRenderable
{

public:
	//beta_objectvec=cache.beta_object
	std::vector <complex_t> beta_objectvec;


    curandGenerator_st* m_randomGenerator;

private:
	uint2 m_objectArrayShape;
	CudaSmartPtr m_objectArray;
	CudaSmartPtr m_objectIntensities;
	CudaSmartPtr m_objectPhases;
	real_t m_maxObjectIntensity;

	CudaSmartPtr m_randStates;

	const CudaSmartPtr& updateRenderable(const char*);

public:
	Sample();
	Sample(unsigned int arrayShapeX, unsigned int arrayShapeY);
	~Sample();

	void initObject();
	CudaSmartPtr generatepureRandKernel(unsigned int x, unsigned int y);

	bool loadFromFile(const char*, const char* p=0, real_t normalize=1.0);
	void loadGuess(const char*);
	void setObjectArrayShape(uint2 xy);
	void setObjectArray(CudaSmartPtr c) {m_objectArray=c;}
	real_t getMaxIntensity()			 const {return m_maxObjectIntensity;}
	uint2 getObjectArrayShape() 		 const {return m_objectArrayShape;}
	unsigned int getObjectArrayShapeX()  const {return m_objectArrayShape.x;}
	unsigned int getObjectArrayShapeY()  const {return m_objectArrayShape.y;}
	CudaSmartPtr getObjectArray() 		 const {return m_objectArray;}
	CudaSmartPtr getIntensities() 		 const {return m_objectIntensities;}
	void clearObjectArray();
	void extractROI(CudaSmartPtr roi, float qx, float qy) const;
	void extractROI(Cuda3DElement<complex_t> roi, float qx, float qy) const;
	void updateObjectEstimate(	const Cuda3DArray<complex_t>* probeModes, const Cuda3DArray<complex_t>* psi,
								const Cuda3DArray<complex_t>* psi_old, unsigned int qx, unsigned int qy,
								real_t probeMaxIntensity, bool phaseConstraint=false);

	void update_object(CudaSmartPtr object_upd_sum, int llo, std::vector<int> g_ind_vec, std::vector<int> scan_idsvec,
			CudaSmartPtr illum_sum_0t, real_t MAX_ILLUM);

	void updateIntensities(bool useSum=false);
	void updateMaxIntensity(bool useSum=false);
	void addNeighborSubsamples(const Cuda3DArray<complex_t>*, CudaSmartPtr, CudaSmartPtr, uint2);
	const CudaSmartPtr& stitchSubsamples(const ICuda2DArray* s1, const ICuda2DArray* s2, unsigned char dir, bool simulated=false);

	//For rendering purposes
	void fillResources();
	unsigned int getWidth() const {return getObjectArrayShapeY();}
	unsigned int getHeight()const {return getObjectArrayShapeX();}
	void toRGBA(float4*,const char*,float,float);
	void toGray(float*,const char*,bool=false);

	void printObject(int column, int row);
};

#endif /* SAMPLE_H_ */
