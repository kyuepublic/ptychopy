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
#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <execinfo.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "datatypes.h"
#include <thrust/device_vector.h>

#if CUDA_VERSION >= 5000
	//#include <cuda_runtime.h>
	#include <helper_functions.h>
	#include <driver_types.h>
	#include <vector_types.h>
	#define _LOAD_PGMF sdkLoadPGM<float>
	#define _LOAD_PGMU sdkLoadPGM<unsigned char>
	#define _SAVE_PGMF sdkSavePGM<float>
	#define _SAVE_PGMUB sdkSavePGM<unsigned char>
	#define _SAVE_PPMUB sdkSavePPM4ub

	#define _CHECK_CMDLINE checkCmdLineFlag
	#define cutGetCmdLineArgumenti(argc,argv,arg,intValuePtr) *intValuePtr=getCmdLineArgumentInt(argc,argv,arg)
	#define cutGetCmdLineArgumentstr getCmdLineArgumentString

	#define TIMER_TYPE StopWatchInterface*
	#define _CREATE_TIMER(x) sdkCreateTimer(&x);
	#define _RESET_TIMER(x) sdkResetTimer(&x);
	#define _START_TIMER(x) sdkStartTimer(&x);
	#define _STOP_TIMER(x) sdkStopTimer(&x);
	#define _GET_TIMER_VALUE(x) sdkGetTimerValue(&x);

	#define cutilCheckMsg(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
	inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
	{
		cudaError_t err = cudaGetLastError();

		if (cudaSuccess != err)
		{
			void *array[10];
			size_t size;

			// get void*'s for all entries on the stack
			size = backtrace(array, 10);

			// print out all the frames to stderr
			fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
								file, line, errorMessage, (int)err, cudaGetErrorString(err));
			backtrace_symbols_fd(array, size, STDERR_FILENO);
			exit(-1);
		}
	}

#else
	#include <cutil.h>
	#include <cutil_inline.h>
	//#include <cutil_gl_inline.h>
	#include <stdint.h>
	#define _LOAD_PGMF cutLoadPGMf
	#define _LOAD_PGMU cutLoadPGMub
	#define _SAVE_PGMF cutSavePGMf
	#define _SAVE_PGMUB cutSavePGMub
	#define _SAVE_PPMUB cutSavePPMub
	
	#define _CHECK_CMDLINE cutCheckCmdLineFlag

	#define TIMER_TYPE unsigned int
	#define _CREATE_TIMER(x) cutCreateTimer(&x);
	#define _RESET_TIMER(x) cutResetTimer(x);
	#define _START_TIMER(x) cutStartTimer(x);
	#define _STOP_TIMER(x) cutStopTimer(x);
	#define _GET_TIMER_VALUE(x) cutGetTimerValue(x);
#endif

///////////////////////////////////////////////////////////////////////////////
// Common host and device function
///////////////////////////////////////////////////////////////////////////////
//ceil(a / b)
inline int gh_iDivUp(int a, int b)
{return ((a % b) != 0) ? (a / b + 1) : (a / b);}

//floor(a / b)
inline int gh_iDivDown(int a, int b)
{return a / b;}

//Align a to nearest higher multiple of b
inline int gh_iAlignUp(int a, int b)
{return ((a % b) != 0) ?  (a - a % b + b) : a;}

//Align a to nearest lower multiple of b
inline int gh_iAlignDown(int a, int b)
{return a - a % b;}

inline unsigned int alignFFTSize(unsigned int dataSize)
{
    int hiBit;
    unsigned int lowPOT, hiPOT;

    dataSize = gh_iAlignUp(dataSize, 16);

    for (hiBit = 31; hiBit >= 0; hiBit--)
        if (dataSize & (1U << hiBit))
            break;

    lowPOT = 1U << hiBit;

    if (lowPOT == (unsigned int)dataSize)
        return dataSize;

    hiPOT = 1U << (hiBit + 1);

    if (hiPOT <= 1024)
        return hiPOT;
    else
        return gh_iAlignUp(dataSize, 512);
}

inline unsigned int factorRadix2(unsigned int &log2N, unsigned int n)
{
    if (!n)
    {
        log2N = 0;
        return 0;
    }
    else
    {
        for (log2N = 0; n % 2 == 0; n /= 2, log2N++);
        return n;
    }
}




typedef struct { double x, y; int group; } point_t, *point;

inline double randf(double m)
{
	return m * rand() / (RAND_MAX - 1.);
}

inline point gen_xy(int count, double radius)
{
	double ang, r;
	point p, pt = (point)malloc(sizeof(point_t) * count);

	/* note: this is not a uniform 2-d distribution */
	for (p = pt + count; p-- > pt;) {
		ang = randf(2 * M_PI);
		r = randf(radius);
		p->x = r * cos(ang);
		p->y = r * sin(ang);
	}

	return pt;
}

inline double dist2(point a, point b)
{
	double x = a->x - b->x, y = a->y - b->y;
	return x*x + y*y;
}

inline int
nearest(point pt, point cent, int n_cluster, double *d2)
{
	int i, min_i;
	point c;
	double d, min_d;

#	define for_n for (c = cent, i = 0; i < n_cluster; i++, c++)
	for_n {
		min_d = HUGE_VAL;
		min_i = pt->group;
		for_n {
			if (min_d > (d = dist2(c, pt))) {
				min_d = d; min_i = i;
			}
		}
	}
	if (d2) *d2 = min_d;
	return min_i;
}

inline void kpp(point pts, int len, point cent, int n_cent)
{
#	define for_len for (j = 0, p = pts; j < len; j++, p++)
//	int i;
	int j;
	int n_cluster;
	double sum, *d = (double*)malloc(sizeof(double) * len);

	point p;
//	point c;
	cent[0] = pts[ rand() % len ];
	for (n_cluster = 1; n_cluster < n_cent; n_cluster++) {
		sum = 0;
		for_len {
			nearest(p, cent, n_cluster, d + j);
			sum += d[j];
		}
		sum = randf(sum);
		for_len {
			if ((sum -= d[j]) > 0) continue;
			cent[n_cluster] = pts[j];
			break;
		}
	}
	for_len p->group = nearest(p, cent, n_cluster, 0);
	free(d);
}

inline point lloyd(point pts, int len, int n_cluster)
{
	int i, j, min_i;
	int changed;

	point cent = (point)malloc(sizeof(point_t) * n_cluster), p, c;

	/* assign init grouping randomly */
	//for_len p->group = j % n_cluster;

	/* or call k++ init */
	kpp(pts, len, cent, n_cluster);

	do {
		/* group element for centroids are used as counters */
		for_n { c->group = 0; c->x = c->y = 0; }
		for_len {
			c = cent + p->group;
			c->group++;
			c->x += p->x; c->y += p->y;
		}
		for_n { c->x /= c->group; c->y /= c->group; }

		changed = 0;
		/* find closest centroid of each point */
		for_len {
			min_i = nearest(p, cent, n_cluster, 0);
			if (min_i != p->group) {
				changed++;
				p->group = min_i;
			}
		}
	} while (changed > (len >> 10)); /* stop when 99.9% of points are good */

	for_n { c->group = i; }

	return cent;
}

inline double rand_gen() {
   // return a uniformly distributed random value
   return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}
inline double normalRandom() {
   // return a normally distributed random value
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}

// [lower upper]
inline int rand_gen_no(int lower, int upper) {
	// return a uniformly distributed random value
    return rand() % (++upper - lower) + lower;
}

__host__ int getReductionThreadNum(int size);
__host__ void h_initColorTransferTexture();
__host__ void h_freeColorTransferTexture();
__host__ void h_simulateGaussian(real_t* d_gauss, real_t window,
								unsigned int x, unsigned int y, unsigned int alignedY);

__host__ float2 h_maxFloat2(float2* a, unsigned int x, unsigned int y, unsigned int alignedY);
__host__ float2 h_minFloat2(float2* a, unsigned int x, unsigned int y, unsigned int alignedY);
__host__ real_t h_maxFloat(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY);

__host__ real_t h_norm2Mat(real_t* d_arr, real_t* d_result, unsigned int x, unsigned int y, unsigned int alignedY);
__host__ real_t h_norm2Mat(complex_t* d_arr, real_t* d_result, unsigned int x, unsigned int y, unsigned int alignedY);


__host__ void h_subtract(const complex_t* a, const complex_t* b, complex_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY);
__host__ void h_subtract(const real_t* a, const real_t* b, real_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY);
__host__ void h_square(real_t* a, real_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_addFactorDivide(real_t* a, real_t* result, real_t factor,
								unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_object_sum_update_Gfun(complex_t* a, real_t* b, complex_t* result, real_t factor,
								unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_addFactor(complex_t* a, complex_t* result, complex_t factor,
								unsigned int x, unsigned int y, unsigned int alignedY);
__host__ void h_addFactor(real_t* a, real_t* result, real_t factor,
								unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_multiply(complex_t* a, complex_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
								bool normalize = false, unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0);
__host__ void h_multiply(real_t* a, complex_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
								bool normalize = false, unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0);
__host__ void h_multiply(real_t* a, real_t* b, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
								bool normalize = false, unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0);
__host__ void h_multiply(const complex_t* a, const complex_t& b, complex_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY, bool normalize = false);
__host__ void h_multiply(const real_t* a, const real_t& b, real_t* result,
								unsigned int x, unsigned int y, unsigned int alignedY, bool normalize = false);

__host__ void h_multiplyPage(complex_t* a, complex_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
		unsigned int pagex, unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0);

__host__ void h_multiplyReal(real_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_multiplyRow(complex_t* a, real_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
								bool normalize = false, unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0);
__host__ void h_multiplyColumn(complex_t* a, real_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
								bool normalize = false, unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0);

__host__ void h_multiplyConju(complex_t* a, complex_t* b, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY,
								bool normalize = false, unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0);



__host__ void h_realToRGBA(const real_t* d_arr, float4* d_output, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor, float tf, float ts);
__host__ void h_realToGray(const real_t* d_arr, float* d_output, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor, bool outAligned=false);
__host__ void h_normalize(complex_t* a, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor);
__host__ void h_normalize(const complex_t* a, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor);
__host__ void h_normalize(real_t* d_arr, unsigned int x, unsigned int y, unsigned int alignedY);
__host__ void h_normalize(real_t* a, real_t factor, unsigned int x, unsigned int y, unsigned int alignedY);

__host__ real_t h_realMax(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_squareRoot(real_t* d_arr, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_realComplexAbs(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared=false);
__host__ void h_realComplexPhase(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared=false);
__host__ void h_realComplexReal(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared=false);
__host__ void h_realComplexImag(const complex_t* a, real_t* result, unsigned int x, unsigned int y, unsigned int alignedY, bool squared=false);
__host__ void h_realComplexExp(const real_t* src, complex_t* result, unsigned int x, unsigned int y, unsigned int alignedY, real_t factor=1.0);


__host__ void h_adjustFFT(const complex_t* d_psi, complex_t* d_output, const real_t* d_det_mod, const real_t* d_mask, const real_t saturationValue,
								unsigned int modeNum, unsigned int x, unsigned int y, unsigned int alignedY, bool normalize=true);
__host__ real_t h_calculateER(const complex_t* d_psi, const real_t* d_det_mod, unsigned int modeNum,
								unsigned int x, unsigned int y, unsigned int alignedY);
__host__ real_t h_calculateER(const complex_t* d_GT, const complex_t* d_obj,
								unsigned int sx, unsigned int sy, unsigned int qx, unsigned int qy,
								unsigned int x1, unsigned int y1, unsigned int alignedY1,
								unsigned int x2, unsigned int y2, unsigned int alignedY2);
__host__ void h_shiftFFT(real_t* d_data, real_t* d_temp, unsigned int x, unsigned int y, unsigned int alignedY, cudaStream_t* stream=0);

__host__ void h_shiftFFTy(real_t* d_data, real_t* d_temp, unsigned int x, unsigned int y, unsigned int alignedY, cudaStream_t* stream=0);

__host__ void h_shiftFFTtmp(complex_t* d_probe, complex_t* d_tempprobe, complex_t* d_copyprobe, unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_shiftFFTtwo(complex_t* d_probe, complex_t* d_tempprobe, unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void imshift_fft(complex_t* d_probe, unsigned int x, unsigned int y, unsigned int alignedY, float radNo1, float radNo2);

__host__ void h_extracSubArrReal(real_t* d_objectArray, real_t* d_output, unsigned int offsetX, unsigned int offsetY,
		unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
		unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY);

__host__ void h_extracSubArrComplex(complex_t* d_objectArray, complex_t* d_output, unsigned int offsetX, unsigned int offsetY,
		unsigned int sampleX, unsigned int sampleY, unsigned int alignedSampleY,
		unsigned int objectArrayX, unsigned int objectArrayY, unsigned int alignedObjectArrayY);


__host__ void h_realRotate90(const real_t* d_data, real_t* d_out, unsigned int x, unsigned int y, unsigned int alignedY, unsigned int times, cudaStream_t* stream=0);
__host__ complex_t h_innerProduct(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
									unsigned int x, unsigned int y, unsigned int alignedY);

__host__ complex_t h_innerProductOne(const complex_t* d_u, const complex_t* d_v, complex_t* d_output,
									unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_innerProductModes( complex_t* d_u,  complex_t* d_v, complex_t* d_factor, unsigned int index,
		unsigned int modesNum, unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_realModalSum(const real_t* d_modes, real_t* d_output, unsigned int modesNum, unsigned int x, unsigned int y, 
								unsigned int alignedY, bool sqaureRoot=false);
__host__ void h_realModalSum(const complex_t* d_modes, complex_t* d_output, unsigned int modesNum, unsigned int x, unsigned int y,
								unsigned int alignedY, bool sqaureRoot=false);

__host__ void h_complexSum(complex_t* d_leftArr, complex_t* d_rightArr, complex_t* d_result, real_t leftFactor, real_t rightFactor, unsigned int x, unsigned int y,
								unsigned int alignedY);
__host__ complex_t h_complexSum(const complex_t* a, unsigned int x1, unsigned int x2, unsigned int y1, unsigned int y2, unsigned int alignedY);

__host__ real_t h_realSum(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY);
__host__ real_t h_realSum(const real_t* a, unsigned int x1, unsigned int x2, unsigned int y1, unsigned int y2, unsigned int alignedY);
__host__ void h_realSum(real_t* d_leftArr, real_t* d_rightArr, real_t* d_result, real_t leftFactor, real_t rightFactor, unsigned int x, unsigned int y,
								unsigned int alignedY);

//__host__ real_t h_realSumCUB(real_t* d_in, unsigned int x, unsigned int y, unsigned int alignedY);

// Get the mean2 of each 2d array
__host__ real_t h_mean2(real_t* a, unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_realSingleSum(real_t* d_leftArr, real_t* d_rightArr, real_t* d_result, unsigned int x, unsigned int y,
								unsigned int alignedY);

__host__ int2 h_realArgMax2D(real_t* d_ncc, unsigned int x, unsigned int y, unsigned int alignedY, unsigned char dir);
__host__ void h_realComplexModulate(const complex_t* d_array1, complex_t* d_array2, int2& peak,
								unsigned int x, unsigned int y, unsigned int alignedY, unsigned char dir);

__host__ void h_set_projections(real_t* p_object, real_t* proj, unsigned int* p_positions_x, unsigned int* p_positions_y,
		unsigned int objectX, unsigned int objectY, unsigned int alignedObjectY, unsigned int probeX, unsigned int probeY, unsigned int Npos);

__host__ void h_set_projections(complex_t* p_object, complex_t* proj, unsigned int* p_positions_x, unsigned int* p_positions_y,
		unsigned int objectX, unsigned int objectY, unsigned int alignedObjectY, unsigned int probeX, unsigned int probeY, unsigned int alignedProbeY,
		unsigned int Npos, bool isFlat);

__host__ void h_get_projections(const complex_t* p_object, complex_t* proj, unsigned int* p_positions_x, unsigned int* p_positions_y,
		unsigned int objectX, unsigned int objectY, unsigned int alignedObjectY, unsigned int probeX, unsigned int probeY, unsigned int probeZ,
		unsigned int alignedProbeY, unsigned int Npos);

__host__ void h_get_projections(const real_t* p_object, real_t* proj, unsigned int* p_positions_x, unsigned int* p_positions_y,
		unsigned int objectX, unsigned int objectY, unsigned int alignedObjectY, unsigned int probeX, unsigned int probeY, unsigned int probeZ,
		unsigned int alignedProbeY, unsigned int Npos);

__host__ void h_get_optimal_step_lsq(complex_t* chi,complex_t* object_update_proj, complex_t* dPO, complex_t* probe, real_t lambda,
		real_t* AA1, complex_t* AA2, real_t* AA4, real_t* Atb1, real_t* Atb2, unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_mul_rca_mulc_rcr(complex_t* obj_proj_i, complex_t* modes_i, complex_t* chi_i, real_t* weight_proj,
		unsigned int x, unsigned int y, unsigned int alignedY);

__host__ void h_multiplyAbsConjuRealWhole(complex_t* a, complex_t* b, complex_t* c, real_t* result1, real_t* result2, unsigned int x, unsigned int y, unsigned int alignedY,
		unsigned int pagex);

__host__ void h_multiplyAbsConjuReal(complex_t* a, complex_t* b, complex_t* c, real_t* result1, real_t* result2, unsigned int x, unsigned int y, unsigned int alignedY,
		unsigned int pagex);

__host__ void h_checkCache(	thrust::device_vector<real_t>& m_factors,
thrust::host_vector<bool>& m_cachedFlags,
thrust::host_vector<real_t>& m_cachedFactors, thrust::device_vector<bool>& m_flags, real_t objMax, real_t probeMax,
		bool phaseConstraint,bool updateProbe, bool updateProbeModes, bool RMS);

template<typename T> class Cuda3DArray;
template<typename T> class Cuda3DElement;
template<typename T> class Cuda2DArray;
#include "CudaSmartPtr.h"
#include "Singleton.h"

class GPUProperties
{
private:
	int m_gpuCount;
	unsigned int m_gpuWarpSize;
	unsigned int m_gpuMaxThread;
	size_t m_gpuAvailableMemory;

public:
	GPUProperties();
	~GPUProperties(){}

	int getDeviceCount()
	{
		return m_gpuCount;
	}
	unsigned int getGPUMaxThreads() const
	{
		return m_gpuMaxThread;
	}
	unsigned int alignToWarp(unsigned int dimension)
	{
		return gh_iAlignUp(dimension, m_gpuWarpSize);
	}
	size_t getGPUAvailableMemory()  const;
};

typedef Singleton<GPUProperties> GPUQuery;

inline bool calcGrids(unsigned int x, unsigned int alignedY, dim3& grid, dim3& block)
{
	unsigned int maxThreads = GPUQuery::getInstance()->getGPUMaxThreads();
	if (alignedY <= maxThreads)
	{
		unsigned int sliceNum = gh_iDivDown(maxThreads, alignedY);
		grid.x=gh_iDivUp(x, sliceNum); grid.y=1; grid.z=1;
		block.x=alignedY; block.y=sliceNum; block.z=1;
		return true;
	}
	else
	{
		unsigned int sliceNum = gh_iDivUp(alignedY, maxThreads);
		grid.x=x; grid.y=sliceNum; grid.z=1;
		block.x=maxThreads; block.y=1; block.z=1;
		return false;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CXUtil
{
private:
	CudaSmartPtr m_workingMemory;

public:
	CXUtil(){}
	~CXUtil(){}

	int4 applyProjectionApprox(const CudaSmartPtr& object, const Cuda3DArray<complex_t>* probeModes,
								Cuda3DArray<complex_t>* psi, float2 scanPos);
	int4 applyProjectionApprox(const CudaSmartPtr& object, Cuda3DElement<complex_t> probeMode,
								CudaSmartPtr psi, float2 scanPos);
	void gaussSmooth(real_t* d_arr, unsigned int smoothWindow, unsigned int x, unsigned int y);
	void applyModulusConstraint(const Cuda3DArray<complex_t>* d_psi, Cuda3DArray<complex_t>* d_output,
										Cuda3DElement<real_t> d_det_mod, const real_t* d_beamstopMask=0);

	void applyFFT(const Cuda3DArray<complex_t>* d_psi, Cuda3DArray<complex_t>* d_output,
						Cuda3DElement<real_t> d_det_mod, const real_t* d_beamstopMask=0);

	void ff2Mat(Cuda3DArray<complex_t>* d_psi, Cuda3DArray<complex_t>* d_output,
			Cuda3DElement<real_t> d_det_mod, const real_t* d_beamstopMask=0);

	void iff2Mat(Cuda3DArray<complex_t>* d_psi, Cuda3DArray<complex_t>* d_output,
			Cuda3DElement<real_t> d_det_mod, const real_t* d_beamstopMask=0);

	real_t getModalDoubleMax(const Cuda3DArray<real_t>* d_arr);
	real_t getModalDoubleSum(const Cuda3DArray<real_t>* d_arr);
	real_t calculateER(Cuda3DArray<complex_t>* d_psi, Cuda3DElement<real_t> d_det_mod);
	real_t calculateER(const CudaSmartPtr& GT, const CudaSmartPtr& object,
						unsigned int qx, unsigned int qy, unsigned int x, unsigned int y);

	template<typename T> int sgn(T val);

	template<typename T> void median(std::vector<T>& vec, T &x);

	template<typename T> void mean(std::vector<T>& vec, T &x);

	template<typename T, typename A> bool load(char* filename, std::vector<T, A>& vec, bool binary=false);
};

template <typename T>
int CXUtil::sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template<typename T>
void CXUtil::median(std::vector<T>& vec, T &x)
{

	size_t size = vec.size();
	sort(vec.begin(), vec.end());
	if (size % 2 == 0)
	{
		x=1.0*(vec[size / 2 - 1] + vec[size / 2]) / 2;
	}
	else
	{
		x=vec[size / 2];
	}

}

template<typename T>
void CXUtil::mean(std::vector<T>& vec, T &meanNo)
{

	double sum = 0.0;
	size_t size=vec.size();
	for(int i=0; i<size; i++)
	   sum += vec[i];

	meanNo=sum/size;

}

template<typename T, typename A>
bool CXUtil::load(char* filename, std::vector<T, A>& vec, bool binary)
{
	vec.clear();

	std::ifstream infile(filename, std::ofstream::in|std::ofstream::binary );
	std::string line = "";
	int rowIdx=0;
	T val=0;
	while (std::getline(infile, line))
	{
        // Create a stringstream of the current line
        std::stringstream ss(line);
        int colIdx = 0;
        while(ss >> val)
        {
        	vec.push_back(val);
            if(ss.peek() == ',') ss.ignore();
            colIdx++;
        }
        rowIdx++;
	}
	infile.close();

	return true;
}

typedef Singleton<CXUtil> PhaserUtil;

#endif /* UTILITIES_H_ */
