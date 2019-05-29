//*************************************************************************//
//                                                                         //
//  LibCudaOptimize                                                        //
//  Copyright (C) 2012 Ibislab, University of Parma                        //
//  Authors: Youssef S.G. Nashed, Roberto Ugolotti                         //
//                                                                         //
//  You should have received a copy of the GNU General Public License      //
//  along with this program.  If not, see <http://www.gnu.org/licenses/>   //
//                                                                         //
//*************************************************************************//

#include <stdint.h>

#ifndef _REDUCTIONS_CU_
#define _REDUCTIONS_CU_

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

/*! \brief inline kernel code for fast parallel reduction operations (max, min, sum).
 *
 * \brief Reduces an array of unsigned int elements to its minimum value
 * \param vet pointer to shared memory data to be reduced
 * \param tid thread index */
template <uint32_t  blockSize>
__device__ void reduceToMin(unsigned int* sdata, uint32_t  tid){

	//Synchronize threads to share shared memory data
	__syncthreads();

	unsigned int mySum = sdata[tid];

	// do reduction in shared mem
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = mySum = min(mySum, sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = min(mySum, sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = min(mySum, sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = min(mySum, sdata[tid +  64]); } __syncthreads(); }

	if (blockSize == 32){
		#ifndef __DEVICE_EMULATION__
		if (tid < 16)
			#endif
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile unsigned int* smem = sdata;
				if (blockSize >=  32) { smem[tid] = mySum = min(mySum, smem[tid + 16]); EMUSYNC; }
				if (blockSize >=  16) { smem[tid] = mySum = min(mySum, smem[tid +  8]); EMUSYNC; }
				if (blockSize >=   8) { smem[tid] = mySum = min(mySum, smem[tid +  4]); EMUSYNC; }
				if (blockSize >=   4) { smem[tid] = mySum = min(mySum, smem[tid +  2]); EMUSYNC; }
				if (blockSize >=   2) { smem[tid] = mySum = min(mySum, smem[tid +  1]); EMUSYNC; }
			}
	}
	else
	{
		#ifndef __DEVICE_EMULATION__
		if (tid < 32)
			#endif
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile unsigned int* smem = sdata;
				if (blockSize >=  64) { smem[tid] = mySum = min(mySum, smem[tid + 32]); EMUSYNC; }
				if (blockSize >=  32) { smem[tid] = mySum = min(mySum, smem[tid + 16]); EMUSYNC; }
				if (blockSize >=  16) { smem[tid] = mySum = min(mySum, smem[tid +  8]); EMUSYNC; }
				if (blockSize >=   8) { smem[tid] = mySum = min(mySum, smem[tid +  4]); EMUSYNC; }
				if (blockSize >=   4) { smem[tid] = mySum = min(mySum, smem[tid +  2]); EMUSYNC; }
				if (blockSize >=   2) { smem[tid] = mySum = min(mySum, smem[tid +  1]); EMUSYNC; }
			}
	}
}



/*! \brief Reduces an array of float elements to its minimum value
 *  \param vet pointer to shared memory data to be reduced
 *  \param tid thread index */
template <uint32_t  blockSize>
__device__ void reduceToMin(float* sdata, uint32_t  tid){

	//Synchronize threads to share shared memory data
	__syncthreads();

	float mySum = sdata[tid];

	// do reduction in shared mem
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = mySum = fminf(mySum, sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = fminf(mySum, sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = fminf(mySum, sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = fminf(mySum, sdata[tid +  64]); } __syncthreads(); }

	if (blockSize == 32){
		#ifndef __DEVICE_EMULATION__
		if (tid < 16)
			#endif
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile float* smem = sdata;
				if (blockSize >=  32) { smem[tid] = mySum = fminf(mySum, smem[tid + 16]); EMUSYNC; }
				if (blockSize >=  16) { smem[tid] = mySum = fminf(mySum, smem[tid +  8]); EMUSYNC; }
				if (blockSize >=   8) { smem[tid] = mySum = fminf(mySum, smem[tid +  4]); EMUSYNC; }
				if (blockSize >=   4) { smem[tid] = mySum = fminf(mySum, smem[tid +  2]); EMUSYNC; }
				if (blockSize >=   2) { smem[tid] = mySum = fminf(mySum, smem[tid +  1]); EMUSYNC; }
			}
	}
	else
	{
		#ifndef __DEVICE_EMULATION__
		if (tid < 32)
			#endif
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile float* smem = sdata;
				if (blockSize >=  64) { smem[tid] = mySum = fminf(mySum, smem[tid + 32]); EMUSYNC; }
				if (blockSize >=  32) { smem[tid] = mySum = fminf(mySum, smem[tid + 16]); EMUSYNC; }
				if (blockSize >=  16) { smem[tid] = mySum = fminf(mySum, smem[tid +  8]); EMUSYNC; }
				if (blockSize >=   8) { smem[tid] = mySum = fminf(mySum, smem[tid +  4]); EMUSYNC; }
				if (blockSize >=   4) { smem[tid] = mySum = fminf(mySum, smem[tid +  2]); EMUSYNC; }
				if (blockSize >=   2) { smem[tid] = mySum = fminf(mySum, smem[tid +  1]); EMUSYNC; }
			}
	}
}



/*! @brief Reduces an array of unsigned int elements to its maximum value
 *  \param vet pointer to shared memory data to be reduced
 *  \param tid thread index */
template <uint32_t  blockSize>
__device__ void reduceToMax(unsigned int* sdata, uint32_t  tid){

	//Synchronize threads to share shared memory data
	__syncthreads();

	unsigned int mySum = sdata[tid];

	// do reduction in shared mem
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = mySum = max(mySum, sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = max(mySum, sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = max(mySum, sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = max(mySum, sdata[tid +  64]); } __syncthreads(); }

	if (blockSize == 32){
		#ifndef __DEVICE_EMULATION__
		if (tid < 16)
			#endif
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile unsigned int* smem = sdata;
				if (blockSize >=  32) { smem[tid] = mySum = max(mySum, smem[tid + 16]); EMUSYNC; }
				if (blockSize >=  16) { smem[tid] = mySum = max(mySum, smem[tid +  8]); EMUSYNC; }
				if (blockSize >=   8) { smem[tid] = mySum = max(mySum, smem[tid +  4]); EMUSYNC; }
				if (blockSize >=   4) { smem[tid] = mySum = max(mySum, smem[tid +  2]); EMUSYNC; }
				if (blockSize >=   2) { smem[tid] = mySum = max(mySum, smem[tid +  1]); EMUSYNC; }
			}
	}
	else
	{
		#ifndef __DEVICE_EMULATION__
		if (tid < 32)
			#endif
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile unsigned int* smem = sdata;
				if (blockSize >=  64) { smem[tid] = mySum = max(mySum, smem[tid + 32]); EMUSYNC; }
				if (blockSize >=  32) { smem[tid] = mySum = max(mySum, smem[tid + 16]); EMUSYNC; }
				if (blockSize >=  16) { smem[tid] = mySum = max(mySum, smem[tid +  8]); EMUSYNC; }
				if (blockSize >=   8) { smem[tid] = mySum = max(mySum, smem[tid +  4]); EMUSYNC; }
				if (blockSize >=   4) { smem[tid] = mySum = max(mySum, smem[tid +  2]); EMUSYNC; }
				if (blockSize >=   2) { smem[tid] = mySum = max(mySum, smem[tid +  1]); EMUSYNC; }
			}
	}
}

/*! \brief Reduces the elements of an array to their maximum value
 *  \param sdata pointer to shared memory data to be reduced
 *  \param tid thread index */
template <uint32_t  blockSize>
__device__ void reduceToMax(float* sdata, uint32_t  tid){

	//Synchronize threads to share shared memory data
	__syncthreads();

	float mySum = sdata[tid];

	// do reduction in shared mem
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = fmaxf(mySum, sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = fmaxf(mySum, sdata[tid +  64]); } __syncthreads(); }

	if (blockSize == 32){
		#ifndef __DEVICE_EMULATION__
		if (tid < 16)
			#endif
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile float* smem = sdata;
				if (blockSize >=  32) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 16]); EMUSYNC; }
				if (blockSize >=  16) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  8]); EMUSYNC; }
				if (blockSize >=   8) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  4]); EMUSYNC; }
				if (blockSize >=   4) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  2]); EMUSYNC; }
				if (blockSize >=   2) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  1]); EMUSYNC; }
			}
	}
	else
	{
		#ifndef __DEVICE_EMULATION__
		if (tid < 32)
			#endif
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile float* smem = sdata;
				if (blockSize >=  64) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 32]); EMUSYNC; }
				if (blockSize >=  32) { smem[tid] = mySum = fmaxf(mySum, smem[tid + 16]); EMUSYNC; }
				if (blockSize >=  16) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  8]); EMUSYNC; }
				if (blockSize >=   8) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  4]); EMUSYNC; }
				if (blockSize >=   4) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  2]); EMUSYNC; }
				if (blockSize >=   2) { smem[tid] = mySum = fmaxf(mySum, smem[tid +  1]); EMUSYNC; }
			}
	}
}




/*! \brief Reduces the elements of an array to their sum
 *  \param sdata pointer to shared memory data to be reduced
 *  \param tid thread index */
template <class T, uint32_t  blockSize>
__device__ void reduceToSum(T* sdata, uint32_t  tid){

	//Synchronize threads to share shared memory data
	__syncthreads();

	T mySum = sdata[tid];

	// do reduction in shared mem
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = mySum = mySum + sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

	if (blockSize == 32){
		#ifndef __DEVICE_EMULATION__
		if (tid < 16)
		#endif
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile T* smem = sdata;
			if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
			if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
			if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
			if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
			if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
		}
	}
	else
	{
		#ifndef __DEVICE_EMULATION__
		if (tid < 32)
			#endif
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile T* smem = sdata;
			if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC; }
			if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
			if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
			if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
			if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
			if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
		}
	}
}

template <uint32_t  blockSize>
__device__ void reduceToSumComplex(complex_t* sdata, uint32_t  tid){

	//Synchronize threads to share shared memory data
	__syncthreads();

	complex_t mySum = sdata[tid];

	// do reduction in shared mem
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = mySum = add_complex_t(mySum, sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = add_complex_t(mySum, sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = add_complex_t(mySum, sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = add_complex_t(mySum, sdata[tid +  64]); } __syncthreads(); }

	if (blockSize == 32){
		#ifndef __DEVICE_EMULATION__
		if (tid < 16)
		#endif
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile complex_t* smem = sdata;
			if (blockSize >=  32) { mySum =add_complex_t(mySum, make_complex_t(smem[tid + 16].x, smem[tid + 16].y)); smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
			if (blockSize >=  16) { mySum =add_complex_t(mySum, make_complex_t(smem[tid +  8].x, smem[tid +  8].y)) ;smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
			if (blockSize >=   8) { mySum =add_complex_t(mySum, make_complex_t(smem[tid +  4].x, smem[tid +  4].y)) ;smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
			if (blockSize >=   4) { mySum =add_complex_t(mySum, make_complex_t(smem[tid +  2].x, smem[tid +  2].y)) ;smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
			if (blockSize >=   2) { mySum =add_complex_t(mySum, make_complex_t(smem[tid +  1].x, smem[tid +  1].y)) ;smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
		}
	}
	else
	{
		#ifndef __DEVICE_EMULATION__
		if (tid < 32)
			#endif
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile complex_t* smem = sdata;
			if (blockSize >=  64) { mySum =add_complex_t(mySum, make_complex_t(smem[tid + 32].x, smem[tid + 32].y)) ;smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
			if (blockSize >=  32) { mySum =add_complex_t(mySum, make_complex_t(smem[tid + 16].x, smem[tid + 16].y)) ;smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
			if (blockSize >=  16) { mySum =add_complex_t(mySum, make_complex_t(smem[tid +  8].x, smem[tid +  8].y)) ;smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
			if (blockSize >=   8) { mySum =add_complex_t(mySum, make_complex_t(smem[tid +  4].x, smem[tid +  4].y)) ;smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
			if (blockSize >=   4) { mySum =add_complex_t(mySum, make_complex_t(smem[tid +  2].x, smem[tid +  2].y)) ;smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
			if (blockSize >=   2) { mySum =add_complex_t(mySum, make_complex_t(smem[tid +  1].x, smem[tid +  1].y)) ;smem[tid].x = mySum.x; smem[tid].y = mySum.y; EMUSYNC; }
		}
	}
}

/*! \brief Reduces the elements of an array to their product
 *  \param sdata pointer to shared memory data to be reduced
 *  \param tid thread index */
template <class T, uint32_t  blockSize>
__device__ void reduceToProduct(T* sdata, uint32_t  tid){

	//Synchronize threads to share shared memory data
	__syncthreads();

	T mySum = sdata[tid];

	// do reduction in shared mem
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = mySum = mySum * sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum * sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum * sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum * sdata[tid +  64]; } __syncthreads(); }

	if (blockSize == 32){
		#ifndef __DEVICE_EMULATION__
		if (tid < 16)
		#endif
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile T* smem = sdata;
			if (blockSize >=  32) { smem[tid] = mySum = mySum * smem[tid + 16]; EMUSYNC; }
			if (blockSize >=  16) { smem[tid] = mySum = mySum * smem[tid +  8]; EMUSYNC; }
			if (blockSize >=   8) { smem[tid] = mySum = mySum * smem[tid +  4]; EMUSYNC; }
			if (blockSize >=   4) { smem[tid] = mySum = mySum * smem[tid +  2]; EMUSYNC; }
			if (blockSize >=   2) { smem[tid] = mySum = mySum * smem[tid +  1]; EMUSYNC; }
		}
	}
	else
	{
		#ifndef __DEVICE_EMULATION__
		if (tid < 32)
			#endif
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile T* smem = sdata;
			if (blockSize >=  64) { smem[tid] = mySum = mySum * smem[tid + 32]; EMUSYNC; }
			if (blockSize >=  32) { smem[tid] = mySum = mySum * smem[tid + 16]; EMUSYNC; }
			if (blockSize >=  16) { smem[tid] = mySum = mySum * smem[tid +  8]; EMUSYNC; }
			if (blockSize >=   8) { smem[tid] = mySum = mySum * smem[tid +  4]; EMUSYNC; }
			if (blockSize >=   4) { smem[tid] = mySum = mySum * smem[tid +  2]; EMUSYNC; }
			if (blockSize >=   2) { smem[tid] = mySum = mySum * smem[tid +  1]; EMUSYNC; }
		}
	}
}





#endif


