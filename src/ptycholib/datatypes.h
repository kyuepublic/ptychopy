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
//Any publication using the package should cite fo
//Yue K, Deng J, Jiang Y, Nashed Y, Vine D, Vogt S.
//Ptychopy: GPU framework for ptychographic data analysis.
//X-Ray Nanoimaging: Instruments and Methods V 2021 .
//International Society for Optics and Photonics.
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


#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <cuComplex.h>

#define USE_SINGLE_PRECISION //Comment out this line to use double point floating precision

#ifdef USE_SINGLE_PRECISION
	typedef float real_t;
	#define round_real_t roundf
	#define sqrt_real_t sqrtf
	#define rsqrt_real_t rsqrtf
	#define sin_real_t sinf
	#define cos_real_t cosf
	#define abs_real_t fabsf
	#define sincos_real_t sincosf
	#define sinpi_real_t sinpif
	#define cospi_real_t cospif
#if CUDA_VERSION < 5000
	#define sincospi_real_t(v,s,c) sincosf(v*CUDART_PI,s,c)
#else
	#define sincospi_real_t(v,s,c) sincospif(v,s,c)
#endif
	#define atan2_real_t atan2f
	#define uint2real_t __uint2float_rn

	typedef cuComplex complex_t;
	#define make_complex_t make_cuComplex
	#define real_complex_t cuCrealf
	#define imag_complex_t cuCimagf
	#define conj_complex_t cuConjf
	#define add_complex_t cuCaddf
	#define sub_complex_t cuCsubf
	#define mul_complex_t cuCmulf
	#define div_complex_t cuCdivf
	#define abs_complex_t cuCabsf
#else
	typedef double real_t;
	#define round_real_t round
	#define sqrt_real_t sqrt
	#define rsqrt_real_t rsqrt
	#define sin_real_t sin
	#define cos_real_t cos
	#define abs_real_t fabs
	#define sincos_real_t sincos
	#define sinpi_real_t sinpi
	#define cospi_real_t cospi
#if CUDA_VERSION < 5000
	#define sincospi_real_t(v,s,c) sincos(v*CUDART_PI,s,c)
#else
	#define sincospi_real_t(v,s,c) sincospi(v,s,c)
#endif
	#define atan2_real_t atan2
	#define uint2real_t __uint2double_rn

	typedef cuDoubleComplex complex_t;
	#define make_complex_t make_cuDoubleComplex
	#define real_complex_t cuCreal
	#define imag_complex_t cuCimag
	#define conj_complex_t cuConj
	#define add_complex_t cuCadd
	#define sub_complex_t cuCsub
	#define mul_complex_t cuCmul
	#define div_complex_t cuCdiv
	#define abs_complex_t cuCabs
#endif


#endif /* DATATYPES_H_ */
