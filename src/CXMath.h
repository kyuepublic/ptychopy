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
//Any publication using the package should cite for
//Yue K, Deng J, Jiang Y, Nashed Y, Vine D, Vogt S.
//Ptychopy: GPU framework for ptychographic data analysis.
//X-Ray Nanoimaging: Instruments and Methods V 2021.
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

#ifndef CXMATH_H_
#define CXMATH_H_

#include "utilities.h"
#include "Cuda2DArray.hpp"

class CXMath
{
public:
	template<typename T>
	static void multiply(const Cuda3DArray<T>*a, const Cuda3DArray<T>*b, Cuda3DArray<T>*c, bool firstAElementOnly=false,
						unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0)
	{
		if(a->checkUseAll())
			for(int m=a->getNum()-1; m>=0; --m)
				multiply(a->getAt(firstAElementOnly?0:m), b->getAt(m), c->getAt(m), axOffset, ayOffset, bxOffset, byOffset);
		else
			multiply(a->getAt(0), b->getAt(0), c->getAt(0), axOffset, ayOffset, bxOffset, byOffset);
	}

//	template<typename T, typename P>
//	static void multiply(const Cuda3DArray<T>*a, const Cuda3DArray<P>*b, Cuda3DArray<T>*c, bool firstAElementOnly=false,
//						unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0)
//	{
//		if(a->checkUseAll())
//			for(int m=a->getNum()-1; m>=0; --m)
//				multiply(a->getAt(firstAElementOnly?0:m), b->getAt(m), c->getAt(m), axOffset, ayOffset, bxOffset, byOffset);
//		else
//			multiply(a->getAt(0), b->getAt(0), c->getAt(0), axOffset, ayOffset, bxOffset, byOffset);
//	}
//
//	template<typename T, typename P>
//	static void multiply(Cuda3DElement<T> a, Cuda3DElement<P> b, Cuda3DElement<T> c,
//						unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0)
//	{
//		h_multiply( a.getDevicePtr(), b.getDevicePtr(), c.getDevicePtr(),
//					a.getX(), a.getY(), a.getAlignedY(), false, axOffset, ayOffset, bxOffset, byOffset);
//	}
	
	template<typename T>
	static void multiply(Cuda3DElement<T> a, Cuda3DElement<T> b, Cuda3DElement<T> c,
						unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0)
	{
		h_multiply( a.getDevicePtr(), b.getDevicePtr(), c.getDevicePtr(),
					a.getX(), a.getY(), a.getAlignedY(), false, axOffset, ayOffset, bxOffset, byOffset);
	}


	template<typename T>
	static void multiply(const ICuda2DArray*a, const ICuda2DArray*b, ICuda2DArray*c,
						unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0)
	{
		h_multiply( a->getDevicePtr<T>(), b->getDevicePtr<T>(), c->getDevicePtr<T>(),
					a->getX(), a->getY(), a->getAlignedY(), false, axOffset, ayOffset, bxOffset, byOffset);
	}

	template<typename T>
	static void multiply(const ICuda2DArray*a, const Cuda3DElement<T> b, ICuda2DArray*c,
						unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0)
	{
		h_multiply( a->getDevicePtr<T>(), b.getDevicePtr(), c->getDevicePtr<T>(),
					a->getX(), a->getY(), a->getAlignedY(), false, axOffset, ayOffset, bxOffset, byOffset);
	}


	template<typename T>
	static void multiply(const ICuda2DArray*a, const Cuda3DArray<T>*b, Cuda3DArray<T>*c,
						unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0)
	{
		if(b->checkUseAll())
			for(unsigned int m=0; m<b->getNum(); ++m)
				h_multiply( a->getDevicePtr<T>(), b->getAt(m).getDevicePtr(), c->getAt(m).getDevicePtr(),
							a->getX(), a->getY(), a->getAlignedY(), false, axOffset, ayOffset, bxOffset, byOffset);
		else
			h_multiply(a->getDevicePtr<T>(), b->getAt(0).getDevicePtr(), c->getAt(0).getDevicePtr(),
							a->getX(), a->getY(), a->getAlignedY(), false, axOffset, ayOffset, bxOffset, byOffset);
	}

	template<typename T>
	static void multiply(Cuda3DElement<T> a, const Cuda3DArray<T>*b, Cuda3DArray<T>*c,
						unsigned int axOffset=0, unsigned int ayOffset=0, unsigned int bxOffset=0, unsigned int byOffset=0)
	{
		if(b->checkUseAll())
			for(unsigned int m=0; m<b->getNum(); ++m)
				h_multiply( a.getDevicePtr(), b->getAt(m).getDevicePtr(), c->getAt(m).getDevicePtr(),
							a.getX(), a.getY(), a.getAlignedY(), false, axOffset, ayOffset, bxOffset, byOffset);
		else
			h_multiply(a.getDevicePtr(), b->getAt(0).getDevicePtr(), c->getAt(0).getDevicePtr(),
							a.getX(), a.getY(), a.getAlignedY(), false, axOffset, ayOffset, bxOffset, byOffset);
	}
};

#endif /* CXMATH_H_ */
