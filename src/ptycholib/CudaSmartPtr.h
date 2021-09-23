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

#ifndef __CUDASMARTPTR_H
#define __CUDASMARTPTR_H

//#include "Cuda2DArray.hpp"

class ICuda2DArray;
//class Cuda2DArray;

class CudaSmartPtr
{
private:
	ICuda2DArray* m_ptr;

public:

    CudaSmartPtr(ICuda2DArray* p = 0)
    {acquire(p);}

    CudaSmartPtr(const CudaSmartPtr& dtPtr)
    {acquire(dtPtr.get());}

    ~CudaSmartPtr()
    {release();}

    CudaSmartPtr& operator=(const CudaSmartPtr& dtPtr)
	{
		if (m_ptr != dtPtr.get())
		{
			release();
			acquire(dtPtr.get());
		}
		return *this;
	}
    bool operator==(const CudaSmartPtr& dtPtr) {return m_ptr == dtPtr.get();}

    ICuda2DArray* get()			const {return m_ptr;}
    ICuda2DArray& operator*()	const {return *m_ptr;}
    ICuda2DArray* operator->()	const {return m_ptr;}
    bool isValid()				const {return (m_ptr != 0);}

    template<typename T>
    bool initFromFile(const char*);

//	template<typename T>
//	bool loadCSV(const char* filename, bool binary=false);

    void acquire(ICuda2DArray* ptr);
    void release();
};

#endif
