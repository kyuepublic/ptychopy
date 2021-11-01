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
//X-Ray Nanoimaging: Instruments and Methods V 2021 .
//International Society for Optics and Photonics.
//
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

#ifndef __CUDA2DARRAY_HPP
#define __CUDA2DARRAY_HPP

#include "utilities.h"
#include <fstream>
#include <iostream>
#include <iomanip>

class CudaSmartPtr;

class ICuda2DArray
{
private:
	unsigned int m_refCount;
	friend class CudaSmartPtr;

protected:
	uint2 m_dimensions;
	unsigned int m_alignedY;
	size_t m_unitSize;
	virtual ~ICuda2DArray() { }

	virtual void allocateDeviceMemory() = 0;
	virtual void freeDeviceMemory() = 0;

public:
	explicit ICuda2DArray(unsigned int x, unsigned int y, unsigned int tsize) : m_refCount(0),
																				m_unitSize(tsize)
	{
		m_dimensions.x = x;
		m_dimensions.y = y;
		m_alignedY = GPUQuery::getInstance()->alignToWarp(y);
//		m_alignedY = y;
	}

	virtual void set(int val=0) = 0;
	virtual void reshapeDeviceMemory(uint2 dims) = 0;

	virtual size_t getSize() 		const {return m_unitSize*(size_t)m_alignedY*(size_t)m_dimensions.x;}
	virtual size_t getUnitSize() 	const {return m_unitSize;}
	virtual uint2 getDimensions()	const {return m_dimensions;}
	virtual size_t getNum()			const {return (size_t)m_dimensions.x*(size_t)m_dimensions.y;}
	virtual unsigned int getX()		const {return m_dimensions.x;}
	virtual unsigned int getY()		const {return m_dimensions.y;}
	virtual unsigned int getAlignedY()	const {return m_alignedY;}

	template<typename T> void setFromHost(const T* copy, unsigned int x, unsigned int y, cudaStream_t* stream=0);
	template<typename T> void setFromDevice(const T* copy, unsigned int x, unsigned int y, cudaStream_t* stream=0);
	template<typename T> T* getDevicePtr() const;
	template<typename T> T* getHostPtr(T* ptr_h=0, cudaStream_t* stream=0) const;
	template<typename T> bool save(const char* filename, bool binary=false)	 const;
	template<typename T> bool load(const char* filename, bool binary=false);
	template<typename T> bool loadMatlab(const char* filename, bool binary=false);
	template<typename T>
	T toPrecision(T input, unsigned precision);

	template<typename T> bool load2Complex(char* filename1, char* filename2, bool binary=false);

//	bool loadCSVother(const char* filename, bool binary=false)
//	{
//		std::ifstream infile(filename, binary?std::ofstream::in|std::ofstream::binary : std::ofstream::in );
//		if(!infile.is_open())
//			return false;
//
//		complex_t* h_array =  getHostPtr<complex_t>();
//
//		if(binary)
//			infile.read((char*)h_array, getNum()*getUnitSize());
//		else
//		{
//			for(unsigned int x=0; x<getX(); ++x)
//				for(unsigned int y=0; y<getY(); ++y)
//					{
//					infile >> h_array[(x*getY())+y];
//					std::cout<<h_array[(x*getY())+y];
//
//					}
//
//		}
//		infile.close();
//		setFromHost(h_array, getX(), getY());
//		delete [] h_array;
//
//		return true;
//	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename T>
class Cuda2DArray : public ICuda2DArray
{

public:
	~Cuda2DArray()
	{
		freeDeviceMemory();
	}

	void freeDeviceMemory()
	{
		if(m_devicePtr)
		{
			cudaFree (m_devicePtr);
			cutilCheckMsg("Device memory deallocation failed!");
			m_devicePtr = 0;
		}
	}

private:
	T* m_devicePtr;

	void allocateDeviceMemory()
	{
//		printf("Size before allocation: %d:%d\n", getUnitSize(), getSize());
		cudaMalloc((void**)&m_devicePtr, getSize());
		cutilCheckMsg("Device memory allocation failed!");
		set();
	}

	void copyFromDeviceMemory(const T* copy, cudaStream_t* stream=0)
	{
		if(stream)
			cudaMemcpyAsync(m_devicePtr, copy, getSize(), cudaMemcpyDeviceToDevice, *stream);
		else
			cudaMemcpy(m_devicePtr, copy, getSize(), cudaMemcpyDeviceToDevice);
		cutilCheckMsg("Device memory copy failed!");
	}

public:
	explicit Cuda2DArray(unsigned int x, unsigned int y) : ICuda2DArray(x,y, sizeof(T)),
														   m_devicePtr(0)

	{
		allocateDeviceMemory();
	}

	//We don't want to call the default copy constructor
	Cuda2DArray(const Cuda2DArray<T>& copy) : ICuda2DArray(copy.getX(), copy.getY(), copy.getUnitSize()),
												m_devicePtr(0)
    {
		allocateDeviceMemory();
		copyFromDeviceMemory(copy.getDevicePtr());
	}

	Cuda2DArray(const ICuda2DArray* copy) : ICuda2DArray(copy->getX(), copy->getY(), copy->getUnitSize()),
											 m_devicePtr(0)
    {
		allocateDeviceMemory();
		copyFromDeviceMemory(copy->getDevicePtr<T>());
	}

	Cuda2DArray(const T* copy, unsigned int x, unsigned int y, bool host=false) : ICuda2DArray(x, y, sizeof(T)),
																					m_devicePtr(0)
    {
		allocateDeviceMemory();
		if(host)
			setFromHost(copy, x, y);
		else
			copyFromDeviceMemory(copy);
	}

    Cuda2DArray & operator= (const Cuda2DArray<T>& copy)
    {
        if (this != &copy)
        {
        	reshapeDeviceMemory(copy.getDimensions());
			copyFromDeviceMemory(copy.getDevicePtr());
        }
        return *this;
    }

	Cuda2DArray & operator= (const ICuda2DArray& copy)
    {
        if (this != &copy)
        {
        	reshapeDeviceMemory(copy.getDimensions());
			copyFromDeviceMemory(copy.getDevicePtr<T>());
        }
        return *this;
    }

	void reshapeDeviceMemory(uint2 dims)
	{
		if(m_dimensions.x!=dims.x || m_dimensions.y!=dims.y)
		{
			freeDeviceMemory();
			m_dimensions = dims;
			m_unitSize = sizeof(T);
			m_alignedY = GPUQuery::getInstance()->alignToWarp(dims.y);
			allocateDeviceMemory();
		}
	}

    void set(int val=0) //WARNING: val will only be set to bytes of memory. Use only for byte arrays if val!=0
    {
    	cudaMemset(m_devicePtr, val, getSize());
    	cutilCheckMsg("Device memory set (cudaMemset) failed!");
    }

    void setFromHost(const T* copy, unsigned int x, unsigned int y, cudaStream_t* stream=0)
    {
    	reshapeDeviceMemory(make_uint2(x,y));
    	if(stream)
			cudaMemcpy2DAsync(m_devicePtr, m_alignedY*m_unitSize, copy, m_dimensions.y*m_unitSize, m_dimensions.y*m_unitSize, m_dimensions.x, cudaMemcpyHostToDevice, *stream);
		else
			cudaMemcpy2D(m_devicePtr, m_alignedY*m_unitSize, copy, m_dimensions.y*m_unitSize, m_dimensions.y*m_unitSize, m_dimensions.x, cudaMemcpyHostToDevice);
    	cutilCheckMsg("Host to Device memory copy failed!");
    }

    void setFromDevice(const T* copy, unsigned int x, unsigned int y, cudaStream_t* stream=0)
	{
		reshapeDeviceMemory(make_uint2(x,y));
		copyFromDeviceMemory(copy, stream);
	}

	T* getDevicePtr() const
	{
		return m_devicePtr;
	}

	T* getHostPtr(T* ptr_h=0, cudaStream_t* stream=0) const
	{
		T* hostPtr = ptr_h;
		if(hostPtr==0)
//			hostPtr = new T[m_dimensions.x*m_dimensions.y]; //user should delete
			hostPtr = new T[m_dimensions.x*m_dimensions.y];
		if(stream)
			cudaMemcpy2DAsync(hostPtr, m_dimensions.y*m_unitSize, m_devicePtr, m_alignedY*m_unitSize, m_dimensions.y*m_unitSize, m_dimensions.x, cudaMemcpyDeviceToHost, *stream);
		else
			cudaMemcpy2D(hostPtr, m_dimensions.y*m_unitSize, m_devicePtr, m_alignedY*m_unitSize, m_dimensions.y*m_unitSize, m_dimensions.x, cudaMemcpyDeviceToHost);
		cutilCheckMsg("Device to Host memory copy failed!");
		return hostPtr;
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Cuda3DElement : public ICuda2DArray
{
private:
	T* m_devicePtr;

	void allocateDeviceMemory(){}
	void freeDeviceMemory(){}
	void set(int val=0){}
	void reshapeDeviceMemory(uint2 dims){}

public:
	Cuda3DElement(T* ptr, unsigned int x, unsigned int y) : ICuda2DArray(x,y, sizeof(T)),
														    m_devicePtr(ptr)
	{}

	~Cuda3DElement() {}

	void setFromHost(const T* copy, unsigned int x, unsigned int y, cudaStream_t* stream=0)
	{
		if(x>m_dimensions.x || y>m_dimensions.y)
		{
			fprintf(stderr, "WARNING: Cuda3DElement::setFromHost() failed! Dimension mismatch.");
			return;
		}
		if(stream)
			cudaMemcpy2DAsync(m_devicePtr, m_alignedY*m_unitSize, copy, y*m_unitSize, y*m_unitSize, x, cudaMemcpyHostToDevice, *stream);
		else
			cudaMemcpy2D(m_devicePtr, m_alignedY*m_unitSize, copy, y*m_unitSize, y*m_unitSize, x, cudaMemcpyHostToDevice);
		cutilCheckMsg("Host to Device memory copy failed!");
	}

	void setFromDevice(const T* copy, unsigned int x, unsigned int y, cudaStream_t* stream=0)
	{
		if(stream)
			cudaMemcpyAsync(m_devicePtr, copy, getSize(), cudaMemcpyDeviceToDevice, *stream);
		else
			cudaMemcpy(m_devicePtr, copy, getSize(), cudaMemcpyDeviceToDevice);
		cutilCheckMsg("Device memory copy failed!");
	}

	T* getDevicePtr() const
	{return m_devicePtr;}

	T* getHostPtr(T* ptr_h=0, cudaStream_t* stream=0) const
	{
		T* hostPtr = ptr_h;
		if(hostPtr==0)
			hostPtr = new T[m_dimensions.x*m_dimensions.y]; //user should delete
		if(stream)
			cudaMemcpy2DAsync(hostPtr, m_dimensions.y*m_unitSize, m_devicePtr, m_alignedY*m_unitSize, m_dimensions.y*m_unitSize, m_dimensions.x, cudaMemcpyDeviceToHost, *stream);
		else
			cudaMemcpy2D(hostPtr, m_dimensions.y*m_unitSize, m_devicePtr, m_alignedY*m_unitSize, m_dimensions.y*m_unitSize, m_dimensions.x, cudaMemcpyDeviceToHost);
		cutilCheckMsg("Device to Host memory copy failed!");
		return hostPtr;
	}
};

#include "CudaSmartPtr.h"

template<typename T>
class Cuda3DArray
{
private:
	unsigned int m_num;
	uint2 m_dimensions;
	bool m_useAll;
	CudaSmartPtr m_array;

public:
	Cuda3DArray():  m_num(0), m_dimensions(make_uint2(0,0)), m_useAll(true), m_array(0)
	{

	}

	Cuda3DArray(unsigned int n, uint2 dims) : m_useAll(true)
	{
		init(n,dims);
	}
	//We don't want to call the default copy constructor
	Cuda3DArray(const Cuda3DArray& copy) : m_useAll(true)
	{
		init(copy.getNum(), copy.getDimensions(), copy.getPtr().get());
	}

	~Cuda3DArray(){}

	void init(unsigned int n, uint2 dims, ICuda2DArray* ptr=0)
	{
		m_num=n; m_dimensions=dims;
		if(ptr)
			m_array = new Cuda2DArray<T>(ptr);
		else
			m_array = new Cuda2DArray<T>(m_num*m_dimensions.x, m_dimensions.y);
	}

	void setUseAll(bool f)
	{
		m_useAll = f;
	}
	bool checkUseAll()	const
	{
		return m_useAll;
	}
	unsigned int getNum()	const
	{
		return m_num;
	}
	uint2 getDimensions()	const
	{
		return m_dimensions;
	}
	const CudaSmartPtr& getPtr()	const
	{
		return m_array;
	}
	void setToZeroes()
	{
		m_array->set();
	}
	Cuda3DElement<T> getAt(unsigned int i)	const 
	{
		return Cuda3DElement<T>(m_array->getDevicePtr<T>()+((size_t)i*(size_t)(m_array->getX()/m_num)*(size_t)m_array->getAlignedY()),
							m_dimensions.x, m_dimensions.y);
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void ICuda2DArray::setFromHost(const T* copy, unsigned int x, unsigned int y, cudaStream_t* stream)
{
	(dynamic_cast<Cuda2DArray<T>&>(*this)).setFromHost(copy, x, y, stream);
}

template<typename T>
void ICuda2DArray::setFromDevice(const T* copy, unsigned int x, unsigned int y, cudaStream_t* stream)
{
	(dynamic_cast<Cuda2DArray<T>&>(*this)).setFromDevice(copy, x, y, stream);
}


template<typename T>
T* ICuda2DArray::getDevicePtr() const
{
	return (dynamic_cast<const Cuda2DArray<T>&>(*this)).getDevicePtr();
}

template<typename T>
T* ICuda2DArray::getHostPtr(T* ptr_h, cudaStream_t* stream) const
{
	return (dynamic_cast<const Cuda2DArray<T>&>(*this)).getHostPtr(ptr_h, stream);
}

std::ostream& operator<<(std::ostream &out, const complex_t &rhs);
std::istream& operator>>(std::istream &in, complex_t &rhs);
//std::ostream& operator<<(std::ostream &out, const float2 &rhs);
//std::istream& operator>>(std::istream &in, float2 &rhs);

template<typename T>
bool ICuda2DArray::save(const char* filename, bool binary)	 const
{
	std::ofstream outfile(filename, binary?std::ofstream::out|std::ofstream::binary : std::ofstream::out);
	if(!outfile.is_open())
		return false;
	T* h_array =  getHostPtr<T>();
	if(binary)
		outfile.write((const char*)h_array, getNum()*getUnitSize());
	else
		for(unsigned int x=0; x<getX(); ++x)
		{
			for(unsigned int y=0; y<getY(); ++y)
			{
				outfile << h_array[(x*getY())+y];

				if(y<getY()-1)
					outfile << ", ";
			}
			outfile << std::endl;
		}
	outfile.close();
	delete [] h_array;

	return true;
}

template<typename T>
bool ICuda2DArray::load2Complex(char* filename1, char* filename2, bool binary)
{

	std::ifstream infile1(filename1, binary?std::ofstream::in|std::ofstream::binary : std::ofstream::in );
	std::ifstream infile2(filename2, binary?std::ofstream::in|std::ofstream::binary : std::ofstream::in );

	if(!infile1.is_open())
		return false;
	if(!infile2.is_open())
		return false;

	T* h_array =  getHostPtr<T>();

	std::string line1 = "";
	std::string line2 = "";

	int rowIdx=0;
	double val1=0;
	double val2=0;

	while (std::getline(infile1, line1)&&std::getline(infile2, line2))
	{
        // Create a stringstream of the current line
        std::stringstream ss1(line1);
        std::stringstream ss2(line2);

        // Keep track of the current column index
        int colIdx = 0;

        // Extract each integer
        while(ss1 >> val1 && ss2 >> val2)
        {
            // Add the current integer to the 'colIdx' column's values vector
        	h_array[(rowIdx*getY())+colIdx]=make_float2(val1, val2);

            // If the next token is a comma, ignore it and move on
            if(ss1.peek() == ',') ss1.ignore();
            if(ss2.peek() == ',') ss2.ignore();
            // Increment the column index
            colIdx++;
        }

        rowIdx++;

	}
	infile1.close();
	infile2.close();

	setFromHost(h_array, getX(), getY());
	delete [] h_array;

	return true;
}

template<typename T>
bool ICuda2DArray::load(const char* filename, bool binary)
{
	std::ifstream infile(filename, binary?std::ofstream::in|std::ofstream::binary : std::ofstream::in );
	if(!infile.is_open())
		return false;
	T* h_array =  getHostPtr<T>();
	if(binary)
		infile.read((char*)h_array, getNum()*getUnitSize());
	else
	{
		for(unsigned int x=0; x<getX(); ++x)
			for(unsigned int y=0; y<getY(); ++y)
				infile >> h_array[(x*getY())+y];
	}
	infile.close();
	setFromHost(h_array, getX(), getY());
	delete [] h_array;

	return true;
}

template<typename T>
T ICuda2DArray::toPrecision(T input, unsigned precision)
{
    static std::stringstream ss;

    T output;
    ss << std::fixed;
    ss.precision(precision);
    ss << input;
    ss >> output;
    ss.clear();

    return output;
}

template<typename T>
bool ICuda2DArray::loadMatlab(const char* filename, bool binary)
{

	char* filename1="/data2/JunjingData/m1.csv";
	char* filename2="/data2/JunjingData/m2.csv";

//	std::ifstream infile(filename, binary?std::ofstream::in|std::ofstream::binary : std::ofstream::in );
	std::ifstream infile1(filename1, binary?std::ofstream::in|std::ofstream::binary : std::ofstream::in );
	std::ifstream infile2(filename2, binary?std::ofstream::in|std::ofstream::binary : std::ofstream::in );

//	if(!infile.is_open())
//		return false;
	if(!infile1.is_open())
		return false;
	if(!infile2.is_open())
		return false;
//
//	float temp1=0;
//	float temp2=0;

	T* h_array =  getHostPtr<T>();

//	std::vector<std::vector<std::string> > dataList;

	std::string line1 = "";
	std::string line2 = "";

	int rowIdx=0;
	float val1=0;
	float val2=0;
//	std::setprecision(4);

	while (std::getline(infile1, line1)&&std::getline(infile2, line2))
	{
        // Create a stringstream of the current line
        std::stringstream ss1(line1);
        std::stringstream ss2(line2);

        // Keep track of the current column index
        int colIdx = 0;

        // Extract each integer
        while(ss1 >> val1 && ss2 >> val2)
        {

            // Add the current integer to the 'colIdx' column's values vector
//            result.at(colIdx).second.push_back(val);
//        	double val3=toPrecision(val1, 3);
//        	double val4=toPrecision(val2, 3);
        	h_array[(rowIdx*getY())+colIdx]=make_float2(val1, val2);

            // If the next token is a comma, ignore it and move on
            if(ss1.peek() == ',') ss1.ignore();
            if(ss2.peek() == ',') ss2.ignore();
            // Increment the column index
            colIdx++;
        }

        rowIdx++;

	}

//	Close the File
	infile1.close();
	infile2.close();

	setFromHost(h_array, getX(), getY());
	delete [] h_array;

	return true;
}
//
//bool ICuda2DArray::loadCSVothers(const char* filename, bool binary)
//{
//	std::ifstream infile(filename, binary?std::ofstream::in|std::ofstream::binary : std::ofstream::in );
//	if(!infile.is_open())
//		return false;
//
//	complex_t* h_array =  getHostPtr<complex_t>();
//
//	if(binary)
//		infile.read((char*)h_array, getNum()*getUnitSize());
//	else
//	{
//		for(unsigned int x=0; x<getX(); ++x)
//			for(unsigned int y=0; y<getY(); ++y)
//				{
//				infile >> h_array[(x*getY())+y];
//				std::cout<<h_array[(x*getY())+y];
//
//				}
//
//	}
//	infile.close();
//	setFromHost(h_array, getX(), getY());
//	delete [] h_array;
//
//	return true;
//}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
