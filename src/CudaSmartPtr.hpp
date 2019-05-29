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

#ifndef CUDASMARTPTR_HPP_
#define CUDASMARTPTR_HPP_

#include "CudaSmartPtr.h"
#include "Cuda2DArray.hpp"
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>

using namespace std;

template<typename T>
bool CudaSmartPtr::initFromFile(const char* fname)
{
	string temp(fname);
	int dotPos = temp.find_last_of('.');
	if(dotPos<=0)
		return false;
	string extension = temp.substr(dotPos+1, temp.length()-dotPos);
	transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

	if(extension != "csv")
		return false;

	ifstream inFile(fname);
	if(!inFile.good())
		return false;
	unsigned int rows = count(istreambuf_iterator<char>(inFile), istreambuf_iterator<char>(), '\n');
	inFile.seekg(0);
	getline(inFile, temp);

	istringstream ss(temp.c_str());
	unsigned int cols = 0;
	while(getline(ss, temp, ','))
		cols++;

	acquire(new Cuda2DArray<T>(rows,cols));
	return true;
}



#endif /* CUDASMARTPTR_HPP_ */
