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


#include "Cuda2DArray.hpp"
#include <string>

using namespace std;

ostream& operator<<(ostream &out, const complex_t &rhs)
{
	out << rhs.x;
	if(rhs.y>=0) out << "+";
	out << rhs.y << "j";
	return out;
}

istream& operator>>(istream &in, complex_t &rhs)
{
	string complex;
	in >> complex;
	size_t found = complex.find(',');
	if (found!=string::npos)
		complex.erase(found, 1);
	found = complex.find('(');
	if (found!=string::npos)
		complex.erase(found, 1);
	found = complex.find(')');
	if (found!=string::npos)
		complex.erase(found, 1);
	found = complex.find('j');
	if (found!=string::npos)
		complex.erase(found, 1);
	found = complex.find('i');
	if (found!=string::npos)
		complex.erase(found, 1);
	found = complex.find("+-");
	if (found!=string::npos)
		complex.erase(found, 1);

	found = complex.find_first_of('e'); //scientific notation
	if (found!=string::npos)
	{
		rhs.x = atof(complex.substr(0, found+4).c_str());
		rhs.y = atof(complex.substr(found+4, complex.length()-(found+4)).c_str());
	}
	else							   //non-scientific format
	{
		found = complex.find_last_of('+');
		if (found!=string::npos)
		{
			rhs.x = atof(complex.substr(0, found).c_str());
			rhs.y = atof(complex.substr(found+1, complex.length()-found).c_str());
		}
		else
		{
			found = complex.find_last_of('-');
			if (found!=string::npos)
			{

				rhs.x = atof(complex.substr(0, found).c_str());
				rhs.y = atof(complex.substr(found, complex.length()-found).c_str());
			}
			else
			{
				rhs.x = atof(complex.c_str());
				rhs.y = 0;
			}
		}
	}
	return in;
}

//ostream& operator<<(ostream &out, const float2 &rhs)
//{
//	out << rhs.x << ' ' << rhs.y;
//	return out;
//}
//
//istream& operator>>(istream &in, float2 &rhs)
//{
//	in >> rhs.x;
//	in >> rhs.y;
//	return in;
//}
