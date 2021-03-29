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

#ifndef IRENDERABLE_H_
#define IRENDERABLE_H_

#include <vector>
#include <string>

struct float4;
class IRenderable;

typedef enum
{
	RAINBOW,
	GRAYS
}COLOR_MAP;

struct Resource
{
	std::string resourceName;
	COLOR_MAP cmap;
	IRenderable* renderable;

	Resource(std::string rName="", COLOR_MAP cm=GRAYS, IRenderable* r=0) :
				resourceName(rName), cmap(cm), renderable(r)
	{}
};

class IRenderable
{
protected:
	bool m_renderableUpdated;
	std::vector<Resource> m_myResources;

public:
	IRenderable() : m_renderableUpdated(false)
	{}
	virtual ~IRenderable(){}

	virtual const std::vector<Resource>& getResources()	{return m_myResources;}
	virtual void fillResources()						= 0;
	virtual void toRGBA(float4*,const char*,float,float)= 0;
	virtual void toGray(float*,const char*,bool=false)	= 0;
	virtual unsigned int getWidth()	 const				= 0;
	virtual unsigned int getHeight() const				= 0;
	
	bool isRenderableUpdated()  const {return m_renderableUpdated;}
};

#endif /* IRENDERABLE_H_ */
