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

#ifndef GLRESOURCEMANAGER_H_
#define GLRESOURCEMANAGER_H_

#include <string>
#include <map>
#include "IRenderer.h"

#ifdef HAVE_SDL2
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <SDL.h>
#undef main

struct CUDAGL_Id
{
	GLuint glID, textureID;
	cudaGraphicsResource_t cudaResource;
	CUDAGL_Id(GLuint gID, GLuint tID, cudaGraphicsResource_t cuRes) :
				glID(gID), textureID(tID), cudaResource(cuRes)
	{}
};
#endif

class GLResourceManager : public IRenderer
{
private:

#ifdef HAVE_SDL2
	SDL_Window* m_mainwindow;
	SDL_GLContext m_glContext;

	std::map<std::string,CUDAGL_Id> m_mappedResources;

	float m_transferOffset;
	float m_transferScale;
#endif

	void handleWindowEvents();
	void initWindow(const char* title, unsigned int w, unsigned int h);

	void addResource(Resource);

public:
	GLResourceManager();
	~GLResourceManager();

	void init();
	void setWindowPos(int,int);
	void getWindowSize(int*,int*);
	void renderResources();
};

#endif
