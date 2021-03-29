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

#include "GLResourceManager.h"
#include "IRenderable.h"
#include "utilities.h"

using namespace std;

#ifdef HAVE_SDL2
void sdldie(const char *msg)
{
	fprintf(stderr,"%s: %s\n", msg, SDL_GetError());
    SDL_Quit();
    exit(1);
}
 
 
void checkSDLError(int line = -1)
{
#ifndef NDEBUG
        const char *error = SDL_GetError();
        if (*error != '\0')
        {
        		fprintf(stderr,"SDL Error: %s\n", error);
                if (line != -1)
                	fprintf(stderr," + line: %i\n", line);
                SDL_ClearError();
        }
#endif
}

GLResourceManager::GLResourceManager() : m_mainwindow(0), m_transferOffset(0.0f), m_transferScale(1.0f)
{}

GLResourceManager::~GLResourceManager()
{
	for(map<string, CUDAGL_Id>::iterator it=m_mappedResources.begin(); it!=m_mappedResources.end(); ++it)
	{
		cudaGraphicsUnregisterResource(it->second.cudaResource);

		glBindBuffer(1, it->second.glID);
		glDeleteBuffers(1, &it->second.glID);
	}
	m_mappedResources.clear();

	if(m_mainwindow)
	{
		SDL_GL_DeleteContext(m_glContext);
		SDL_DestroyWindow(m_mainwindow);
		SDL_Quit();
	}

	h_freeColorTransferTexture();
}
#else
GLResourceManager::GLResourceManager() {}

GLResourceManager::~GLResourceManager() {}
#endif

void GLResourceManager::init()
{
#ifdef HAVE_SDL2
	cudaSetDevice(0);
	//cudaGLSetGLDevice( 0 );

	// Create GL context
	if (SDL_Init(SDL_INIT_VIDEO) < 0) /* Initialize SDL's Video subsystem */
        sdldie("Unable to initialize SDL"); /* Or die on error */
 
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    //SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	h_initColorTransferTexture();
#endif
}

void GLResourceManager::initWindow(const char* title, unsigned int w, unsigned int h)
{
#ifdef HAVE_SDL2
	m_mainwindow = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		w, h, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
	if (!m_mainwindow) /* Die if creation failed */
		sdldie("Unable to create window");

	checkSDLError(__LINE__);
 
	/* Create our opengl context and attach it to our window */
	m_glContext = SDL_GL_CreateContext(m_mainwindow);
	checkSDLError(__LINE__);
 
 
	/* This makes our buffer swap syncronized with the monitor's vertical refresh */
	SDL_GL_SetSwapInterval(1);

	// initialize necessary OpenGL extensions
	glewInit();

	glClearColor(0.5, 0.5, 0.5, 1.0);
	glDisable(GL_DEPTH_TEST);
#endif
}

void GLResourceManager::setWindowPos(int x, int y)
{
#ifdef HAVE_SDL2
	if(!m_mainwindow)
		return;
	SDL_SetWindowPosition(m_mainwindow, x, y);
#endif
}

void GLResourceManager::getWindowSize(int* w, int* h)
{
#ifdef HAVE_SDL2
	if(!m_mainwindow)
		return;
	SDL_GetWindowSize(m_mainwindow, w, h);
#endif
}

void GLResourceManager::addResource(Resource r)
{
	IRenderer::addResource(r);
#ifdef HAVE_SDL2
	IRenderable* renderThis = r.renderable;
	unsigned int w=renderThis->getWidth(), h=renderThis->getHeight();
	if(!m_mainwindow)
		initWindow(r.resourceName.c_str(), w, h);
	else
	{
		int oldWidth, oldHeight;
		SDL_GetWindowSize(m_mainwindow, &oldWidth, &oldHeight);
		string oldTitle = SDL_GetWindowTitle(m_mainwindow);
		oldTitle += (" + " + string(r.resourceName));
		oldWidth+= w; oldHeight = max((int)h, oldHeight);
		SDL_SetWindowSize(m_mainwindow, oldWidth, oldHeight);
		SDL_SetWindowTitle(m_mainwindow, oldTitle.c_str());
	}

    GLuint pboID, texID;
	cudaGraphicsResource_t pbo_res;

	// create buffer object
	size_t bufferSize;
	switch(r.cmap)
	{
	case RAINBOW: bufferSize = w * h * sizeof(float4);break;
	case GRAYS: bufferSize = w * h * sizeof(float);break;
	default: bufferSize = w * h * sizeof(float4);break;
	}
	glGenBuffers(1,&pboID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);	glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, NULL, GL_STREAM_DRAW);

	// create texture
	glGenTextures(1, &texID);
	glBindTexture(GL_TEXTURE_2D, texID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(&pbo_res, pboID, cudaGraphicsMapFlagsWriteDiscard);
	cutilCheckMsg("GLResourceManager::addResource() failed!\n");

	m_mappedResources.insert(pair<string,CUDAGL_Id>(r.resourceName, CUDAGL_Id(pboID, texID, pbo_res)));
#endif
}

void GLResourceManager::handleWindowEvents()
{
#ifdef HAVE_SDL2
	SDL_Event sdl_event;
	while (SDL_PollEvent(&sdl_event)) 
	{
		switch (sdl_event.type) 
		{
		case SDL_WINDOWEVENT:
			switch (sdl_event.window.event) 
			{
			case SDL_WINDOWEVENT_CLOSE:
				if (m_mainwindow) 
				{
                    SDL_DestroyWindow(m_mainwindow);
					m_mainwindow = 0;
					SDL_VideoQuit();
                }
            break;
			}
        break;
		case SDL_KEYDOWN:
			switch (sdl_event.key.keysym.sym)
			{
			case SDLK_PLUS:
			case SDLK_EQUALS:
				m_transferOffset += 0.01f;
			break;
			case SDLK_MINUS:
				m_transferOffset -= 0.01f;
			break;
			case SDLK_RIGHTBRACKET:
				m_transferScale += 0.01f;
			break;
			case SDLK_LEFTBRACKET:
				m_transferScale -= 0.01f;
			break;
			case SDLK_SPACE:
				m_transferOffset = 0.0f;
				m_transferScale = 1.0f;
			break;
			}
		break;
		}
	}
#endif
}

void GLResourceManager::renderResources()
{
#ifdef HAVE_SDL2
	handleWindowEvents();
	if(!m_mainwindow)
		return;

	int xPos=0;
	//SDL_GetWindowSize(m_mainwindow, &xPos, &yPos);
	SDL_ShowWindow(m_mainwindow);
	for(unsigned int i=0; i<m_resources.size(); ++i)
	{
		map<string, CUDAGL_Id>::iterator it = m_mappedResources.find(m_resources[i].resourceName);

		if(!m_resources[i].renderable->isRenderableUpdated())
		{
			xPos += m_resources[i].renderable->getWidth();
			continue;
		}

		size_t num_bytes; void* mappedPtr;
		GLint texInternalFormat, texFormat;

		cudaGraphicsMapResources(1, &(it->second.cudaResource), 0);
		cudaGraphicsResourceGetMappedPointer(&mappedPtr, &num_bytes, it->second.cudaResource);
		cutilCheckMsg("GLResourceManager::getResourcePtr() cudaGraphicsResourceGetMappedPointer failed!\n");
		
		switch(m_resources[i].cmap)
		{
		case RAINBOW:
			m_resources[i].renderable->toRGBA((float4*)mappedPtr, m_resources[i].resourceName.c_str(), m_transferOffset, m_transferScale);
			texInternalFormat = GL_RGBA; texFormat = GL_RGBA;
		break;
		case GRAYS:
			m_resources[i].renderable->toGray((float*)mappedPtr, m_resources[i].resourceName.c_str());
			texInternalFormat = GL_INTENSITY; texFormat = GL_LUMINANCE;
		break;
		}

		cudaGraphicsUnmapResources(1, &it->second.cudaResource, 0);
		cutilCheckMsg("GLResourceManager::renderResource() cudaGraphicsUnmapResources failed!\n");

		glViewport(xPos, 0, m_resources[i].renderable->getWidth(), m_resources[i].renderable->getHeight());
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, it->second.glID);

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, it->second.textureID);

		glTexImage2D(GL_TEXTURE_2D, 0, texInternalFormat, m_resources[i].renderable->getWidth(), m_resources[i].renderable->getHeight(), 0, texFormat, GL_FLOAT, NULL);

		glBegin(GL_QUADS);
			glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0);
			glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0);
			glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0);
			glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0);
		glEnd();

		//glDrawPixels(it->second.glWidth, it->second.glHeight, GL_LUMINANCE, GL_FLOAT, NULL);
		glDisable(GL_TEXTURE_2D);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glPopMatrix();
		xPos += m_resources[i].renderable->getWidth();
	}

	SDL_GL_SwapWindow(m_mainwindow);
#endif
}
