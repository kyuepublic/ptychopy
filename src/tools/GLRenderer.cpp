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

#include "GLRenderer.h"
#include <iostream>

using namespace std;

void sdldie(const char *msg)
{
	fprintf(stderr,"%s: %s\n", msg, SDL_GetError());
    SDL_Quit();
    exit(1);
}


void checkSDLError(int line = -1)
{
	const char *error = SDL_GetError();
	if (*error != '\0')
	{
			fprintf(stderr,"SDL Error: %s\n", error);
			if (line != -1)
				fprintf(stderr," + line: %i\n", line);
			SDL_ClearError();
	}
}

/////////////////////////////////////////////////////////////

GLRenderer::GLRenderer() : m_mainwindow(0)
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0) /* Initialize SDL's Video subsystem */
		sdldie("Unable to initialize SDL"); /* Or die on error */

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	//SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
}

GLRenderer::~GLRenderer()
{
	for(size_t i=0; i<m_renderables.size(); ++i)
	{
		glDeleteTextures(1, &(m_renderables[i].texID));
		delete m_renderables[i].renderable;
	}
	m_renderables.clear();
	if(m_mainwindow)
	{
		SDL_GL_DeleteContext(m_glContext);
		SDL_DestroyWindow(m_mainwindow);
		SDL_Quit();
	}
}

void GLRenderer::initWindow(const char* title, unsigned int w, unsigned int h)
{
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
	//glewInit();

	glClearColor(0.5, 0.5, 0.5, 1.0);
	glDisable(GL_DEPTH_TEST);
}

void GLRenderer::setWindowPos(int x, int y)
{
	if(!m_mainwindow)
		return;
	SDL_SetWindowPosition(m_mainwindow, x, y);
}

void GLRenderer::getWindowSize(int* w, int* h)
{
	if(!m_mainwindow)
		return;
	SDL_GetWindowSize(m_mainwindow, w, h);
}


void GLRenderer::addRenderable(GLRenderable* r)
{
	unsigned int w=r->getWidth(), h=r->getHeight();
	if(!m_mainwindow)
		initWindow(r->getName().c_str(), w, h);
	else
	{
		int oldWidth, oldHeight;
		SDL_GetWindowSize(m_mainwindow, &oldWidth, &oldHeight);
		string oldTitle = SDL_GetWindowTitle(m_mainwindow);
		oldTitle += (" + " + r->getName());
		oldWidth+= w; oldHeight = max((int)h, oldHeight);
		SDL_SetWindowSize(m_mainwindow, oldWidth, oldHeight);
		SDL_SetWindowTitle(m_mainwindow, oldTitle.c_str());
	}
	GLuint texID;
	// create texture
	glGenTextures(1, &texID);
	glBindTexture(GL_TEXTURE_2D, texID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	m_renderables.push_back(Resource(texID,r));
}

void GLRenderer::handleWindowEvents()
{
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
			break;
			case SDLK_MINUS:
			break;
			case SDLK_RIGHTBRACKET:
			break;
			case SDLK_LEFTBRACKET:
			break;
			case SDLK_SPACE:
			break;
			}
		break;
		}
	}
}

void GLRenderer::renderResources(const char* allFrames)
{
	handleWindowEvents();
	if(!m_mainwindow)
		return;

	int xPos=0, bufferIndex=0;
	//SDL_GetWindowSize(m_mainwindow, &xPos, &yPos);
	SDL_ShowWindow(m_mainwindow);
	for(size_t i=0; i<m_renderables.size(); ++i)
	{
		GLRenderable* r = m_renderables[i].renderable;
		unsigned int width = r->getWidth();
		unsigned int height = r->getHeight();

		size_t num_bytes; void* mappedPtr;

		glViewport(xPos, 0, width, height);

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, m_renderables[i].texID);

		const GLvoid* texData = (const GLvoid*) (allFrames+bufferIndex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY, width, height, 0, GL_LUMINANCE, GL_FLOAT, texData);

		glBegin(GL_QUADS);
			glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0);
			glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0);
			glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0);
			glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0);
		glEnd();

		glDisable(GL_TEXTURE_2D);
		glPopMatrix();

		xPos += width;
		bufferIndex += (width*height*r->getDataBytes());
	}

	SDL_GL_SwapWindow(m_mainwindow);
}
