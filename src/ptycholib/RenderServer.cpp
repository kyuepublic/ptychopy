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
#include "RenderServer.h"
#include "IRenderable.h"
#include "CudaSmartPtr.h"
#include "Cuda2DArray.hpp"
#include "utilities.h"
#include "WS/easywsclient.cpp"
#include <sstream>

using namespace std;
using namespace easywsclient;

RenderServer::RenderServer(const char* url) : m_bufferSize(0), m_framesBuffer(0)
{m_webSocket = url? (void*)WebSocket::from_url(url) : 0;}

RenderServer::~RenderServer()
{
	if(m_webSocket)
		delete ((WebSocket::pointer)m_webSocket);
	clearResources();
}

void RenderServer::addResource(Resource r)
{
	IRenderer::addResource(r);
	if(m_webSocket)
	{
		m_frames.push_back(new Cuda2DArray<real_t>(r.renderable->getHeight(), r.renderable->getWidth()));
		m_bufferSize += (r.renderable->getHeight()*r.renderable->getWidth());
		m_cudaStreams.push_back(cudaStream_t());
		cudaStreamCreate(&m_cudaStreams.back());
		cutilCheckMsg("RenderServer::addResource() CUDA stream creation failed!");
	}
}

void RenderServer::clearResources()
{
	IRenderer::clearResources();
	for(unsigned int i=0; i<m_cudaStreams.size(); ++i)
	{
		cudaStreamSynchronize(m_cudaStreams[i]);
		cudaStreamDestroy(m_cudaStreams[i]);
	}
	m_cudaStreams.clear();
	m_frames.clear();
	if(m_framesBuffer) 	cudaFreeHost(m_framesBuffer);
	m_framesBuffer = 0;
	m_bufferSize = 0;
}

void RenderServer::renderResources()
{
	if(m_webSocket)
	{
		stringstream stringBuffer;

		if(!m_framesBuffer) //First send
		{
			cudaMallocHost((void**)&m_framesBuffer, m_bufferSize*sizeof(real_t));//Allocate pinned memory
			cutilCheckMsg(" RenderServer::renderResources() cudaMallocHost failed!");

			//Make ASCII header
			stringBuffer << m_resources.size() << endl << sizeof(real_t);
			for(unsigned int i=0; i<m_resources.size(); ++i)
				stringBuffer << endl << m_resources[i].resourceName << " " <<
										m_resources[i].renderable->getHeight() << " " <<
										m_resources[i].renderable->getWidth();
			((WebSocket::pointer)m_webSocket)->send(stringBuffer.str()); //Send header in ASCII
			stringBuffer.str(string());
			stringBuffer.clear();
		}

		//Copy frames from GPU Memory to host buffer
		unsigned int framesBufferIndex = 0;
		for(unsigned int i=0; i<m_resources.size(); ++i)
		{
			m_resources[i].renderable->toGray(m_frames[i]->getDevicePtr<real_t>(), m_resources[i].resourceName.c_str(), true);
			m_frames[i]->getHostPtr<real_t>(m_framesBuffer+framesBufferIndex, &m_cudaStreams[i]);
			framesBufferIndex += m_frames[i]->getNum();
		}

		stringBuffer.write((char*)m_framesBuffer, m_bufferSize*sizeof(real_t));
		((WebSocket::pointer)m_webSocket)->sendBinary(stringBuffer.str()); //Send frames in binary

		((WebSocket::pointer)m_webSocket)->poll();
	}
}
