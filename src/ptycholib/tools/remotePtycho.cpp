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

#include "GLRenderer.h"
#include <pthread.h>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <iostream>

using namespace std;

typedef websocketpp::server<websocketpp::config::asio> Server;

void onMessage(websocketpp::connection_hdl, Server::message_ptr msg)
{
	if(msg->get_opcode() == websocketpp::frame::opcode::text) //read header
	{
		stringstream hdr(msg->get_payload());
		int nFrames = 0, bytes = 4;
		hdr >> nFrames;
		hdr >> bytes;
		for(int n=0; n<nFrames; ++n)
		{
			unsigned int w=0,h=0;
			string name;
			hdr >> name;
			hdr >> h;
			hdr >> w;
			renderer_ptr->addRenderable(new GLRenderable(name, bytes, w, h));
		}
	}
	else if(msg->get_opcode() == websocketpp::frame::opcode::binary) //render frames
		renderer_ptr->renderResources(msg->get_payload().c_str());
}

void* startServer(void *args)
{
	Server* serverPtr = (Server*)args;
	serverPtr->set_message_handler(&onMessage);
	serverPtr->init_asio();
	serverPtr->listen(8889);
	serverPtr->start_accept();
	serverPtr->run();
	return NULL;
}

int main(int argc, char *argv[])
{
	if(argc<4)
	{
		fprintf(stderr, "Usage: remotePtycho <username> <host> <command file> [port]\n");
		return 0;
	}

	Server server;
	pthread_t wsServerThread;
	if(pthread_create(&wsServerThread, NULL, startServer, &server))
	{
		fprintf(stderr, "Error creating websocket server thread\n");
		return 1;
	}

	FILE *in;
	char command[10240], msg[10240];
	if(!(in = fopen(argv[3], "r")))
	{
		fprintf(stderr, "Error opening command file\n");
		return 2;
	}
	if(!(fgets(msg, sizeof(msg), in)))
	{
		fprintf(stderr, "Error reading from command file\n");
		return 3;
	}
	fclose(in);
	string port = "22";
	if(argc>=4)
		port = argv[4];
	sprintf(command, "ssh -p %s -R 8889:localhost:8889 %s@%s '%s'", port.c_str(), argv[1], argv[2], msg);
	if(!(in = popen(command, "r")))
	{
		fprintf(stderr, "Error connecting (ssh) to reconstruction machine\n");
		return 4;
	}

	while(fgets(msg, sizeof(msg), in)!=NULL);
		cout << msg;
	pclose(in);
	server.stop();

	if(pthread_join(wsServerThread, NULL))
	{
		fprintf(stderr, "Error joining server thread\n");
		return 5;
	}

	return 0;
}


