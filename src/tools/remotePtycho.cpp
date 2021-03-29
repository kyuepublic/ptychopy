/*
 * remotePtycho.cpp
 *
 *  Created on: Mar 25, 2016
 *      Author: ynashed
 */

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


