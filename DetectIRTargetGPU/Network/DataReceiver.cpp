﻿#include "DataReceiver.h"
#include <iostream>

#pragma comment(lib, "ws2_32.lib")

int hostPort = 8889;
int width = 320;
int height = 256;

SOCKET serverSocket = 0;
sockaddr_in remoteAddress;

int remoteAddressLen;
int reveiceDataBufferlen = 0;

char* receiveDataBuffer = nullptr;

bool InitNetworks()
{
	std::cout << "Init" << std::endl;

	// Init Network environment
	WSADATA wsaData;
	auto sockVersion = MAKEWORD(2, 2);
	if (WSAStartup(sockVersion, &wsaData) != 0)
	{
		std::cout << "Init network failed!" << std::endl;
		return false;
	}

	// Create Socket
	serverSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (serverSocket == SOCKET_ERROR)
	{
		std::cout << "Scoket error!" << std::endl;
		return false;
	}

	// Bind network address
	sockaddr_in serverAddress;
	serverAddress.sin_family = AF_INET;
	serverAddress.sin_port = htons(hostPort);
	serverAddress.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	if (bind(serverSocket, reinterpret_cast<sockaddr *>(&serverAddress), sizeof(serverAddress)) == SOCKET_ERROR)
	{
		std::cout << "Bind socket failed!" << std::endl;
		closesocket(serverSocket);
		return false;
	}

	// definition of remote address
	remoteAddressLen = sizeof(remoteAddress);

	// definition of receive data buffer
	reveiceDataBufferlen = width * height / 4;
	receiveDataBuffer = new char[reveiceDataBufferlen];

	return true;
}

void Run(unsigned char* frameData)
{
	std::cout << "Wait for client" << std::endl;

	for (auto i = 0; i < 4; ++i)
	{
		auto ret = recvfrom(serverSocket, receiveDataBuffer, reveiceDataBufferlen, 0, reinterpret_cast<sockaddr *>(&remoteAddress), &remoteAddressLen);
		if (ret > 0)
		{
			memcpy(frameData + i * reveiceDataBufferlen, receiveDataBuffer, sizeof(unsigned char) * reveiceDataBufferlen);
			std::cout << "Receive data from clent!" << std::endl;
		}
		else
		{
			std::cout << " Finish receive data from client" << std::endl;
			break;
		}
	}
}

bool DestroyNetWork()
{
	delete[] receiveDataBuffer;

	closesocket(serverSocket);
	WSACleanup();
	return true;
}
