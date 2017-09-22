#include "DataReceiver.h"
#include <iostream>
#include "../Monitor/Filter.hpp"
#include "../Models/ResultSegment.hpp"

#pragma comment(lib, "ws2_32.lib")

// Definition of all variables used in network
int HostPort = 8889;
SOCKET ServerSocket = 0;
sockaddr_in RemoteAddress;
int RemoteAddressLen = 0;
int ReveiceDataBufferlen = 0;
unsigned char* ReceiveDataBuffer;

bool InitNetworks()
{
	logPrinter.PrintLogs("Init Network for receive frame data from remote device!",LogLevel::Info);

	// Init Network environment
	WSADATA wsaData;
	auto sockVersion = MAKEWORD(2, 2);
	if (WSAStartup(sockVersion, &wsaData) != 0)
	{
		logPrinter.PrintLogs("Init network failed!", LogLevel::Error);
		return false;
	}

	// Create Socket
	ServerSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (ServerSocket == SOCKET_ERROR)
	{
		logPrinter.PrintLogs(" Create scoket error!", LogLevel::Error);
		return false;
	}

	// Bind network address
	sockaddr_in serverAddress;
	serverAddress.sin_family = AF_INET;
	serverAddress.sin_port = htons(HostPort);
	serverAddress.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	if (bind(ServerSocket, reinterpret_cast<sockaddr *>(&serverAddress), sizeof(serverAddress)) == SOCKET_ERROR)
	{
		logPrinter.PrintLogs("Bind socket failed!", LogLevel::Error);
		closesocket(ServerSocket);
		return false;
	}

	// definition of remote address
	RemoteAddressLen = sizeof(RemoteAddress);

	// definition of receive data buffer
	ReveiceDataBufferlen = WIDTH * HEIGHT * BYTESIZE / 4;
	ReceiveDataBuffer = new unsigned char[ReveiceDataBufferlen];

	return true;
}

bool GetOneFrameFromNetwork(unsigned char* frameData)
{
	std::cout << "Receive one frame data from rempote device \n";
	for (auto i = 0; i < 4; ++i)
	{
		auto ret = recvfrom(ServerSocket, reinterpret_cast<char*>(ReceiveDataBuffer), ReveiceDataBufferlen, 0, reinterpret_cast<sockaddr *>(&RemoteAddress), &RemoteAddressLen);
		if (ret > 10)
		{
			memcpy(frameData + i * ReveiceDataBufferlen, ReceiveDataBuffer, sizeof(unsigned char) * ReveiceDataBufferlen);
			std::cout << "Data segment " << i + 1 << "\n";
		}
		else if(ret == 10)
		{
			std::cout << "Finish receive data from client!\n";
			return false;
		}
	}
	return true;
}

bool SendResultToRemoteServer(ResultSegment& result)
{
	std::cout << "Sending result to remote server \n";

	auto sendStatus = sendto(ServerSocket, reinterpret_cast<char*>(&result), sizeof(ResultSegment), 0, reinterpret_cast<sockaddr *>(&RemoteAddress), RemoteAddressLen);
	if(sendStatus == SOCKET_ERROR)
	{
		return false;
	}
	return true;
}

bool DestroyNetWork()
{
	delete[] ReceiveDataBuffer;

	closesocket(ServerSocket);
	WSACleanup();
	return true;
}
