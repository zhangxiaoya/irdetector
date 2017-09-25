#include "DataReceiver.h"
#include <iostream>
#include "../Monitor/Filter.hpp"
#include "../Models/ResultSegment.hpp"

#pragma comment(lib, "ws2_32.lib")

// Definition of all variables used in network
int HostPortForRemoteDataServer = 8889;        // 接收数据端口
int HostPortForRemoteResultServer = 8888;      // ToDo
SOCKET RemoteDataServerSocket = 0;             // 接收数据SOCKET
SOCKET RemoteResultServerSocket = 0;           // ToDo
sockaddr_in RemoteDataServerSocketAddress;     // 接收数据Socket地址
sockaddr_in RemoteResultServerSocketAddress;   // ToDo
int RemoteDataServerSocketAddressLen = 0;      // 接收数据Socket地址长度
int RemoteResultServerSocketAddressLen = 0;    // ToDo

int ReveiceDataBufferlen = 0;                  // 接收数据缓冲区大小
unsigned char* ReceiveDataBuffer;              // 接收

bool InitNetworkEnvironment()
{
	WSADATA wsaData;
	auto sockVersion = MAKEWORD(2, 2);
	if (WSAStartup(sockVersion, &wsaData) != 0)
	{
		logPrinter.PrintLogs("Init network failed!", LogLevel::Error);
		return false;
	}
	return true;
}

bool InitSocketForDataServer()
{
	// Create Socket for data server
	RemoteDataServerSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (RemoteDataServerSocket == SOCKET_ERROR)
	{
		logPrinter.PrintLogs(" Create remote data server scoket error!", LogLevel::Error);
		return false;
	}

	// Bind network address for data server and socket address
	sockaddr_in serverAddress;
	serverAddress.sin_family = AF_INET;
	serverAddress.sin_port = htons(HostPortForRemoteDataServer);
	serverAddress.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	if (bind(RemoteDataServerSocket, reinterpret_cast<sockaddr *>(&serverAddress), sizeof(serverAddress)) == SOCKET_ERROR)
	{
		logPrinter.PrintLogs("Bind socket failed!", LogLevel::Error);
		closesocket(RemoteDataServerSocket);
		return false;
	}

	// definition of remote data server address
	RemoteDataServerSocketAddressLen = sizeof(RemoteDataServerSocketAddress);
	return true;
}

bool InitSocketForResultServer()
{
	// Create Socket for result server
	RemoteResultServerSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if(RemoteResultServerSocket == SOCKET_ERROR)
	{
		logPrinter.PrintLogs(" Create remote result server scoket error!", LogLevel::Error);
		return false;
	}

	// Bind network address for result server and socket address
	sockaddr_in resultServerAddress;
	resultServerAddress.sin_family = AF_INET;
	resultServerAddress.sin_port = htons(HostPortForRemoteResultServer);
	resultServerAddress.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	if (bind(RemoteResultServerSocket, reinterpret_cast<sockaddr *>(&resultServerAddress), sizeof(resultServerAddress)) == SOCKET_ERROR)
	{
		logPrinter.PrintLogs("Bind result server socket failed!", LogLevel::Error);
		closesocket(RemoteResultServerSocket);
		return false;
	}

	// definition of remote data server address
	RemoteResultServerSocketAddressLen = sizeof(RemoteResultServerSocketAddress);
	return true;
}

bool InitNetworks()
{
	logPrinter.PrintLogs("Init Network for receive frame data from remote device!",LogLevel::Info);

	if (InitNetworkEnvironment() == false) return false;

	if (InitSocketForDataServer() == false) return false;

//	if (InitSocketForResultServer() == false) return false;

	// definition of receive data buffer
	ReveiceDataBufferlen = WIDTH * HEIGHT * BYTESIZE / 4 + 2;
	ReceiveDataBuffer = new unsigned char[ReveiceDataBufferlen];

	return true;
}

bool GetOneFrameFromNetwork(unsigned char* frameData)
{
	std::cout << "Receive one frame data from rempote device \n";
	unsigned char frameIndex;
	bool subIndex[4] = {false};
	for (auto i = 0; i < 4; ++i)
	{
		auto ret = recvfrom(RemoteDataServerSocket, reinterpret_cast<char*>(ReceiveDataBuffer), ReveiceDataBufferlen, 0, reinterpret_cast<sockaddr *>(&RemoteDataServerSocketAddress), &RemoteDataServerSocketAddressLen);
		if (ret > 10)
		{
			if (i == 0)
				frameIndex = ReceiveDataBuffer[0];
			else
			{
				if(frameIndex == ReceiveDataBuffer[0])
				{
					if (subIndex[static_cast<int>(ReceiveDataBuffer[1])] == false)
						subIndex[static_cast<int>(ReceiveDataBuffer[1])] == true;
					else
						std::cout << "Invalid data order" << std::endl;
				}
				else
				{
					std::cout << "Invalid data order" << std::endl;
				}
			}
			memcpy(frameData + i * ReveiceDataBufferlen, ReceiveDataBuffer+ 2, sizeof(unsigned char) * ReveiceDataBufferlen);
			std::cout << "Data segment " << i + 1 << "\n";
		}
		else if(ret == 10)
		{
			std::cout << "Finish receive data from client!\n";
			return false;
		}
		else
		{
			printf("send error:%d\n", WSAGetLastError());
			std::cout << "Error" << std::endl;;
		}
	}
	return true;
}

bool SendResultToRemoteServer(ResultSegment& result)
{
	std::cout << "Sending result to remote server \n";

//	auto sendStatus = sendto(RemoteResultServerSocket, reinterpret_cast<char*>(&result), sizeof(ResultSegment), 0, reinterpret_cast<sockaddr *>(&RemoteResultServerSocketAddress), RemoteResultServerSocketAddressLen);
	auto sendStatus = sendto(RemoteDataServerSocket, reinterpret_cast<char*>(&result), sizeof(ResultSegment), 0, reinterpret_cast<sockaddr *>(&RemoteDataServerSocketAddress), RemoteDataServerSocketAddressLen);
	if(sendStatus == SOCKET_ERROR)
	{
		std::cout << WSAGetLastError() << std::endl;
		return false;
	}
	return true;
}

bool DestroyNetWork()
{
	delete[] ReceiveDataBuffer;

	closesocket(RemoteDataServerSocket);
	closesocket(RemoteResultServerSocket);
	WSACleanup();
	return true;
}
