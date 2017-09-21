#pragma once
#include <WinSock2.h>
#include <winsock.h>

class DataReceiver
{
public:
	static bool InitNetworks();

	static void Run();

	static bool DestroyNetWork();

	static int hostPort;
	static int width;
	static int height;

	static int reveiceDataBufferlen;

	static SOCKET serverSocket;
	static sockaddr_in remoteAddress;
	static int remoteAddressLen;

	static char* receiveDataBuffer;
};
