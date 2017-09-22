#pragma once
#include <winsock.h>

bool InitNetworks();

void GetOneFrameFromNetwork(unsigned char* frameData);

bool DestroyNetWork();

extern int hostPort;
extern int width;
extern int height;
extern int reveiceDataBufferlen;
extern SOCKET serverSocket;
extern sockaddr_in remoteAddress;
extern int remoteAddressLen;
extern char* receiveDataBuffer;
