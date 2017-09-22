#pragma once
#include <winsock.h>

bool InitNetworks();

void GetOneFrameFromNetwork(unsigned char* frameData);

bool DestroyNetWork();

// Declare of all vriable used in network
extern int HostPort;

extern const unsigned int WIDTH;
extern const unsigned int HEIGHT;
extern const unsigned int BYTESIZE;

extern int ReveiceDataBufferlen;
extern SOCKET ServerSocket;
extern sockaddr_in RemoteAddress;
extern int RemoteAddressLen;
extern unsigned char* ReceiveDataBuffer;
