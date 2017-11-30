#pragma once
#include <winsock.h>
#include "../Models/DetectResultSegment.hpp"

bool InitNetworks();

bool GetOneFrameFromNetwork(unsigned char* frameData);

bool SendResultToRemoteServer(DetectResultSegment& result);

bool DestroyNetWork();

// Socket环境全局变量声明
extern int HostPortForRemoteDataServer;

extern int ReveiceDataBufferlen;
extern SOCKET RemoteDataServerSocket;
extern sockaddr_in RemoteDataServerSocketAddress;
extern int RemoteDataServerSocketAddressLen;
extern unsigned char* ReceiveDataBuffer;
