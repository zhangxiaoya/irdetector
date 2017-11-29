#include "NetworkTransfer.h"
#include "../Monitor/Filter.hpp"
#include "../Models/DetectResultSegment.hpp"

#pragma comment(lib, "ws2_32.lib")

// Definition of all variables used in network
int HostPortForRemoteDataServer = 8889; // 接收数据端口
int HostPortForRemoteResultServer = 8889; // 发送结果端口
char RemoteResultServerHostIP[] = "192.168.2.111"; // 发送结果主机地址
SOCKET RemoteDataServerSocket = 0; // 接收数据SOCKET
SOCKET RemoteResultServerSocket = 0; // 发送结果SOCKET
sockaddr_in RemoteDataServerSocketAddress; // 接收数据Socket地址
sockaddr_in RemoteResultServerSocketAddress; // 发送结果Socket地址
int RemoteDataServerSocketAddressLen = 0; // 接收数据Socket地址长度
int RemoteResultServerSocketAddressLen = 0; // 发送结果Socket地址长度

auto SocketLen = 500 * 1024 * 1024; // Socket缓冲区大小

int ReveiceDataBufferlen = 0; // 接收数据缓冲区大小
unsigned char* ReceiveDataBuffer; // 接收

const int packageCount = 4;

/************************************************************************/
/*                            Network Initial                           */
/************************************************************************/
bool InitNetworkEnvironment()
{
	WSADATA wsaData; // 记录网络环境初始化结果
	auto sockVersion = MAKEWORD(2, 2); // Socket版本号
	if (WSAStartup(sockVersion, &wsaData) != 0) // 初始化Socket网络编程环境
	{
		// 若初始化网络编程环境出错，输出错误信息
		logPrinter.PrintLogs("Init network failed!", Error);
		return false;
	}
	return true;
}

/************************************************************************/
/*                          Data Server Initial                         */
/************************************************************************/
bool InitSocketForDataServer()
{
	// 创建Socket，用于接收图像数据
	RemoteDataServerSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	// 如果Socket创建失败，打印错误信息
	if (RemoteDataServerSocket == SOCKET_ERROR)
	{
		logPrinter.PrintLogs(" Create remote data server scoket error!", Error);
		return false;
	}

	// 初始化Socket地址和协议族等信息
	RemoteDataServerSocketAddress.sin_family = AF_INET; // 协议族
	RemoteDataServerSocketAddress.sin_port = htons(HostPortForRemoteDataServer); // 端口号
	RemoteDataServerSocketAddress.sin_addr.S_un.S_addr = htonl(INADDR_ANY); // 网络地址

	//设置socket缓冲区大小
	setsockopt(RemoteDataServerSocket, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&SocketLen), sizeof(SocketLen));

	// Socket地址长度
	RemoteDataServerSocketAddressLen = sizeof(RemoteDataServerSocketAddress);

	// 绑定Socket地址和Socket，用于接收数据
	if (bind(RemoteDataServerSocket, reinterpret_cast<sockaddr *>(&RemoteDataServerSocketAddress), RemoteDataServerSocketAddressLen) == SOCKET_ERROR)
	{
		logPrinter.PrintLogs("Bind socket failed!", Error);
		closesocket(RemoteDataServerSocket);
		return false;
	}

	return true;
}

/************************************************************************/
/*                          Result Server Initial                       */
/************************************************************************/
bool InitSocketForResultServer()
{
	// 创建Socket，用于发送结果数据
	RemoteResultServerSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	// 如果创建失败，打印错误信息
	if (RemoteResultServerSocket == SOCKET_ERROR)
	{
		logPrinter.PrintLogs(" Create remote result server scoket error!", Error);
		return false;
	}

	// 初始化Socket地址和协议族等信息
	RemoteResultServerSocketAddress.sin_family = AF_INET; // 协议族
	RemoteResultServerSocketAddress.sin_port = htons(HostPortForRemoteResultServer); // 端口号
	RemoteResultServerSocketAddress.sin_addr.S_un.S_addr = inet_addr(RemoteResultServerHostIP); // 网络地址

	// Socket地址长度
	RemoteResultServerSocketAddressLen = sizeof(RemoteResultServerSocketAddress);
	return true;
}

/************************************************************************/
/*                        Network Version Initial                       */
/************************************************************************/
bool InitNetworks()
{
	// 打印日志信息
	logPrinter.PrintLogs("Init Network for receive frame data from remote device!", Info);

	// 初始化Socket网络环境
	if (InitNetworkEnvironment() == false) return false;

	// 初始化接收数据Socket参数
	if (InitSocketForDataServer() == false) return false;

	// 初始化发送结果Socket参数
	if (InitSocketForResultServer() == false) return false;

	// 定义缓冲区长度
	ReveiceDataBufferlen = WIDTH * HEIGHT * BYTESIZE + 2 * packageCount;
	// 申请缓冲区
	ReceiveDataBuffer = new unsigned char[ReveiceDataBufferlen];

	return true;
}

/************************************************************************/
/*                        Received One Frame Data                       */
/************************************************************************/
bool GetOneFrameFromNetwork(unsigned char* frameData)
{
	//	打印开始接收数据消息
//	printf("Receiving one frame data from remote device ...\n");

	// 记录当前帧的帧号
	unsigned char frameIndex = 0;
	// 记录每一帧每一段的是否已经接收
	bool subIndex[packageCount] = {false};

	// 每段数据长度
	auto quarterBufferSize = ReveiceDataBufferlen / packageCount;

	// 循环接收多次（分包数量）
	for (auto i = 0; i < packageCount; ++i)
	{
		auto partBuffer = ReceiveDataBuffer + i * quarterBufferSize;
		// 接收一次数据
		memset(partBuffer, 0, sizeof(partBuffer));
		const auto receivedStatus = recvfrom(
			RemoteDataServerSocket,
			reinterpret_cast<char*>(partBuffer),
			quarterBufferSize,
			0,
			reinterpret_cast<sockaddr *>(&RemoteDataServerSocketAddress),
			&RemoteDataServerSocketAddressLen);

		// 如果不是数据发送方发送的结束消息，那么就是真实数据，开始处理数据
		if (receivedStatus > 10)
		{
			// 第一个数据段，记录下当前的帧号（每段数据第一个字节表示帧号，第二个字节表示段号，都是无符号字符型）
			if (i == 0)
				frameIndex = static_cast<unsigned char>(partBuffer[0]);
			else
			{
				// 其他数据段，检测帧号是否与当前正在接收的帧号一致
				if (frameIndex == static_cast<unsigned char>(partBuffer[0]))
				{
					// 如果帧号一致，检测数据段是不是已经接收过了，如果没有接收过，修改对应的标志
					if (subIndex[static_cast<int>(partBuffer[1])] == false)
						subIndex[static_cast<int>(partBuffer[1])] = true;
					else // 如果已经接收到了，输出错误信息
					{
						printf("Invalid data order, duplicated segment!");
						printf("Segment %d had received more than once!", static_cast<int>(partBuffer[1]));
						i--; // 多接收一个数据段
					}
				}
				else // 如果帧号不一致，输出错误信息
				{
					printf("Invalid frame order： \nExpected frame index is %d , \nbut actualy index is %d ", static_cast<int>(frameIndex), static_cast<int>(partBuffer[0]));
					printf("Resetting .....>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
					frameIndex = static_cast<unsigned char>(partBuffer[0]);
					for (auto idx = 0; idx < packageCount; ++idx)
						subIndex[idx] = false;
					i = 0;
				}
			}
			// 将除去帧号和段号的数据部分复制到图像帧数据对应的位置
			memcpy(frameData + i * (quarterBufferSize - 2), partBuffer + 2, sizeof(unsigned char) * (quarterBufferSize - 2));
			// 并输出当前接收到的帧号和段号
		}
		else if (receivedStatus == 10) // 长度为10的任意数据表示发送结束，输出提示信息，并返回false
		{
			printf("Finish receive data from client!\n");
			return false;
		}
		else // 如果既不是数据部分，也不是数据结束符，输出错误代码
		{
			printf("Received data error:%d\n", WSAGetLastError());
			return false;
		}
	}
//	printf("Frame index is %d\n", static_cast<int>(frameIndex));
	return true;
}

/************************************************************************/
/*                       Send One Result to Remote                      */
/************************************************************************/
bool SendResultToRemoteServer(DetectResultSegment& result)
{
	// 打印日志消息
//	printf("Sending result to remote server \n");

	// 发送结果数据到远端服务器
	auto sendStatus = sendto(
		RemoteResultServerSocket,
		reinterpret_cast<char*>(&result),
		sizeof(DetectResultSegment),
		0,
		reinterpret_cast<sockaddr *>(&RemoteResultServerSocketAddress),
		RemoteResultServerSocketAddressLen);

	// 如果出错，打印消息代码
	if (sendStatus == SOCKET_ERROR)
	{
		printf("%d\n", WSAGetLastError());
		return false;
	}
	return true;
}

/************************************************************************/
/*                             Clean Environment                        */
/************************************************************************/
bool DestroyNetWork()
{
	delete[] ReceiveDataBuffer; // 销毁临时缓冲区

	closesocket(RemoteDataServerSocket); // 关闭结束数据的Socket
	closesocket(RemoteResultServerSocket); // 关闭发送结果数据的Socket
	WSACleanup(); // 清理网络环境
	return true;
}
