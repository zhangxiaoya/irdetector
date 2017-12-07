// 接收数据端口号
#ifndef HOSTPORT_OF_REMOTE_DATA_SERVER
#define HOSTPORT_OF_REMOTE_DATA_SERVER (8889)
#endif

// 发送结果端口号
#ifndef HOSTPORT_OF_REMOTE_RESULT_SERVER
#define HOSTPORT_OF_REMOTE_RESULT_SERVER (8889)
#endif

// Socket 缓冲区大小
#ifndef SOCKET_BUFFER_LENGTH
#define SOCKET_BUFFER_LENGTH (500 * 1024 * 1024)
#endif

// 是否进行大小端变换
#ifndef NEED_CHANGE_LITTEL_ENDIAN_AND_BIG_ENDIAN
#define NEED_CHANGE_LITTEL_ENDIAN_AND_BIG_ENDIAN true
#endif

// 大小端转换宏
#ifndef LittleEndianAndBigEndianChange
#define LittleEndianAndBigEndianChange(call)              \
{                                                         \
	if(NEED_CHANGE_LITTEL_ENDIAN_AND_BIG_ENDIAN == true)  \
	{                                                     \
		call;                                             \
	}                                                     \
}
#endif