#include "Validation/CorrectnessValidation.hpp"
#include "Init/Init.hpp"
#include "Validation/PerformanceValidation.hpp"
#include "Network/NetworkTransfer.h"

#include <windows.h>
#include <iostream>
#include <mutex>
#include <thread>
#include "Models/FrameDataRingBufferStruct.hpp"
#include "Models/DetectResultRingBufferStruct.hpp"
#include "Validation/Validation.h"
#include "Monitor/Monitor.hpp"

const bool IsSendResultToServer = true; // 是否发送结果到服务端(测试用)

/****************************************************************************************/
/* 参数定义： 图像信息全局变量声明与定义                                                   */
/****************************************************************************************/
extern const unsigned int WIDTH = 320;   // 图像宽度
extern const unsigned int HEIGHT = 256;  // 图像高度
extern const unsigned int BYTESIZE = 2;  // 每个像素字节数

/****************************************************************************************/
/* 参数定义：预处理阶段参数                                                               */
/****************************************************************************************/
const int DilationRadius = 1;            // 滤波器半径
const int DiscretizationScale = 15;      // 离散化尺度

/****************************************************************************************/
/* 其他参数定义                                                                          */
/****************************************************************************************/
static const int FrameDataSize = WIDTH * HEIGHT * BYTESIZE;        // 每个图像帧数据大小
static const int ImageSize = WIDTH * HEIGHT;                       // 每一帧图像像素大小
unsigned char FrameData[FrameDataSize];                            // 每一帧图像临时缓冲
unsigned short FrameDataInprocessing[ImageSize] = {0};             // 每一帧图像临时缓冲
unsigned short FrameDataToShow[ImageSize] = {0};                   // 每一帧显示结果图像临时缓冲
DetectResultSegment ResultItemSendToServer;                              // 每一帧图像检测结果
DetectResultSegment ResultItemToShow;                                    // 每一帧图像显示结果
static const int ResultItemSize = sizeof(DetectResultSegment);           // 每一帧图像检测结果大小
Detector* detector = new Detector(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);  // 初始化检测器
Monitor* monitor = new Monitor(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);     // init monitor
cv::Mat CVFrame(HEIGHT, WIDTH, CV_8UC1);

/****************************************************************************************/
/* 参数定义：缓冲区全局变量声明与定义                                                      */
/****************************************************************************************/
static const int BufferSize = 10;                                              // 线程同步缓冲区大小
FrameDataRingBufferStruct Buffer(FrameDataSize, BufferSize);                   // 数据接收线程环形缓冲区初始化
DetectResultRingBufferStruct ResultBuffer(WIDTH, HEIGHT, BufferSize);          // 结果发送线程环形缓冲区初始化

/****************************************************************************************/
/* 函数定义：从网络读取数据操作（读取一帧）                                                 */
/****************************************************************************************/
bool InputDataToBuffer(FrameDataRingBufferStruct* buffer)
{
	// Check buffer is full or not, if full automatic unlock mutex
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while ((buffer->write_position + 1) % BufferSize == buffer->read_position)
	{
		std::cout << "Image Data Producer is waiting for an empty slot...\n";
		buffer->buffer_not_full.wait(lock);
	}

	// Check finish stream end flag
	if (GetOneFrameFromNetwork(FrameData) == false)
	{
		buffer->finish_flag = true;
		return false;
	}

	// Copy data received from network to ring buffer and update ring buffer header pointer
	memcpy(buffer->item_buffer + buffer->write_position * FrameDataSize, FrameData, FrameDataSize);
	buffer->write_position++;

	// Reset data header pointer when to the end of buffer
	if (buffer->write_position == BufferSize)
		buffer->write_position = 0;

	// Notify Detect thread
	buffer->buffer_not_empty.notify_all();
	lock.unlock();
	return true;
}

/****************************************************************************************/
/* 函数定义：检测一帧数据，并且把结果放在缓冲区                                             */
/****************************************************************************************/
bool DetectTarget(FrameDataRingBufferStruct* buffer, DetectResultRingBufferStruct* resultBuffer)
{
	// 并发读取图像数据
	std::unique_lock<std::mutex> readLock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		if (buffer->finish_flag == true)
		{
			resultBuffer->finish_flag = true;
			return false;
		}

		std::cout << "Detector Consumer is waiting for items...\n";
		buffer->buffer_not_empty.wait(readLock);
	}

	memcpy(FrameDataInprocessing, buffer->item_buffer + buffer->read_position * FrameDataSize, FrameDataSize);

	buffer->read_position++;

	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	buffer->buffer_not_full.notify_all();
	readLock.unlock();

	// 检测目标，并检测性能
	CheckPerf(detector->DetectTargets(FrameDataInprocessing, &ResultItemSendToServer), "Total process");
//	CheckPerf(monitor->Process(FrameDataInprocessing, &ResultItemSendToServer), "Total Tracking Process");

//	LARGE_INTEGER t1, t2, tc;
//	QueryPerformanceFrequency(&tc);
//	QueryPerformanceCounter(&t1);
//	detector->DetectTargets(FrameDataInprocessing, &ResultItemSendToServer);
//	QueryPerformanceCounter(&t2);
//	const auto timeC = (t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart;
//	printf("Operation of %20s Use Time:%f\n", "Total process", timeC);
//	if (timeC >= 0.006)
//		printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

	if(IsSendResultToServer)
	{
		// 并发存储检测结果到缓冲区
		std::unique_lock<std::mutex> writerLock(resultBuffer->bufferMutex);
		while ((resultBuffer->write_position + 1) % BufferSize == resultBuffer->read_position)
		{
			std::cout << "Result Send Producer is waiting for an empty slot...\n";
			resultBuffer->buffer_not_full.wait(writerLock);
		}

		// Copy data received from network to ring buffer and update ring buffer header pointer
		memcpy(resultBuffer->item_buffer + resultBuffer->write_position * ResultItemSize, &ResultItemSendToServer, ResultItemSize);
		resultBuffer->write_position++;

		// Reset data header pointer when to the end of buffer
		if (resultBuffer->write_position == BufferSize)
			resultBuffer->write_position = 0;

		// Notify Detect thread
		resultBuffer->buffer_not_empty.notify_all();
		writerLock.unlock();
	}

	// 临时显示结果
	ShowFrame::ToMat(FrameDataInprocessing, WIDTH, HEIGHT, CVFrame);
	ShowFrame::DrawRectangles(CVFrame, &ResultItemSendToServer);
	cv::imshow("Result", CVFrame);
	cv::waitKey(1);

	// 返回一次线程执行状态
	return true;
}

/****************************************************************************************/
/* 函数定义：发送一帧检测结果                                                             */
/****************************************************************************************/
bool OutputData(DetectResultRingBufferStruct* buffer)
{
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		if (buffer->finish_flag == true)
			return false;

		std::cout << "Result send thread is waiting for result items...\n";
		buffer->buffer_not_empty.wait(lock);
	}

	memcpy(&ResultItemSendToServer, buffer->item_buffer + buffer->read_position * ResultItemSize, ResultItemSize);
	SendResultToRemoteServer(ResultItemSendToServer);

	buffer->read_position++;

	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	buffer->buffer_not_full.notify_all();
	lock.unlock();

	return true;
}

/****************************************************************************************/
/* 函数定义：读取图像帧线程任务                                                           */
/****************************************************************************************/
void InputDataTask()
{
	// 循环接收数据
	while (true)
	{
		if(InputDataToBuffer(&Buffer) == false) break;
	}
}

/****************************************************************************************/
/* 函数定义：检测目标线程任务                                                             */
/****************************************************************************************/
void DetectTask()
{
	// 循环读取图像，检测目标
	while (true)
	{
		if (DetectTarget(&Buffer, &ResultBuffer) == false) break;
	}
}

/****************************************************************************************/
/* 函数定义：发送结果线程任务                                                             */
/****************************************************************************************/
void OutputDataTask()
{
	// 循环发送检测结果
	while (true)
	{
		if (OutputData(&ResultBuffer) == false) break;
	}
}

/****************************************************************************************/
/* 函数定义：初始化接受数据缓冲                                                            */
/****************************************************************************************/
void InitDataSourceBuffer(FrameDataRingBufferStruct* buffer)
{
	buffer->write_position = 0;
	buffer->read_position = 0;
}

/****************************************************************************************/
/* 函数定义：初始化检测结果缓冲                                                           */
/****************************************************************************************/
void InitResultBuffer(DetectResultRingBufferStruct* buffer)
{
	buffer->read_position = 0;
	buffer->write_position = 0;
}

/****************************************************************************************/
/* 函数定义：在网络环境运行                                                               */
/****************************************************************************************/
void RunOnNetwork()
{
	// 初始化Socket网络环境
	InitNetworks();

	// 初始化检测子局部存储和检测参数
	detector->InitSpace();
	detector->SetRemoveFalseAlarmParameters(true, false, false, false, true, true);

	// 初始化数据缓冲和结果缓冲
	InitDataSourceBuffer(&Buffer);
	InitResultBuffer(&ResultBuffer);

	// 创建三个线程：读取数据线程、计算结果、返回结果
	std::thread InputDataThread(InputDataTask);
	std::thread DetectorThread(DetectTask);
	std::thread OutputDataThread(OutputDataTask);

	// 三个线程开始运行
	InputDataThread.join();
	DetectorThread.join();
	OutputDataThread.join();

	// 销毁网络
	DestroyNetWork();
}

/****************************************************************************************/
/*                                     Main Function                                    */
/****************************************************************************************/
int main(int argc, char* argv[])
{
	// 初始化CUDA设备
	const auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
//		RunOnNetwork();

//		CheckConrrectness(WIDTH, HEIGHT);

//		CheckPerformance(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);

//		CheckTracking(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);

		CheckSearching(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);
	}

	// 销毁检测子
	delete detector;
	delete monitor;

	// 释放CUDA设备
	CUDAInit::cudaDeviceRelease();

	// 系统暂停
	system("Pause");
	return 0;
}
