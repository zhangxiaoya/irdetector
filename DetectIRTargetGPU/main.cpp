#include "Validation/CorrectnessValidation.hpp"
#include "Init/Init.hpp"
#include "Validation/PerformanceValidation.hpp"
#include "Network/NetworkTransfer.h"

#include <windows.h>
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
extern const unsigned int WIDTH = 320 * 2;   // 图像宽度
extern const unsigned int HEIGHT = 256 * 2;  // 图像高度
extern const unsigned int BYTESIZE = 2;  // 每个像素字节数

static const int FrameDataSize = WIDTH * HEIGHT * BYTESIZE;        // 每个图像帧数据大小
static const int ImageSize = WIDTH * HEIGHT;                       // 每一帧图像像素大小

/****************************************************************************************/
/* 参数定义：预处理阶段参数                                                               */
/****************************************************************************************/
const int DilationRadius = 1;            // 滤波器半径
const int DiscretizationScale = 15;      // 离散化尺度

/****************************************************************************************/
/* 其他参数定义                                                                          */
/****************************************************************************************/
unsigned char FrameData[FrameDataSize];                            // 每一帧图像临时缓冲
unsigned short FrameDataInprocessing[ImageSize] = {0};             // 每一帧图像临时缓冲
unsigned short FrameDataToShow[ImageSize] = {0};                   // 每一帧显示结果图像临时缓冲
DetectResultSegment ResultItemSendToServer;                              // 每一帧图像检测结果
static const int ResultItemSize = sizeof(DetectResultSegment);           // 每一帧图像检测结果大小
cv::Mat CVFrame(HEIGHT, WIDTH, CV_8UC1);

/****************************************************************************************/
/* 检测器定义                                                                          */
/****************************************************************************************/
Detector* detector = new Detector(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);  // 初始化检测器
Monitor* monitor = new Monitor(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);     // 初始化Monitor

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
	// 检查环形缓冲区是否已经满了，如果满了，则解锁，让消费缓冲区内容的线程读取换冲区内的数据
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while ((buffer->write_position + 1) % BufferSize == buffer->read_position)
	{
		// 可以删除打印日志，降低没必要的时间消耗
		// printf("Image Data Producer is waiting for an empty slot...\n");
		buffer->buffer_not_full.wait(lock);
	}

	// 从网络获取一帧图像数据，若获取的结果是结束标志（10个字节的任意数据），修改缓冲结束标志为true，终止接受网络数据线程
	// Check finish stream end flag
	if (GetOneFrameFromNetwork(FrameData) == false)
	{
		buffer->finish_flag = true;
		return false;
	}

	// 把接受到的数据放在缓冲区，并且修改缓冲队列指针
	// Copy data received from network to ring buffer and update ring buffer header pointer
	memcpy(buffer->item_buffer + buffer->write_position * FrameDataSize, FrameData, FrameDataSize);
	buffer->write_position++;

	// 环形缓冲指针判断，防止指针访问越界
	// Reset data header pointer when to the end of buffer
	if (buffer->write_position == BufferSize)
		buffer->write_position = 0;

	// 通知其他消费线程，已经有数据存在缓冲区，并解锁缓冲区
	// Notify Detect thread
	buffer->buffer_not_empty.notify_all();
	lock.unlock();
	return true;
}

/****************************************************************************************/
/* 函数定义：显示结果（测试用）                                                           */
/****************************************************************************************/
void ShowLastResult(int shouLastResultDelay)
{
	ShowFrame::ToMat(FrameDataInprocessing, WIDTH, HEIGHT, CVFrame);
	ShowFrame::DrawRectangles(CVFrame, &ResultItemSendToServer);
	cv::imshow("Result", CVFrame);
	cv::waitKey(shouLastResultDelay);
}

/****************************************************************************************/
/* 函数定义：检测一帧数据，并且把结果放在缓冲区                                             */
/****************************************************************************************/
bool DetectTarget(FrameDataRingBufferStruct* buffer, DetectResultRingBufferStruct* resultBuffer)
{
	// Check buffer is empty or not, if empty automatic unlock mutex
	// 检查缓冲区是否为空，如果为空，则解锁缓冲区
	std::unique_lock<std::mutex> readLock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		// 判断是缓冲结束标志，若是缓冲区结束表示为true，则结束线程
		if (buffer->finish_flag == true)
		{
			resultBuffer->finish_flag = true;
			return false;
		}
		// 注释打印日志，避免不必要的时间开销
		// printf("Detector Consumer is waiting for items...\n");
		buffer->buffer_not_empty.wait(readLock);
	}

	// 从缓冲区中取出需要处理的数据，并存储在临时图像存储区中
	memcpy(FrameDataInprocessing, buffer->item_buffer + buffer->read_position * FrameDataSize, FrameDataSize);

	// 修改缓冲区标志
	buffer->read_position++;

	// 
	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	// 环形缓冲指针判断，防止指针访问越界
	buffer->buffer_not_full.notify_all();
	readLock.unlock();

	// 检测目标，并检测性能
	CheckPerf(detector->DetectTargets(FrameDataInprocessing, &ResultItemSendToServer), "Total process");
	// 检测并跟踪目标，检测整个过程的时间系能
	// CheckPerf(monitor->Process(FrameDataInprocessing, &ResultItemSendToServer), "Total Tracking Process");

	// 并发存储检测结果到缓冲区
	std::unique_lock<std::mutex> writerLock(resultBuffer->bufferMutex);
	while ((resultBuffer->write_position + 1) % BufferSize == resultBuffer->read_position)
	{
		// 注释打印日志，避免不不要的时间开销
		// printf("Result Send Producer is waiting for an empty slot...\n");
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

	// 临时显示结果
	auto shouLastResultDelay = 1;
	ShowLastResult(shouLastResultDelay);

	// 返回一次线程执行状态
	return true;
}

/****************************************************************************************/
/* 函数定义：发送一帧检测结果                                                             */
/****************************************************************************************/
bool OutputData(DetectResultRingBufferStruct* buffer)
{
	// 检查数据结果缓冲区
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		if (buffer->finish_flag == true)
			return false;

//		printf("Result send thread is waiting for result items...\n");
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
		RunOnNetwork();

//		CheckConrrectness(WIDTH, HEIGHT);

//		CheckPerformance(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);

//		CheckTracking(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);

//		CheckSearching(WIDTH, HEIGHT, DilationRadius, DiscretizationScale);
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
