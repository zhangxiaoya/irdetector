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
#include "Models/ShowResultRingBuffer.hpp"

const bool IsSendResultToServer = true; // 是否发送结果到服务端

// 图像信息全局变量声明与定义
extern const unsigned int WIDTH = 320;   // 图像宽度
extern const unsigned int HEIGHT = 256;  // 图像高度
extern const unsigned int BYTESIZE = 2;  // 每个像素字节数

static const int FrameDataSize = WIDTH * HEIGHT * BYTESIZE;        // 每个图像帧数据大小
static const int ImageSize = WIDTH * HEIGHT;                       // 每一帧图像像素大小
unsigned char FrameData[FrameDataSize];                            // 每一帧图像临时缓冲
unsigned short FrameDataInprocessing[ImageSize] = {0};             // 每一帧图像临时缓冲
ResultSegment ResultItem;                                      // 每一帧图像检测结果
static const int ResultItemSize = sizeof(ResultSegment);       // 每一帧图像检测结果大小

Detector* detector = new Detector();                  // 初始化检测器

// 缓冲区全局变量声明与定义
static const int BufferSize = 10;                     // 线程同步缓冲区大小
FrameDataRingBufferStruct Buffer(FrameDataSize, BufferSize);    // 数据接收线程环形缓冲区初始化
DetectResultRingBufferStruct ResultBuffer(BufferSize);          // 结果发送线程环形缓冲区初始化
ShowResultRingBufferStruct ShowResultBuffer(BufferSize);        // 显示结果线程环形缓冲区

/****************************************************************************************/
/*                                Input Data Operation                                  */
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

	auto frameIndex = reinterpret_cast<int*>(FrameData + 2);
	std::cout<<" ====================================================================>" << *frameIndex << std::endl;

	// Reset data header pointer when to the end of buffer
	if (buffer->write_position == BufferSize)
		buffer->write_position = 0;

	// Notify Detect thread
	buffer->buffer_not_empty.notify_all();
	lock.unlock();
	return true;
}

/****************************************************************************************/
/*                              Detect target Operation                                 */
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
	CheckPerf(detector->DetectTargets(FrameDataInprocessing, &ResultItem), "Total process");

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
		memcpy(resultBuffer->item_buffer + resultBuffer->write_position * ResultItemSize, &ResultItem, ResultItemSize);
		resultBuffer->write_position++;

		// Reset data header pointer when to the end of buffer
		if (resultBuffer->write_position == BufferSize)
			resultBuffer->write_position = 0;

		// Notify Detect thread
		resultBuffer->buffer_not_empty.notify_all();
		writerLock.unlock();
	}

	// 返回一次线程执行状态
	return true;
}

/****************************************************************************************/
/*                               Send Result Operation                                  */
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

	memcpy(&ResultItem, buffer->item_buffer + buffer->read_position * ResultItemSize, ResultItemSize);
	SendResultToRemoteServer(ResultItem);
//	SendResultToRemoteServer(buffer->item_buffer[buffer->read_position]);

	buffer->read_position++;

	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	buffer->buffer_not_full.notify_all();
	lock.unlock();

	return true;
}

bool ShowResult(ShowResultRingBufferStruct* show_result_buffer)
{
//To-Do
	return true;
}


/****************************************************************************************/
/*                                  Input Data Task                                     */
/****************************************************************************************/
void InputDataTask()
{
	// 循环接收数据（线程优先级太高）
	while (true)
	{
		if(InputDataToBuffer(&Buffer) == false) break;
	}
}

/****************************************************************************************/
/*                                      Detect Task                                     */
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
/*                               Output Result Task                                     */
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
/*                                 Show Result Task                                     */
/****************************************************************************************/
void ShowResultTask()
{
	// 循环显示检测结果
	while (true)
	{
		if(ShowResult(&ShowResultBuffer) == false) break;
	}
}

/****************************************************************************************/
/*                               Initial Data Buffer                                    */
/****************************************************************************************/
void InitBuffer(FrameDataRingBufferStruct* buffer)
{
	buffer->write_position = 0;
	buffer->read_position = 0;
}

/****************************************************************************************/
/*                               Initial Result Buffer                                  */
/****************************************************************************************/
void InitResultBuffer(DetectResultRingBufferStruct* buffer)
{
	buffer->read_position = 0;
	buffer->write_position = 0;
}

/****************************************************************************************/
/*                          Initial Show Result Buffer                                  */
/****************************************************************************************/
void InitShowResultBuffer(ShowResultRingBufferStruct* show_result_buffer)
{
	show_result_buffer->write_position = 0;
	show_result_buffer->read_position = 0;
}

void RunOnNetwork()
{
	// 初始化Socket网络环境
	InitNetworks();

	// 初始化检测子局部存储和检测参数
	detector->InitSpace();
	detector->SetAllParameters();

	// 初始化数据缓冲和结果缓冲
	InitBuffer(&Buffer);
	InitResultBuffer(&ResultBuffer);
	InitShowResultBuffer(&ShowResultBuffer);

	// 创建三个线程：读取数据线程、计算结果、返回结果
	std::thread InputDataThread(InputDataTask);
	std::thread DetectorThread(DetectTask);
	std::thread OutputDataThread(OutputDataTask);
	std::thread ShowResultThread(ShowResultTask);

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
	auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
		RunOnNetwork();

//		CheckConrrectness();

//		CheckPerformance();
	}

	// 销毁检测子
	delete detector;

	// 释放CUDA设备
	CUDAInit::cudaDeviceRelease();

	// 系统暂停
	system("Pause");
	return 0;
}
