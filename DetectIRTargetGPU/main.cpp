#include "Validation/Validation.hpp"
#include "Init/Init.hpp"
#include "Validation/DetectorValidation.hpp"
#include "Network/DataReceiver.h"

#include <windows.h>
#include <iostream>
#include <mutex>
#include <thread>
#include "Models/RingBufferStruct.hpp"
#include "Models/ResultBufferStruct.hpp"

const bool IsSendResultToServer = false; // 是否发送结果到服务端

// 图像信息全局变量声明与定义
extern const unsigned int WIDTH = 320;   // 图像宽度
extern const unsigned int HEIGHT = 256;  // 图像高度
extern const unsigned int BYTESIZE = 2;  // 每个像素字节数

static const int FrameSize = WIDTH * HEIGHT * BYTESIZE;        // 每个图像帧的大小
unsigned char FrameData[FrameSize];                            // 每一帧图像临时缓冲
unsigned char FrameDataInprocessing[FrameSize] = {0};          // 每一帧图像临时缓冲
ResultSegment ResultItem;                                      // 每一帧图像检测结果
static const int ResultItemSize = sizeof(ResultSegment);       // 每一帧图像检测结果大小

Detector* detector = new Detector();                  // 初始化检测器

// 缓冲区全局变量声明与定义
static const int BufferSize = 10;                     // 线程同步缓冲区大小
RingBufferStruct Buffer(FrameSize, BufferSize);       // 数据接收线程环形缓冲区初始化
ResultBufferStruct ResultBuffer(BufferSize);          // 结果发送线程环形缓冲区初始化

/****************************************************************************************/
/*                                Input Data Operation                                  */
/****************************************************************************************/
bool InputDataToBuffer(RingBufferStruct* buffer)
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
	memcpy(buffer->item_buffer + buffer->write_position * FrameSize, FrameData, FrameSize * sizeof(unsigned char));
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
bool DetectTarget(RingBufferStruct* buffer, ResultBufferStruct* resultBuffer)
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

	memcpy(FrameDataInprocessing, buffer->item_buffer + buffer->read_position * FrameSize, FrameSize);

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
bool OutputData(ResultBufferStruct* buffer)
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

/****************************************************************************************/
/*                                  Input Data Task                                     */
/****************************************************************************************/
void InputDataTask()
{
	// 循环接收数据（线程优先级太高）
	while (true)
	{
		if(InputDataToBuffer(&Buffer) == false) break;
		//Sleep(1);/////???????????????????????????????????????????
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
/*                               Initial Data Buffer                                    */
/****************************************************************************************/
void InitBuffer(RingBufferStruct* buffer)
{
	buffer->write_position = 0;
	buffer->read_position = 0;
}

/****************************************************************************************/
/*                               Initial Result Buffer                                  */
/****************************************************************************************/
void InitResultBuffer(ResultBufferStruct* buffer)
{
	buffer->read_position = 0;
	buffer->write_position = 0;
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

	// 创建三个线程：读取数据线程、计算结果、返回结果
	std::thread InputDataThread(InputDataTask);
	std::thread DetectorThread(DetectTask);
//	std::thread OutputDataThread(OutputDataTask);

	// 三个线程开始运行
	InputDataThread.join();
	DetectorThread.join();
//	OutputDataThread.join();

	// 销毁网络
	DestroyNetWork();
}

void TestUsingBinaryFile()
{
	DetectorValidation visualEffectValidator;
	visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
//	visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1.bin");
	visualEffectValidator.VailidationAll();
}

void TestPerformance()
{
	/* 下面是CUDA核函数测试程序和单机文本文件测试，开发阶段不要删除，方便调试 */
	Validation validation;
	validation.InitValidationData("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
	validation.VailidationAll();
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
//		RunOnNetwork();

		TestPerformance();

//		TestUsingBinaryFile();
	}

	// 销毁检测子
	delete detector;

	// 释放CUDA设备
	CUDAInit::cudaDeviceRelease();

	system("Pause");
	return 0;
}
