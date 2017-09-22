#include "Validation/Validation.hpp"
#include "Init/Init.hpp"
#include "Validation/DetectorValidation.hpp"
#include "Network/DataReceiver.h"

#include <windows.h>
#include <iostream>
#include <mutex>
#include <thread>

const unsigned int WIDTH = 320;
const unsigned int HEIGHT = 256;
const unsigned BYTESIZE = 1;

static const int BufferSize = 10;
static const int FrameSize = WIDTH * HEIGHT * BYTESIZE;

unsigned char FrameData[FrameSize];
unsigned char FrameDataInprocessing[FrameSize] = { 0 };

Detector* detector = new Detector();

//HANDLE dataMutex;
//
//DWORD WINAPI Detect(LPVOID lpParam)
//{
//	WaitForSingleObject(dataMutex, INFINITE);
//	CheckPerf(detector->DetectTargets(FrameData), "whole");
//	ReleaseMutex(dataMutex);
//	return 0;
//}
//
//DWORD WINAPI ReadData(LPVOID lpParam)
//{
//	WaitForSingleObject(dataMutex, INFINITE);
//	GetOneFrameFromNetwork(FrameData);
//	ReleaseMutex(dataMutex);
//
//	return 0;
//}

//static const int TotalCountOfProduction = 10;

struct BufferStruct
{
	unsigned char item_buffer[BufferSize * FrameSize]; // 环形缓冲
	size_t read_position;
	size_t write_position;
	std::mutex bufferMutex;
	std::condition_variable buffer_not_full; // 条件变量, 指示产品缓冲区不为满.
	std::condition_variable buffer_not_empty; // 条件变量, 指示产品缓冲区不为空.
};

BufferStruct Buffer;


void ProduceItem(BufferStruct* buffer)
{
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while ((buffer->write_position + 1) % BufferSize == buffer->read_position)
	{
		std::cout << "Producer is waiting for an empty slot...\n";
		buffer->buffer_not_full.wait(lock);
	}

	GetOneFrameFromNetwork(FrameData);
	memcpy(buffer->item_buffer + buffer->write_position * FrameSize, FrameData, FrameSize * sizeof(unsigned char));

	buffer->write_position++;

	if (buffer->write_position == BufferSize)
		buffer->write_position = 0;

	buffer->buffer_not_empty.notify_all();
	lock.unlock();
}

void ConsumeItem(BufferStruct* buffer)
{
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		std::cout << "Consumer is waiting for items...\n";
		buffer->buffer_not_empty.wait(lock);
	}

	memcpy(FrameDataInprocessing, buffer->item_buffer + buffer->read_position * FrameSize, FrameSize);

	buffer->read_position++;

	if (buffer->read_position >= BufferSize)
		buffer->read_position = 0;

	buffer->buffer_not_full.notify_all();
	lock.unlock();

	CheckPerf(detector->DetectTargets(FrameDataInprocessing), "whole");
}


void ProducerTask()
{
	while(true)
	{
		std::cout << "Receivce data from network" <<std::endl;
		ProduceItem(&Buffer);
		Sleep(1);
	}
}

void ConsumerTask()
{
	while (true)
	{
		ConsumeItem(&Buffer);
		std::cout << "Detect target" << std::endl;
	}
}

void InitBuffer(BufferStruct* buffer)
{
	buffer->write_position = 0;
	buffer->read_position = 0;
}


int main(int argc, char* argv[])
{
	// Init CUDA
	auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
		// Init Network socket
		InitNetworks();

		// Init Detector Space
		detector->InitSpace();

//		while (true)
//		{
			InitBuffer(&Buffer);

			std::thread producer(ProducerTask);
			std::thread consumer(ConsumerTask);

			producer.join();
			consumer.join();

//		}
//		// Create Mutex
//		dataMutex = CreateMutex(nullptr, FALSE, nullptr);
//
//		// Init Network socket
//		InitNetworks();
//
//		// Init Detector Space
//		detector->InitSpace();
//
//		HANDLE hThread1;
//		HANDLE hThread2;
//		while (true)
//		{
//			hThread1 = CreateThread(nullptr,
//			                        0,
//			                        ReadData,
//			                        nullptr,
//			                        0,
//			                        nullptr);
//			hThread2 = CreateThread(nullptr,
//			                        0,
//			                        Detect,
//			                        nullptr,
//			                        0,
//			                        nullptr);
//		}
//
//		CloseHandle(hThread1);
//		CloseHandle(hThread2);
//		DestroyNetWork();
//
//
////		Validation validation;
////		validation.InitValidationData("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
////		validation.VailidationAll();

//		DetectorValidation visualEffectValidator;
//		visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
//		visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_2.bin");
//		visualEffectValidator.VailidationAll();
	}
	CUDAInit::cudaDeviceRelease();

	system("Pause");
	return 0;
}
