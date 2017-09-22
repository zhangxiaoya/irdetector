#include "Validation/Validation.hpp"
#include "Init/Init.hpp"
#include "Validation/DetectorValidation.hpp"
#include "Network/DataReceiver.h"

#include <windows.h>
#include <iostream>
#include <mutex>
#include <thread>
#include "Models/RingBufferStruct.hpp"

const unsigned int WIDTH = 320;
const unsigned int HEIGHT = 256;
const unsigned BYTESIZE = 1;

static const int BufferSize = 10;
static const int FrameSize = WIDTH * HEIGHT * BYTESIZE;

unsigned char FrameData[FrameSize];
unsigned char FrameDataInprocessing[FrameSize] = {0};

Detector* detector = new Detector();

RingBufferStruct Buffer(FrameSize, BufferSize);

void ProduceItem(RingBufferStruct* buffer)
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

void ConsumeItem(RingBufferStruct* buffer)
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
	while (true)
	{
		std::cout << "Receivce data from network" << std::endl;
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

void InitBuffer(RingBufferStruct* buffer)
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

		// Init ring buffer
		InitBuffer(&Buffer);

		std::thread producer(ProducerTask);
		std::thread consumer(ConsumerTask);

		producer.join();
		consumer.join();

		DestroyNetWork();

		//		Validation validation;
		//		validation.InitValidationData("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
		//		validation.VailidationAll();

		//		DetectorValidation visualEffectValidator;
		//		visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
		//		visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_2.bin");
		//		visualEffectValidator.VailidationAll();
	}
	// Release Cuda device
	CUDAInit::cudaDeviceRelease();

	system("Pause");
	return 0;
}
