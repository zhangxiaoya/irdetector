#include "Validation/Validation.hpp"
#include "Init/Init.hpp"
#include "Validation/DetectorValidation.hpp"
#include "Network/DataReceiver.h"

#include <windows.h>
#include <iostream>
#include <mutex>
#include <thread>
#include "Models/RingBufferStruct.hpp"

// Definition of all const varibales
extern const unsigned int WIDTH = 320;
extern const unsigned int HEIGHT = 256;
extern const unsigned int BYTESIZE = 1;

static const int BufferSize = 2;
static const int FrameSize = WIDTH * HEIGHT * BYTESIZE;

unsigned char FrameData[FrameSize];
unsigned char FrameDataInprocessing[FrameSize] = {0};

// Init one detector
Detector* detector = new Detector();

// Definition of a ring buffer
RingBufferStruct Buffer(FrameSize, BufferSize);

bool InputDataToBuffer(RingBufferStruct* buffer)
{
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while ((buffer->write_position + 1) % BufferSize == buffer->read_position)
	{
		std::cout << "Producer is waiting for an empty slot...\n";
		buffer->buffer_not_full.wait(lock);
	}

	if (GetOneFrameFromNetwork(FrameData) == false)
	{
		buffer->finish_flag = true;
		return false;
	}
	memcpy(buffer->item_buffer + buffer->write_position * FrameSize, FrameData, FrameSize * sizeof(unsigned char));

	buffer->write_position++;

	if (buffer->write_position == BufferSize)
		buffer->write_position = 0;

	buffer->buffer_not_empty.notify_all();
	lock.unlock();
	return true;
}

bool DetectTarget(RingBufferStruct* buffer)
{
	std::unique_lock<std::mutex> lock(buffer->bufferMutex);
	while (buffer->write_position == buffer->read_position)
	{
		if (buffer->finish_flag == true)
			return false;

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
	return true;
}


void InputDataTask()
{
	// Receiving data from network
	while (true)
	{
		if(InputDataToBuffer(&Buffer) == false)
			break;
		Sleep(1);
	}
}

void DetectTask()
{
	// Detecting target
	while (true)
	{
		if (DetectTarget(&Buffer) == false)
			break;
	}
}

void InitBuffer(RingBufferStruct* buffer)
{
	buffer->write_position = 0;
	buffer->read_position = 0;
}


int main(int argc, char* argv[])
{
	// Init CUDA device
	auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
		// Init Network socket
		InitNetworks();

		// Init Detector Space
		detector->InitSpace();

		// Init ring buffer
		InitBuffer(&Buffer);

		std::thread InputDataThread(InputDataTask);
		std::thread DetectorThread(DetectTask);

		InputDataThread.join();
		DetectorThread.join();

		// Destroy Network
		DestroyNetWork();

		//		Validation validation;
		//		validation.InitValidationData("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
		//		validation.VailidationAll();

		//		DetectorValidation visualEffectValidator;
		//		visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
		//		visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_2.bin");
		//		visualEffectValidator.VailidationAll();
	}

	// Destroy detector
	delete detector;

	// Release Cuda device
	CUDAInit::cudaDeviceRelease();

	system("Pause");
	return 0;
}
